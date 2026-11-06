"""Rollout integrity guards and source checks."""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

_MOTION_WRITES = {
    "set_pose",
    "set_root_pose",
    "set_qpos",
    "set_state",
    "set_state_dict",
    "set_sim_state",
}
_CONSTRAINT_WRITES = {"create_drive", "create_constraint", "create_weld"}
_BLOCKED_MODULE_PARTS = {
    "mplib",
    "ompl",
    "ik",
    "kinematics",
    "fast_kinematics",
    "motionplanner",
    "motionplanning",
    "motion_planning",
    "planner",
    "pybullet_planning",
    "pytorch_kinematics",
}


def _planner_module(name: str) -> bool:
    return bool(_BLOCKED_MODULE_PARTS.intersection(name.split(".")))


@contextmanager
def learned_rollout_guard() -> Iterator[None]:
    """Reject imports and calls into planner and IK modules, including cached modules."""
    import_builtin = builtins.__import__
    import_module = importlib.import_module
    previous_profile = sys.getprofile()

    def deny(name: str) -> None:
        if _planner_module(name):
            raise RuntimeError(f"Planner or IK module in learned rollout: {name}")

    def checked_import(name: str, globals=None, locals=None, fromlist=(), level=0):
        resolved = name
        if level and globals and globals.get("__package__"):
            resolved = importlib.util.resolve_name(
                "." * level + name, globals["__package__"]
            )
        deny(resolved)
        for item in fromlist or ():
            deny(f"{resolved}.{item}")
        return import_builtin(name, globals, locals, fromlist, level)

    def checked_import_module(name: str, package=None):
        deny(importlib.util.resolve_name(name, package))
        return import_module(name, package)

    def profile(frame, event, arg):
        if event == "call":
            deny(frame.f_globals.get("__name__", ""))
        elif event == "c_call":
            deny(getattr(arg, "__module__", None) or "")
        if previous_profile is not None:
            previous_profile(frame, event, arg)

    with (
        patch.object(builtins, "__import__", checked_import),
        patch.object(importlib, "import_module", checked_import_module),
    ):
        sys.setprofile(profile)
        try:
            yield
        finally:
            sys.setprofile(previous_profile)


def scan_env_source(path: Path) -> list[str]:
    """Only reset code may write pose or joint state, and grasp constraints are forbidden."""
    violations = []

    class Scanner(ast.NodeVisitor):
        function = ""

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            previous = self.function
            self.function = node.name
            self.generic_visit(node)
            self.function = previous

        def visit_Attribute(self, node: ast.Attribute) -> None:
            unsafe_write = node.attr in _MOTION_WRITES or (
                node.attr in {"pose", "root_pose", "qpos"}
                and isinstance(node.ctx, ast.Store)
            )
            if node.attr in _CONSTRAINT_WRITES or (
                unsafe_write and self.function != "_initialize_episode"
            ):
                violations.append(f"{path.name}:{node.lineno}: {node.attr}")
            self.generic_visit(node)

    Scanner().visit(ast.parse(path.read_text(), filename=str(path)))
    return violations


def scan_rollout_log(path: Path) -> list[dict[str, Any]]:
    """Return prohibited calls found in the simulator write trace."""
    violations = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            operation = row["event"].rsplit(".", 1)[-1]
            if operation in _CONSTRAINT_WRITES or (
                operation in _MOTION_WRITES | {"pose", "root_pose", "qpos"}
                and row["step"] > 0
            ):
                violations.append(row)
    return violations


@contextmanager
def physics_write_guard(events: list[dict[str, Any]], step: int) -> Iterator[None]:
    """Instrument ManiSkill setters while stepping an already initialized scene."""
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs import Actor, Articulation, Link

    def forbidden(name: str):
        def reject(*args, **kwargs):
            events.append({"event": name, "step": step})
            raise RuntimeError(
                f"Physics state write during control step {step}: {name}"
            )

        return reject

    with ExitStack() as stack:
        for cls in (Actor, Articulation, Link, ManiSkillScene):
            for name in _MOTION_WRITES | _CONSTRAINT_WRITES:
                if hasattr(cls, name):
                    stack.enter_context(
                        patch.object(cls, name, forbidden(f"{cls.__name__}.{name}"))
                    )
            for name in ("pose", "root_pose", "qpos"):
                prop = getattr(cls, name, None)
                if isinstance(prop, property) and prop.fset is not None:
                    guarded = prop.setter(forbidden(f"{cls.__name__}.{name}"))
                    stack.enter_context(patch.object(cls, name, guarded))
        yield
