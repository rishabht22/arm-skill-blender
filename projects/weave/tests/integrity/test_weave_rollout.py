import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from weave.envs.integrity import (
    learned_rollout_guard,
    scan_env_source,
    scan_rollout_log,
)


def test_guard_rejects_cached_planner_import_and_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("mplib")
    exec("def plan(): return True", module.__dict__)  # noqa: S102 - synthetic module
    monkeypatch.setitem(sys.modules, "mplib", module)

    with pytest.raises(RuntimeError, match="Planner or IK"), learned_rollout_guard():
        importlib.import_module("mplib")
    with pytest.raises(RuntimeError, match="Planner or IK"), learned_rollout_guard():
        __import__("mplib")
    with pytest.raises(RuntimeError, match="Planner or IK"), learned_rollout_guard():
        module.plan()
    assert module.plan()


def test_source_scan_checks_runtime_writes_and_constraints(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2] / "src/weave/envs"
    for name in (
        "two_arm_cell.py",
        "objects.py",
        "reset_dists.py",
        "contact_monitor.py",
    ):
        assert scan_env_source(root / name) == []

    path = tmp_path / "bad_env.py"
    path.write_text(
        "def _initialize_episode():\n"
        "    actor.set_pose(pose)\n"
        "    scene.create_drive(a, b)\n"
        "def step():\n"
        "    actor.pose = pose\n"
    )
    assert scan_env_source(path) == [
        "bad_env.py:3: create_drive",
        "bad_env.py:5: pose",
    ]


def test_rollout_log_rejects_runtime_pose_writes_and_any_grasp_constraint(
    tmp_path: Path,
) -> None:
    events = [
        {"event": "Actor.set_pose", "step": 0},
        {"event": "ManiSkillScene.create_drive", "step": 0},
        {"event": "Actor.pose", "step": 3},
        {"event": "control_step", "step": 3},
    ]
    trace = tmp_path / "rollout.jsonl"
    trace.write_text("".join(json.dumps(event) + "\n" for event in events))
    assert scan_rollout_log(trace) == events[1:3]
