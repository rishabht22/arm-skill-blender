"""Execute goal-conditioned skill choices in the existing two-arm MuJoCo cell."""

from dataclasses import asdict
import time

import mujoco
import numpy as np

from ..cell.experiment import camera
from ..cell.program import Node
from ..cell.world import AssemblyCell
from .domain import State, candidates, shortest_plan

LAYOUTS = ("direct", "relay", "relay-swapped")
PAYLOAD = "part0"
GOAL_TOLERANCE = 0.025


def _make_cell(layout, seed):
    # AssemblyCell builds geometry from declarations. These have no edges and
    # are discarded before the first policy observation or physical dispatch.
    geometry = [
        Node("pick", 0, obj=PAYLOAD),
        Node("place", 1, obj=PAYLOAD, target="goal0", stack=True),
        Node("place", 0, obj=PAYLOAD, target="dock0"),
        Node("place", 0, obj=PAYLOAD, target="dock1"),
    ]
    cell = AssemblyCell(geometry, seed=seed)
    cell.nodes = []
    cell.status = np.empty(0, dtype=int)

    source = cell.data.body(PAYLOAD).xpos.copy()
    # Use the table itself as the private staging surface. Hide and disable the
    # stock raised dock rather than moving a compiled static collision geom.
    # The moved site is only a visual marker for the new pad.
    cell.docks["dock1"][1:] = (-0.22, source[2])
    dock = cell.model.geom("dock1").id
    cell.model.geom_contype[dock] = 0
    cell.model.geom_conaffinity[dock] = 0
    cell.model.geom_rgba[dock, 3] = 0
    marker = cell.model.site("dock1_marker").id
    cell.model.site_pos[marker, 1:] = (-0.22, 0.001)
    cell.model.site_rgba[marker] = (0.85, 0.5, 0.18, 0.65)
    if layout == "relay-swapped":
        cell.docks["dock0"], cell.docks["dock1"] = cell.docks["dock1"], cell.docks["dock0"]

    sites = {
        "source": source,
        "dock0": cell.docks["dock0"],
        "dock1": cell.docks["dock1"],
        "goal": cell.goals["goal0"],
    }
    cell.goals["source"] = source
    if layout == "direct":
        position = sites["dock0"].copy()
        position[:2] += (source[:2] - np.array([-0.26, -0.22])) / 2
        joint = cell.model.joint(f"{PAYLOAD}_free")
        index = int(joint.qposadr.item())
        cell.data.qpos[index:index + 3] = position
        velocity_index = int(joint.dofadr.item())
        cell.data.qvel[velocity_index:velocity_index + 6] = 0
    mujoco.mj_forward(cell.model, cell.data)
    for _ in range(150):
        mujoco.mj_step(cell.model, cell.data)
    return cell, sites


def _local_reach(cell, sites):
    """Check each arm and site independently with the executor's IK waypoints."""
    reach = {}
    for name, point in sites.items():
        available = []
        for arm, cruise in enumerate((0.37, 0.45)):
            joints = cell.data.qpos[arm * 9:(arm + 1) * 9]
            try:
                approach = cell.solve(arm, [*point[:2], cruise], joints)
                cell.solve(arm, point, approach)
            except RuntimeError:
                continue
            available.append(arm)
        reach[name] = tuple(available)
    return reach


def _observe(cell, sites, reach, goal):
    if PAYLOAD in cell.held:
        return State(location=None, holder=cell.held.index(PAYLOAD), reach=reach, goal=goal)
    position = cell.data.body(PAYLOAD).xpos
    distances = {
        name: float(np.linalg.norm(position - target))
        for name, target in sites.items()
    }
    location = min(distances, key=distances.get)
    if distances[location] >= GOAL_TOLERANCE:
        raise ValueError(f"Payload outside all modeled sites at {position.round(3).tolist()}")
    return State(location=location, holder=None, reach=reach, goal=goal)


def rollout(layout="relay", seed=0, chooser=None, viewer=False, max_calls=12, goal="goal"):
    """Run a learned chooser or the shortest-plan reference on the same cell.

    A chooser receives just the current semantic state. Each invocation adds
    one executor call, which runs to completion before the next observation.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}. Choose one of {LAYOUTS}.")
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    if goal not in ("goal", "source"):
        raise ValueError("goal must be 'goal' or 'source'")

    cell, sites = _make_cell(layout, seed)
    reach = _local_reach(cell, sites)
    initial = _observe(cell, sites, reach, goal)
    calls = []
    failure = None
    window = None

    def step():
        started = time.monotonic()
        if window is not None and not window.is_running():
            return False
        cell.step()
        if window is not None:
            window.sync()
            time.sleep(max(0.0, cell.dt - (time.monotonic() - started)))
        return True

    try:
        if viewer:
            from mujoco import viewer as mj_viewer

            window = mj_viewer.launch_passive(cell.model, cell.data)
            view = camera()
            with window.lock():
                window.cam.lookat[:] = view.lookat
                window.cam.distance = view.distance
                window.cam.azimuth = view.azimuth
                window.cam.elevation = view.elevation
                for index, (name, point) in enumerate(sites.items()):
                    label = window.user_scn.geoms[index]
                    position = point + np.array([0, 0, 0.08])
                    mujoco.mjv_initGeom(label, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                                       position, np.eye(3).reshape(-1), np.ones(4))
                    label.label = name
                window.user_scn.ngeom = len(sites)

        state = initial
        for _ in range(max_calls):
            if state.location == state.goal:
                break
            if window is not None and not window.is_running():
                failure = "Viewer closed"
                break
            try:
                action = shortest_plan(state)[0] if chooser is None else chooser(state)
            except ValueError as error:
                failure = str(error)
                break

            legal = candidates(state)
            if action not in legal:
                failure = f"Chooser returned an infeasible call: {action}"
                break

            call = {
                **asdict(action),
                "object": PAYLOAD,
                "time_s": round(cell.time, 3),
                "candidates": [asdict(candidate) for candidate in legal],
            }
            calls.append(call)
            if viewer:
                print(f"Arm {action.arm} {action.skill} {PAYLOAD} at {action.target}", flush=True)

            target = "goal0" if action.target == "goal" else action.target
            node = Node(action.skill, action.arm, obj=PAYLOAD, target=target)
            index = len(cell.nodes)
            cell.nodes.append(node)
            cell.status = np.append(cell.status, 0)
            choices = [-1, -1]
            choices[action.arm] = index
            cell.dispatch(choices)

            while cell.active[action.arm] is not None and not cell.failures:
                if not step():
                    failure = "Viewer closed"
                    break
            if cell.failures or failure is not None or cell.status[index] != 2:
                failure = failure or (cell.failures[-1] if cell.failures else "Executor rejected call")
                break

            # Settle a released block before classifying its physical location.
            if action.skill == "place" and not all(step() for _ in range(12)):
                failure = "Viewer closed"
                break
            try:
                state = _observe(cell, sites, reach, goal)
            except ValueError as error:
                failure = str(error)
                break
            call["resulting_location"] = state.location
            call["resulting_holder"] = state.holder
    finally:
        if window is not None:
            window.close()

    position = cell.data.body(PAYLOAD).xpos.copy()
    error = float(np.linalg.norm(position - sites[goal]))
    success = (
        failure is None
        and PAYLOAD not in cell.held
        and all(index is None for index in cell.active)
        and error < GOAL_TOLERANCE
    )
    return {
        "layout": layout,
        "seed": seed,
        "controller": "shortest_plan" if chooser is None else "learned",
        "initial_location": initial.location,
        "reach": reach,
        "goal": {"relation": "on", "object": PAYLOAD, "target": goal},
        "goal_position": sites[goal].tolist(),
        "payload_position": position.tolist(),
        "goal_error_m": error,
        "success": success,
        "failure": failure if failure is not None or success else "Goal not achieved within skill budget",
        "sampled_arm_contact_frames": cell.arm_contacts,
        "time_s": round(cell.time, 3),
        "skill_calls": calls,
    }
