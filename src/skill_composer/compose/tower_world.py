"""Execute dynamically composed three-block goals in the MuJoCo workbench."""

from dataclasses import asdict
import time

import mujoco
import numpy as np

from ..cell.experiment import camera
from ..cell.program import Node
from ..cell.world import AssemblyCell
from .rollout import _local_reach
from .tower_domain import TowerState, candidates, shortest_plan


LAYOUTS = ("direct", "relay", "relay-swapped", "occupied-dock")
BLOCKS = ("part0", "part1", "part2")
COLORS = dict(zip(BLOCKS, ("red", "blue", "yellow")))
BLOCK_HEIGHT = 0.048
XY_TOLERANCE = 0.021
Z_TOLERANCE = 0.014


def _make_cell(layout, seed):
    start_arms = (1, 1, 1) if layout == "direct" else (0, 1, 0)
    geometry = [Node("pick", arm, obj=block) for arm, block in zip(start_arms, BLOCKS)]
    geometry.extend([
        Node("place", 1, obj=BLOCKS[0], target="base"),
        Node("place", 0, obj=BLOCKS[0], target="dock0"),
        Node("place", 0, obj=BLOCKS[0], target="dock1"),
    ])
    cell = AssemblyCell(geometry, seed=seed)
    # Geometry declarations carry no actions or dependencies into the policy.
    cell.nodes = []
    cell.status = np.empty(0, dtype=int)

    sites = {f"supply{i}": cell.data.body(block).xpos.copy() for i, block in enumerate(BLOCKS)}
    # The private dock is a marked area of the table, so it can be placed
    # without changing the compiled static collision geometry.
    private = np.array([-0.10, -0.36, 0.024])
    dock = cell.model.geom("dock1").id
    cell.model.geom_contype[dock] = 0
    cell.model.geom_conaffinity[dock] = 0
    cell.model.geom_rgba[dock, 3] = 0
    marker = cell.model.site("dock1_marker").id
    cell.model.site_pos[marker] = (*private[:2], 0.001)
    cell.model.site_rgba[marker] = (0.85, 0.5, 0.18, 0.65)
    shared = cell.docks["dock0"]
    sites["dock0"], sites["dock1"] = (private, shared) if layout == "relay-swapped" else (shared, private)
    sites["base"] = cell.goals["base"].copy()
    sites["base"][2] = 0.030  # Top of the plate plus half a block.
    if layout == "occupied-dock":
        joint = cell.model.joint(f"{BLOCKS[2]}_free")
        index = int(joint.qposadr.item())
        cell.data.qpos[index:index + 3] = shared
        velocity_index = int(joint.dofadr.item())
        cell.data.qvel[velocity_index:velocity_index + 6] = 0
    mujoco.mj_forward(cell.model, cell.data)
    if layout == "occupied-dock":
        for _ in range(150):
            mujoco.mj_step(cell.model, cell.data)
    return cell, sites


def _observe(cell, sites, reach, goals):
    holders = {block: arm for arm, block in enumerate(cell.held) if block is not None}
    centers = {block: cell.data.body(block).xpos.copy() for block in BLOCKS if block not in holders}
    positions = {}
    for block, center in centers.items():
        supports = dict(sites)
        supports.update({
            other: lower + [0, 0, BLOCK_HEIGHT]
            for other, lower in centers.items() if other != block
        })
        matches = []
        for support, expected in supports.items():
            offset = center - expected
            if np.linalg.norm(offset[:2]) < XY_TOLERANCE and abs(offset[2]) < Z_TOLERANCE:
                matches.append((float(np.linalg.norm(offset)), support))
        if not matches:
            raise ValueError(f"Cannot locate {COLORS[block]} block on any support at {center.round(3).tolist()}")
        positions[block] = min(matches)[1]
    if len(set(positions.values())) != len(positions):
        raise ValueError("Two blocks were observed on the same support")
    return TowerState(positions=positions, holders=holders, reach=reach, goals=goals)


def _place_pose(cell, sites, target):
    if target in sites:
        return sites[target]
    return cell.data.body(target).xpos + [0, 0, BLOCK_HEIGHT + 0.001]


def _configure_viewer(cell, sites):
    from mujoco import viewer

    window = viewer.launch_passive(cell.model, cell.data)
    view = camera()
    with window.lock():
        window.cam.lookat[:] = view.lookat
        window.cam.distance = view.distance
        window.cam.azimuth = view.azimuth
        window.cam.elevation = view.elevation
        for index, (name, point) in enumerate(sites.items()):
            geom = window.user_scn.geoms[index]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                               point + [0, 0, 0.1], np.eye(3).reshape(-1), np.ones(4))
            geom.label = name
        window.user_scn.ngeom = len(sites)
    return window


def rollout(layout="relay", seed=0, chooser=None, viewer=False, max_calls=24, order=BLOCKS):
    """Run one candidate chooser against measured support and possession state."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown tower layout {layout!r}. Choose one of {LAYOUTS}.")
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    order = tuple(order)
    if len(order) != len(BLOCKS) or set(order) != set(BLOCKS):
        raise ValueError(f"order must be a permutation of {BLOCKS}")
    goals = dict(zip(order, ("base", *order[:-1])))
    cell, sites = _make_cell(layout, seed)
    reach = _local_reach(cell, sites)
    initial = _observe(cell, sites, reach, goals)
    state = initial
    calls = []
    failure = None
    window = None

    def step():
        start = time.monotonic()
        if window is not None and not window.is_running():
            return False
        cell.step()
        if window is not None:
            window.sync()
            time.sleep(max(0.0, cell.dt - (time.monotonic() - start)))
        return True

    try:
        if viewer:
            window = _configure_viewer(cell, sites)
            print("Goal tower (bottom to top): " + " → ".join(COLORS[block] for block in order), flush=True)
        for _ in range(max_calls):
            if state.complete:
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

            call = {**asdict(action), "time_s": round(cell.time, 3),
                    "candidates": [asdict(candidate) for candidate in legal]}
            calls.append(call)
            if viewer:
                target_label = COLORS.get(action.target, action.target)
                print(f"Arm {action.arm}: {action.skill} {COLORS[action.obj]} at {target_label}", flush=True)

            # AssemblyCell consumes a concrete pose. Support choice and arm
            # selection are made before this conversion to a motor target.
            executor_skill = "pick" if action.skill == "pick" else "place"
            target = "base"
            if executor_skill == "place":
                target = f"placement_{len(calls)}"
                cell.goals[target] = _place_pose(cell, sites, action.target).copy()
            node = Node(executor_skill, action.arm, obj=action.obj, target=target)
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
            if executor_skill == "place" and not all(step() for _ in range(12)):
                failure = "Viewer closed"
                break
            try:
                state = _observe(cell, sites, reach, goals)
            except ValueError as error:
                failure = str(error)
                break
            call["positions_after"] = dict(state.positions)
            call["holders_after"] = dict(state.holders)
    finally:
        if window is not None:
            window.close()

    success = failure is None and state.complete
    return {
        "layout": layout,
        "seed": seed,
        "controller": "shortest_plan" if chooser is None else "learned",
        "goal": goals,
        "order": list(order),
        "colors": COLORS,
        "reach": reach,
        "initial_positions": dict(initial.positions),
        "final_positions": dict(state.positions),
        "final_holders": dict(state.holders),
        "final_xyz": {block: cell.data.body(block).xpos.tolist() for block in BLOCKS},
        "success": success,
        "failure": failure if failure is not None or success else "Goal not achieved within skill budget",
        "sampled_arm_contact_frames": cell.arm_contacts,
        "time_s": round(cell.time, 3),
        "skill_calls": calls,
    }
