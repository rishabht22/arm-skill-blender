"""Goal-conditioned plans for loading, inspection, and cargo exchange."""

from dataclasses import dataclass

import numpy as np

from baseline.workshop.examples import grasp, lift, step
from baseline.workshop.tasks import destination, goal_satisfied, joint_goal
from .environment import BOLT_UP, DRAWER_OPEN, PARTS, SCENE


SLOTS = ("slot_a", "slot_b")
MODES = {"locked": (0., 0.), "unlocked": (0., BOLT_UP), "open": (DRAWER_OPEN, BOLT_UP)}


@dataclass(frozen=True)
class Job:
    placements: tuple[tuple[str, str], ...]
    finish: str = "locked"
    stocked: tuple[tuple[str, str], ...] = ()
    held_out: bool = False
    scene: str = SCENE

    def goals(self, arm):
        drawer, bolt = MODES[self.finish]
        goals = [joint_goal("drawer", drawer, .014), joint_goal("bolt", bolt, .012)]
        for obj, station in self.placements:
            center = arm.location(station)
            if station in SLOTS:
                center[0] += arm.joint("drawer") - drawer
                goals.append(destination(obj, center[:2].tolist(), float(center[2] + PARTS[obj]),
                                         support="drawer_floor", slot=station))
            else:
                height = arm.env.sim.model.geom_size[arm.env.sim.model.geom_name2id(station), 2]
                goals.append(destination(obj, center[:2].tolist(), float(center[2] + height + PARTS[obj]),
                                         support=station))
        return tuple(goals)


JOBS = {
    "pack_red_a": Job((("red", "slot_a"),)),
    "pack_red_b": Job((("red", "slot_b"),)),
    "pack_blue_a": Job((("blue", "slot_a"),)),
    "pack_blue_b": Job((("blue", "slot_b"),)),
    "pack_green_a": Job((("green", "slot_a"),)),
    "pack_green_b": Job((("green", "slot_b"),)),
    "ship_red_blue": Job((("red", "slot_a"), ("blue", "slot_b"))),
    "stage_green_blue": Job((("green", "slot_a"), ("blue", "slot_b")), finish="open"),
    "inspect_red": Job((("red", "inspection"),), stocked=(("red", "slot_a"),)),
    "inspect_blue": Job((("blue", "buffer"),), stocked=(("blue", "slot_b"),)),
    "ship_green_blue": Job((("green", "slot_a"), ("blue", "slot_b")), held_out=True),
    "ship_blue_red": Job((("blue", "slot_a"), ("red", "slot_b")), held_out=True),
    "inspect_red_and_blue": Job((("red", "inspection"), ("blue", "buffer")),
                                stocked=(("red", "slot_a"), ("blue", "slot_b")), held_out=True),
    "exchange_red_for_green": Job((("red", "inspection"), ("green", "slot_a")),
                                  stocked=(("red", "slot_a"),), held_out=True),
}


def reset_cell(env, seed, mode, stocked):
    env.rng = np.random.default_rng(seed)
    env.reset()
    drawer, bolt = MODES[mode]
    env.sim.data.set_joint_qpos("drawer", drawer)
    env.sim.data.set_joint_qpos("bolt", bolt)
    env.sim.forward()
    for obj, slot in stocked:
        geom = env.sim.model.geom_name2id(slot)
        center = env.sim.data.geom_xpos[geom].copy()
        center[2] += PARTS[obj] + .0008
        env.sim.data.set_joint_qpos(obj, [*center, 1, 0, 0, 0])
    env.sim.forward()


def job_start(job, rng, mode=None):
    mode = mode or rng.choice(tuple(MODES))
    stocked = list(job.stocked)
    cargo = [(obj, station) for obj, station in job.placements if station in SLOTS]
    # Some two-part jobs start with one correct part already loaded.
    if not stocked and len(cargo) == 2 and rng.random() < .35:
        stocked.append(cargo[int(rng.integers(2))])
    return str(mode), tuple(stocked)


def satisfied(arm, goal):
    if "slot" in goal:
        center = arm.location(goal["slot"])
        goal = {**goal, "xy": center[:2], "z": center[2] + PARTS[goal["object"]]}
    return goal_satisfied(arm, goal)


def operate(obj, skill, target):
    return (*grasp(obj), step(skill, obj=obj, target=target),
            step("open_gripper", obj=obj), step("raise_by", height=.105))


def plan_job(arm, goals, rng):
    drawer_goal, bolt_goal = goals[:2]
    pending = [goal for goal in goals[2:] if not satisfied(arm, goal)]
    if pending or drawer_goal["joint"] == DRAWER_OPEN:
        if arm.joint("drawer") < DRAWER_OPEN - .014:
            if arm.joint("bolt") < BOLT_UP - .012:
                yield from operate("bolt", "set_bolt", BOLT_UP)
            yield from operate("drawer", "slide_drawer", DRAWER_OPEN)

        # Unload first to free any occupied slot required by the job.
        ordered = sorted(rng.permutation(len(pending)), key=lambda i: "slot" in pending[i])
        for i in ordered:
            goal = pending[i]
            obj = goal["object"]
            if arm.contact(obj, "drawer_floor"):
                slot = min(SLOTS, key=lambda name: np.linalg.norm(arm.location(name)[:2] - arm.location(obj)[:2]))
                yield from grasp(obj)
                yield step("lift_from_slot", obj=obj, slot=slot)
            else:
                yield from lift(obj, height=.16)
            if "slot" in goal:
                yield step("center_over_slot", obj=obj, slot=goal["slot"])
                yield step("seat_in_slot", obj=obj, slot=goal["slot"])
            else:
                yield step("hover_over", held=obj, xy=goal["xy"])
                yield step("seat_on_support", obj=obj, support=goal["support"], half_height=PARTS[obj])
            yield step("open_gripper", obj=obj)
            yield step("raise_by", height=.13)

    if not satisfied(arm, drawer_goal):
        yield from operate("drawer", "slide_drawer", drawer_goal["joint"])
    if not satisfied(arm, bolt_goal):
        yield from operate("bolt", "set_bolt", bolt_goal["joint"])
