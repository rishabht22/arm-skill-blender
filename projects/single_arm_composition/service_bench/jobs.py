"""Installation, replacement, and decommissioning recipes for the service bench."""

from dataclasses import dataclass

import numpy as np

from baseline.workshop.control import WorkshopArm
from baseline.workshop.examples import grasp, lift, step
from baseline.workshop.tasks import destination, goal_satisfied, joint_goal
from .environment import CONTACT_PART, KEEPER_OPEN, SENSORS, STEM_LENGTH, TONGUE_DEPTH
from .skills import aligned, cartridge_pose, sensor_pose, top


PORT = "sensor_floor"
RAIL = "rail_bed"


@dataclass(frozen=True)
class Job:
    placements: tuple[tuple[str, str], ...]
    installed: tuple[str, ...] = ()
    keeper: float | None = None
    held_out: bool = False
    inspection_yaw: tuple[tuple[str, float], ...] = ()

    def goals(self, arm):
        goals = []
        if self.keeper is not None:
            goals.append(joint_goal("keeper", self.keeper, .10))
        for obj, station in self.placements:
            if station == PORT:
                center = sensor_pose(arm)
            elif station == RAIL:
                center = cartridge_pose(arm)
            else:
                height = TONGUE_DEPTH if obj == "cartridge" else STEM_LENGTH
                center = [*arm.location(station)[:2], top(arm, station) + height]
            goal = destination(obj, list(map(float, center[:2])), float(center[2]), support=station,
                               xy_tolerance=.014, z_tolerance=.008)
            yaw = dict(self.inspection_yaw).get(obj)
            if yaw is not None:
                goal.update(yaw=yaw, yaw_tolerance=.20)
            goals.append(goal)
        return tuple(goals)


JOBS = {
    "install_sensor": Job((("sensor", PORT),), keeper=0),
    "install_spare": Job((("spare", PORT),), keeper=0),
    "install_power": Job((("cartridge", RAIL),)),
    "remove_power": Job((("cartridge", "cartridge_nest"),), installed=("cartridge",)),
    "quarantine_sensor": Job((("sensor", "quarantine"),), installed=("sensor",), keeper=KEEPER_OPEN),
    "return_spare": Job((("spare", "sensor_nest"),), installed=("spare",), keeper=KEEPER_OPEN),
    "commission_panel": Job((("sensor", PORT), ("cartridge", RAIL)), keeper=0),
    "inspect_sensor_key": Job((("sensor", "quarantine"),), keeper=0,
                              inspection_yaw=(("sensor", -.70),)),
    "inspect_spare_key": Job((("spare", "sensor_nest"),), keeper=0,
                             inspection_yaw=(("spare", -.70),)),
    "service_power_with_sensor": Job((("sensor", PORT), ("cartridge", RAIL)),
                                     installed=("sensor",), keeper=0),
    "replace_sensor": Job((("sensor", "quarantine"), ("spare", PORT)),
                          installed=("sensor",), keeper=0, held_out=True),
    "recommission_panel": Job((("sensor", "quarantine"), ("spare", PORT), ("cartridge", RAIL)),
                              installed=("sensor",), keeper=0, held_out=True),
    "decommission_panel": Job((("sensor", "quarantine"), ("cartridge", "cartridge_nest")),
                              installed=("sensor", "cartridge"), keeper=KEEPER_OPEN, held_out=True),
    "swap_spare_for_original": Job((("spare", "sensor_nest"), ("sensor", PORT)),
                                   installed=("spare",), keeper=0, held_out=True),
    "inspect_and_replace": Job((("sensor", "quarantine"), ("spare", PORT)), installed=("sensor",),
                               keeper=0, held_out=True, inspection_yaw=(("sensor", -.70),)),
    "decommission_power_only": Job((("sensor", PORT), ("cartridge", "cartridge_nest")),
                                    installed=("sensor", "cartridge"), keeper=0, held_out=True),
}


def reset_bench(env, seed, keeper, installed=()):
    env.rng = np.random.default_rng(seed)
    env.reset()
    env.sim.data.set_joint_qpos("keeper", keeper)
    env.sim.forward()
    arm = WorkshopArm(env)
    for obj in installed:
        position = sensor_pose(arm) if obj in SENSORS else cartridge_pose(arm)
        position[2] += .0006
        env.sim.data.set_joint_qpos(obj, [*position, 1, 0, 0, 0])
    env.sim.forward()


def satisfied(arm, goal):
    if "joint" in goal:
        return goal_satisfied(arm, goal)
    station = goal["support"]
    pose_goal = {key: value for key, value in goal.items() if key != "support"}
    obj = goal["object"]
    return (goal_satisfied(arm, pose_goal)
            and arm.contact(CONTACT_PART[obj], station)
            and (station not in (PORT, RAIL) or aligned(arm, obj))
            and ("yaw" not in goal or aligned(arm, obj, goal["yaw"], goal["yaw_tolerance"])))


def grasp_sensor(obj):
    return (step("hover_over", obj=obj), step("descend_to_sensor", obj=obj), step("close_gripper", obj=obj))


def grasp_keeper():
    return (step("approach_keeper", obj="keeper", height=.09),
            step("approach_keeper", obj="keeper", height=.003),
            step("close_gripper", obj="keeper"))


def operate_keeper(target):
    return (*grasp_keeper(), step("swing_keeper", obj="keeper", target=target),
            step("open_gripper", obj="keeper"), step("raise_by", height=.10))


def plan_job(arm, goals, rng):
    pending = [goal for goal in goals if "xy" in goal and not satisfied(arm, goal)]
    needs_port = any(goal["object"] in SENSORS and
                     (goal["support"] == PORT or arm.contact(CONTACT_PART[goal["object"]], PORT))
                     for goal in pending)
    if needs_port and abs(arm.joint("keeper") - KEEPER_OPEN) > .10:
        yield from operate_keeper(KEEPER_OPEN)

    # Preserve the replacement dependency while varying independent service order.
    ordered = sorted(rng.permutation(len(pending)),
                     key=lambda i: not (pending[i]["object"] in SENSORS and pending[i]["support"] != PORT))
    for index in ordered:
        goal = pending[index]
        obj, station = goal["object"], goal["support"]
        if arm.contact(CONTACT_PART[obj], PORT):
            yield from grasp_sensor(obj)
            yield step("extract_sensor", obj=obj)
        elif arm.contact(CONTACT_PART[obj], RAIL):
            yield from grasp(obj)
            yield step("withdraw_cartridge", obj=obj)
            yield step("raise_by", height=.14, held=obj)
        elif obj in SENSORS:
            yield from grasp_sensor(obj)
            yield step("raise_by", height=.14, held=obj)
        else:
            yield from lift(obj, height=.14)

        if station == PORT:
            yield step("align_sensor", obj=obj)
            yield step("seat_sensor", obj=obj)
        elif station == RAIL:
            yield step("align_cartridge", obj=obj)
            yield step("feed_cartridge", obj=obj)
            yield step("seat_cartridge_stop", obj=obj)
        else:
            if "yaw" in goal:
                yield step("orient_for_inspection", obj=obj, yaw=goal["yaw"])
            yield step("hover_over", held=obj, xy=goal["xy"], height=.22)
            yield step("seat_in_nest", obj=obj, nest=station)
        yield step("open_gripper", obj=obj)
        yield step("raise_by", height=.12)

    for goal in goals:
        if "joint" in goal and not satisfied(arm, goal):
            yield from operate_keeper(goal["joint"])
