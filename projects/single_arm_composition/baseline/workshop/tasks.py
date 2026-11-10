"""Full tasks call the same primitive interface used by standalone trials."""

from dataclasses import dataclass

import numpy as np

from .examples import grasp, lift, step


@dataclass(frozen=True)
class Task:
    scene: str
    goals: tuple
    plan: tuple
    held_out: bool = False

    def succeeded(self, arm):
        return all(goal_satisfied(arm, goal) for goal in self.goals)


def goal_satisfied(arm, goal):
    obj = goal["object"]
    if "joint" in goal:
        return abs(arm.joint(obj) - goal["joint"]) <= goal["tolerance"]
    position = arm.location(obj)
    if (arm.holding(obj) or np.linalg.norm(position[:2] - goal["xy"]) > goal["xy_tolerance"]
            or abs(position[2] - goal["z"]) > goal["z_tolerance"]):
        return False
    if "support" in goal and not arm.contact(obj, goal["support"]):
        return False
    return "upright" not in goal or arm.orientation(obj).magnitude() <= goal["upright"]


STATIONS = {
    "pedestal": ((.17, -.13), .852, .025),
    "mat": ((.17, .12), .802, .025),
    "buffer": ((-.17, .012), .802, .025),
}


def destination(obj, xy, z, xy_tolerance=.014, z_tolerance=.012, **checks):
    return dict(object=obj, xy=xy, z=z, xy_tolerance=xy_tolerance, z_tolerance=z_tolerance, **checks)


def joint_goal(obj, joint, tolerance):
    return dict(object=obj, joint=joint, tolerance=tolerance)


def sorting(assignments, held_out=False):
    goals, plan = [], []
    for obj, station in assignments:
        xy, surface, tolerance = STATIONS[station]
        half_height = .016 if obj == "green" else .019
        goals.append(destination(obj, xy, surface + half_height, xy_tolerance=tolerance, support=station))
        plan.extend((*lift(obj), step("hover_over", xy=xy, held=obj),
                     step("seat_on_support", obj=obj, support=station, half_height=half_height),
                     step("open_gripper", obj=obj), step("raise_by", height=.13)))
    return Task("tabletop", tuple(goals), tuple(plan), held_out)


def fixtures(operations, held_out=False):
    goals, plan = [], []
    for obj, skill, target, tolerance in operations:
        goals.append(joint_goal(obj, target, tolerance))
        plan.extend((*grasp(obj), step(skill, obj=obj, target=target),
                     step("open_gripper", obj=obj), step("raise_by", height=.10)))
    return Task("fixtures", tuple(goals), tuple(plan), held_out)


PEG = (
    *lift("peg"),
    step("align_to_socket", obj="peg", socket="socket"),
    step("insert_peg", obj="peg", socket="socket"),
    step("open_gripper", obj="peg"),
    step("raise_by", height=.13),
)
KEY = (
    *lift("key", height=.009),
    step("slide_key_in", obj="key", stop="guide_stop"),
    step("open_gripper", obj="key"),
    step("raise_by", height=.13),
)
PEG_GOAL = destination("peg", (.14, -.13), .845, xy_tolerance=.010, upright=.20)
KEY_GOAL = destination("key", (.037, .125), .841, xy_tolerance=.019, upright=.22)
LATCH = ("latch", "lift_latch", .047, .013)
DRAWER = ("drawer", "pull_drawer", .093, .018)
DIAL = ("dial", "turn_dial", .72, .17)
GATE = ("gate", "swing_gate", .83, .17)
TOGGLE = ("toggle", "flip_toggle", .64, .18)


TASKS = {
    "sort_red_blue": sorting((("red", "pedestal"), ("blue", "mat"))),
    "sort_green_red": sorting((("green", "buffer"), ("red", "mat"))),
    "sort_blue_green": sorting((("blue", "buffer"), ("green", "pedestal"))),
    "sort_three": sorting((("green", "mat"), ("blue", "pedestal"), ("red", "buffer")), held_out=True),
    "carry_over_barrier": Task("tabletop", (destination("green", (.17, -.13), .868),), (
        *lift("green", height=.085),
        step("hover_over", xy=[-.018, -.13], held="green", height=.10),
        step("clear_obstacle", obj="green", xy=[.17, -.13]),
        step("seat_on_support", obj="green", support="pedestal", half_height=.016),
        step("open_gripper", obj="green"),
        step("raise_by", height=.13),
    )),
    "stack_blue_on_red": Task("tabletop", (
        destination("red", (.17, -.13), .871, support="pedestal"),
        destination("blue", (.17, -.13), .909, support="red"),
    ), (
        *sorting((("red", "pedestal"),)).plan,
        *lift("blue"),
        step("hover_over", xy=[.17, -.13], held="blue", height=.22),
        step("seat_on_support", obj="blue", support="red", half_height=.019),
        step("open_gripper", obj="blue"),
        step("raise_by", height=.13),
    ), held_out=True),
    "latch_and_drawer": fixtures((LATCH, DRAWER)),
    "dial_and_gate": fixtures((DIAL, GATE)),
    "set_toggle": fixtures((TOGGLE,)),
    "slider_then_button": Task("fixtures", (joint_goal("slider", .060, .014),
                                           joint_goal("button", .022, .008)), (
        step("close_gripper"),
        step("move_behind", obj="slider", direction=[1, 0]),
        step("descend_behind", obj="slider"),
        step("slide_slider", obj="slider"),
        step("raise_by", height=.13),
        step("hover_over", obj="button", height=.17),
        step("press_button", obj="button"),
    )),
    "dial_latch_drawer": fixtures((DIAL, LATCH, DRAWER), held_out=True),
    "fit_peg": Task("assembly", (PEG_GOAL,), PEG),
    "fit_key": Task("assembly", (KEY_GOAL,), KEY),
    "seat_pin_and_screw": Task("assembly", (joint_goal("pin", .026, .008),
                                           joint_goal("plug", .79, .17)), (
        step("close_gripper"),
        step("hover_over", obj="pin"),
        step("seat_press_fit_pin", obj="pin"),
        step("raise_by", height=.13),
        step("open_gripper"),
        *grasp("plug"),
        step("twist_in_socket", obj="plug"),
        step("open_gripper", obj="plug"),
        step("raise_by", height=.12),
    )),
    "fit_key_and_peg": Task("assembly", (KEY_GOAL, PEG_GOAL), (*KEY, *PEG), held_out=True),
}
