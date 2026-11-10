"""Isolated starting conditions and expert task sequences."""

from dataclasses import dataclass, field

import numpy as np

from .primitives import PRIMITIVES


@dataclass(frozen=True)
class Step:
    skill: str
    args: dict = field(default_factory=dict)

    def execute(self, arm):
        return arm.run(PRIMITIVES[self.skill].build(arm, **self.args))


def step(skill, **args):
    return Step(skill, args)


@dataclass(frozen=True)
class Example:
    scene: str
    call: Step
    setup: tuple = ()
    stage: tuple | None = None
    joints: tuple = ()


def grasp(obj):
    return (step("hover_over", obj=obj), step("descend_to_grasp", obj=obj), step("close_gripper", obj=obj))


def lift(obj, height=.11):
    return (*grasp(obj), step("raise_by", height=height, held=obj))


def stage_push(obj, direction=(1, 0), height=None):
    return (step("close_gripper"), step("move_behind", obj=obj, direction=direction),
            step("descend_behind", obj=obj, height=height))


def tabletop(skill, setup=(), stage=None, **args):
    return Example("tabletop", step(skill, **args), setup, stage)


def fixture(skill, obj, setup=(), stage=None, joints=(), **args):
    return Example("fixtures", step(skill, obj=obj, **args), setup, stage, joints)


def assembly(skill, obj, setup=(), stage=None, **args):
    return Example("assembly", step(skill, obj=obj, **args), setup, stage)


RED = lift("red")
PEG = (*lift("peg"), step("align_to_socket", obj="peg", socket="socket"))

EXAMPLES = {
    "hover_over": tabletop("hover_over", obj="blue"),
    "descend_to_grasp": tabletop("descend_to_grasp", grasp("blue")[:1], obj="blue"),
    "close_gripper": tabletop("close_gripper", grasp("blue")[:2], obj="blue"),
    "open_gripper": tabletop("open_gripper", grasp("blue"), obj="blue"),
    "raise_by": tabletop("raise_by", grasp("blue"), height=.13, held="blue"),
    "lower_to": tabletop("lower_to", RED, height=.895, held="red"),
    "move_behind": tabletop("move_behind", (step("close_gripper"),), obj="blue", direction=[1, 0]),
    "descend_behind": tabletop("descend_behind", stage_push("blue")[:2], obj="blue"),
    "push_along": tabletop("push_along", stage_push("blue"), obj="blue", direction=[1, 0], distance=.078),
    "yaw_in_hand": tabletop("yaw_in_hand", RED, obj="red", angle=.68),
    "tilt_in_hand": tabletop("tilt_in_hand", RED, obj="red", angle=.42),
    "drag_on_table": tabletop("drag_on_table", grasp("red"), obj="red", xy=[-.15, -.025]),
    "seat_on_support": tabletop("seat_on_support",
                                (*RED, step("hover_over", xy=[.17, -.13], held="red")),
                                obj="red", support="pedestal", half_height=.019),
    "push_to_stop": tabletop("push_to_stop", stage_push("green", (0, 1)) + (
        step("push_along", obj="green", direction=[0, 1], distance=.145),
        *stage_push("green"),
    ), obj="green", stop="stop"),
    "topple": tabletop("topple", stage_push("tower", (0, 1), height=.893), obj="tower"),
    "roll_cylinder": tabletop("roll_cylinder", stage_push("roller"), obj="roller", distance=.085),
    "sweep_arc": tabletop("sweep_arc", stage_push("blue"), obj="blue", radius=.137, angle=.95),
    "slide_in_channel": tabletop("slide_in_channel", stage_push("puck"), obj="puck", end_x=.059),
    "side_approach": fixture("side_approach", "drawer", stage=(-.01, -.13, .902)),
    "retract_from_fixture": fixture("retract_from_fixture", "drawer", grasp("drawer")[:2]),
    "clear_obstacle": tabletop("clear_obstacle",
                              (*RED, step("hover_over", xy=[-.018, -.13], held="red", height=.10)),
                              obj="red", xy=[.135, -.13]),
    "press_button": fixture("press_button", "button", (step("close_gripper"),), stage=(.02, -.24, .936)),
    "pull_drawer": fixture("pull_drawer", "drawer", grasp("drawer")),
    "push_drawer": fixture("push_drawer", "drawer", (step("close_gripper"),),
                           stage=(-.035, -.13, .902), joints=(("drawer", .095),)),
    "slide_slider": fixture("slide_slider", "slider", (step("close_gripper"),), stage=(.015, .12, .848)),
    "lift_latch": fixture("lift_latch", "latch", grasp("latch")),
    "turn_dial": fixture("turn_dial", "dial", grasp("dial")),
    "swing_gate": fixture("swing_gate", "gate", grasp("gate")),
    "flip_toggle": fixture("flip_toggle", "toggle", grasp("toggle")),
    "align_to_socket": assembly("align_to_socket", "peg", lift("peg"), socket="socket"),
    "insert_peg": assembly("insert_peg", "peg", PEG, socket="socket"),
    "extract_peg": assembly("extract_peg", "peg", (*PEG, step("insert_peg", obj="peg", socket="socket")),
                            socket="socket"),
    "slide_key_in": assembly("slide_key_in", "key", lift("key", height=.009), stop="guide_stop"),
    "seat_press_fit_pin": assembly("seat_press_fit_pin", "pin", (step("close_gripper"),), stage=(.15, .17, .94)),
    "twist_in_socket": assembly("twist_in_socket", "plug", grasp("plug")),
}


def prepare_example(arm, example):
    for name, position in example.joints:
        arm.env.sim.data.set_joint_qpos(name, position)
    arm.env.sim.forward()
    reset_state = arm.env.sim.get_state().flatten()
    actions = []
    setup = [step("open_gripper"), *example.setup]
    # Side-contact tests can first position the open hand and then close it.
    if example.stage is not None:
        result = setup.pop(0).execute(arm)
        actions.extend(result.actions)
        if not result.success:
            return reset_state, actions, "open_gripper"
        result = arm.stage(example.stage)
        actions.extend(result.actions)
        if not result.success:
            return reset_state, actions, "stage"
    for call in setup:
        result = call.execute(arm)
        actions.extend(result.actions)
        if not result.success:
            return reset_state, actions, call.skill
    return reset_state, actions, None
