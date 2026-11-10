"""Atomic motions specific to the interlocked packing cell."""

import numpy as np

from baseline.workshop.primitives import PRIMITIVES, hand_for_object
from .environment import DRAWER_OPEN, PARTS


def set_bolt(arm, obj, target):
    """Translate the grasped lock vertically along its guide."""
    dz = target - arm.joint(obj)
    return arm.line(arm.position + [0, 0, dz], held=obj, steps=26,
                    effect=lambda: abs(arm.joint(obj) - target) < .009)


def slide_drawer(arm, obj, target):
    """Translate the grasped drawer while retaining all supported cargo."""
    floor = arm.location("drawer_floor")
    cargo = {part: arm.location(part) - floor for part in PARTS if arm.contact(part, "drawer_floor")}

    def retained(tolerance):
        center = arm.location("drawer_floor")
        return all(np.linalg.norm(arm.location(part) - center - offset) < tolerance
                   for part, offset in cargo.items())

    dx = arm.joint(obj) - target
    return arm.line(arm.position + [dx, 0, 0], held=obj, steps=62,
                    guard=lambda: retained(.024),
                    effect=lambda: abs(arm.joint(obj) - target) < .011 and retained(.015)
                    and all(arm.contact(part, "drawer_floor") for part in cargo))


def center_over_slot(arm, obj, slot):
    """Transfer one held part to the clearance pose of a moving slot."""
    target = arm.location(slot) + [0, 0, .16]
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=34,
                    effect=lambda: abs(arm.joint("drawer") - DRAWER_OPEN) < .016)


def seat_in_slot(arm, obj, slot):
    """Lower a grasped part onto the exposed drawer floor at a named slot."""
    target = arm.location(slot) + [0, 0, PARTS[obj] - .006]

    def seated():
        delta = arm.location(obj) - arm.location(slot)
        return (arm.contact(obj, "drawer_floor") and np.linalg.norm(delta[:2]) < .013
                and abs(delta[2] - PARTS[obj]) < .009)

    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=35, check_pose=False,
                    effect=seated, guard=lambda: abs(arm.joint("drawer") - DRAWER_OPEN) < .016)


def lift_from_slot(arm, obj, slot):
    """Extract a held part vertically until it clears the drawer dividers."""
    origin = arm.location(slot)
    target = arm.location(obj)
    target[2] = origin[2] + .13
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=30,
                    effect=lambda: arm.location(obj)[2] > origin[2] + .113,
                    guard=lambda: abs(arm.joint("drawer") - DRAWER_OPEN) < .016)


CELL_SKILLS = {build.__name__: build for build in
               (set_bolt, slide_drawer, center_over_slot, seat_in_slot, lift_from_slot)}
BUILDERS = {**{name: primitive.build for name, primitive in PRIMITIVES.items()}, **CELL_SKILLS}


def execute(arm, call):
    return arm.run(BUILDERS[call.skill](arm, **call.args))
