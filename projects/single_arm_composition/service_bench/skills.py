"""Atomic connector service motions with measured mechanical outcomes."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from baseline.individual_skill_generation.skills import TOP_DOWN

from baseline.workshop.primitives import PRIMITIVES, follow_hinge, hand_for_object, top_grasp
from .environment import CONTACT_PART, STEM_LENGTH, TONGUE_DEPTH


def top(arm, geom):
    model = arm.env.sim.model
    return arm.location(geom)[2] + model.geom_size[model.geom_name2id(geom), 2]


def sensor_pose(arm):
    return np.array([*arm.location("sensor_floor")[:2], top(arm, "sensor_floor") + STEM_LENGTH])


def cartridge_pose(arm):
    return np.array([arm.location("rail_stop")[0] - .046,
                     arm.location("rail_bed")[1], top(arm, "rail_bed") + TONGUE_DEPTH])


def aligned(arm, obj, yaw=0., tolerance=.16):
    desired = Rotation.from_euler("z", yaw)
    return (desired * arm.orientation(obj).inv()).magnitude() < tolerance


def approach_keeper(arm, obj, height):
    """Reach the keeper handle using the nearer of its two symmetric grasps."""
    rotation = top_grasp(arm, obj)
    other = Rotation.from_euler("z", np.pi) * rotation
    nearest = min((rotation, other), key=lambda pose: (pose * arm.rotation.inv()).magnitude())
    return arm.line(arm.location(obj) + [0, 0, height], nearest, steps=28)


def swing_keeper(arm, obj, target):
    """Swing the grasped keeper around its hinge."""
    motion = follow_hinge(arm, obj, [0, 0, 1], target)
    motion.effect = lambda: abs(arm.joint(obj) - target) < .085
    return motion


def descend_to_sensor(arm, obj):
    """Approach the round grip with the jaws clear of the open keeper."""
    rotation = Rotation.from_euler("z", np.pi / 2) * TOP_DOWN
    return arm.line(arm.location(obj) + [0, 0, .003], rotation, steps=26)


def align_sensor(arm, obj):
    """Carry a connector over the keeper and align it at the keyed socket mouth."""
    target = sensor_pose(arm) + [0, 0, .067]
    rotation = arm.orientation(obj).inv() * arm.rotation
    end = hand_for_object(arm, obj, target, rotation)
    t = np.linspace(0, 1, 43)[1:]
    positions = arm.position + t[:, None] * (end - arm.position)
    positions[:, 2] += .075 * np.sin(np.pi * t)
    rotations = Slerp([0, 1], Rotation.concatenate([arm.rotation, rotation]))(t)
    return arm.path(positions, rotations, held=obj,
                    effect=lambda: aligned(arm, obj)
                    and np.linalg.norm(arm.location(obj)[:2] - target[:2]) < .005)


def seat_sensor(arm, obj):
    """Insert along the socket axis and confirm depth and bottom contact."""
    target = sensor_pose(arm)
    hand = hand_for_object(arm, obj, target - [0, 0, .005])
    return arm.line(hand, held=obj, steps=46, check_pose=False,
                    effect=lambda: aligned(arm, obj)
                    and np.linalg.norm(arm.location(obj) - target) < .008
                    and arm.contact(CONTACT_PART[obj], "sensor_floor"))


def extract_sensor(arm, obj):
    """Withdraw the grasped connector vertically until its stem clears the socket."""
    target = sensor_pose(arm) + [0, 0, .12]
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=40,
                    effect=lambda: arm.location(obj)[2] > target[2] - .010)


def align_cartridge(arm, obj):
    """Align the offset flange at the open end of the rail."""
    target = cartridge_pose(arm) + [-.217, 0, .007]
    rotation = arm.orientation(obj).inv() * arm.rotation
    return arm.line(hand_for_object(arm, obj, target, rotation), rotation, held=obj, steps=40,
                    effect=lambda: aligned(arm, obj)
                    and np.linalg.norm(arm.location(obj) - target) < .009)


def feed_cartridge(arm, obj):
    """Feed the held cartridge horizontally to engagement depth."""
    target = cartridge_pose(arm) + [0, 0, .006]
    rotation = arm.orientation(obj).inv() * arm.rotation
    return arm.line(hand_for_object(arm, obj, target, rotation), rotation, held=obj, steps=58,
                    effect=lambda: aligned(arm, obj)
                    and np.linalg.norm(arm.location(obj) - target) < .009)


def seat_cartridge_stop(arm, obj):
    """Advance a fed cartridge into contact with the rail's rear stop."""
    target = cartridge_pose(arm) + [.010, 0, .006]
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=18, check_pose=False,
                    effect=lambda: aligned(arm, obj)
                    and arm.contact(CONTACT_PART[obj], "rail_stop"))


def orient_for_inspection(arm, obj, yaw):
    """Present the connector key at a requested yaw while retaining the grasp."""
    rotation = Rotation.from_euler("z", yaw) * arm.orientation(obj).inv() * arm.rotation
    return arm.line(arm.position, rotation, held=obj, steps=30,
                    effect=lambda: aligned(arm, obj, yaw))


def withdraw_cartridge(arm, obj):
    """Slide a grasped cartridge out of the channel along its insertion axis."""
    target = arm.location(obj)
    target[0] = cartridge_pose(arm)[0] - .217
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=55,
                    effect=lambda: arm.location(obj)[0] < target[0] + .009)


def seat_in_nest(arm, obj, nest):
    """Lower a held component onto the nest and verify support under its stem or tongue."""
    height = TONGUE_DEPTH if obj == "cartridge" else STEM_LENGTH
    target = [*arm.location(nest)[:2], top(arm, nest) + height - .005]
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=33, check_pose=False,
                    effect=lambda: arm.contact(CONTACT_PART[obj], nest)
                    and np.linalg.norm(arm.location(obj)[:2] - target[:2]) < .015)


SERVICE_SKILLS = {build.__name__: build for build in (
    approach_keeper, swing_keeper, descend_to_sensor, align_sensor, seat_sensor, extract_sensor, align_cartridge, feed_cartridge,
    seat_cartridge_stop, orient_for_inspection, withdraw_cartridge, seat_in_nest,
)}
BUILDERS = {**{name: primitive.build for name, primitive in PRIMITIVES.items()}, **SERVICE_SKILLS}


def execute(arm, call):
    return arm.run(BUILDERS[call.skill](arm, **call.args))
