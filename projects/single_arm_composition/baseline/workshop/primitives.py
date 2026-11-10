"""Primitive catalog. Each handler builds one motion, without dispatching other skills."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..individual_skill_generation.skills import TOP_DOWN

from .control import Motion
from .environment import TABLE_Z


@dataclass(frozen=True)
class Primitive:
    name: str
    description: str
    build: Callable[..., Motion]


PRIMITIVES = {}


def primitive(description):
    def register(build):
        PRIMITIVES[build.__name__] = Primitive(build.__name__, description, build)
        return build
    return register


def top_grasp(arm, obj):
    forward = arm.orientation(obj).apply([1, 0, 0])
    yaw = np.arctan2(forward[1], forward[0])
    return Rotation.from_euler("z", yaw) * TOP_DOWN


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def hand_for_object(arm, obj, position, rotation=None):
    """Resolve a desired object-center position using the measured in-hand offset."""
    rotation = arm.rotation if rotation is None else rotation
    offset = arm.rotation.inv().apply(arm.location(obj) - arm.position)
    return np.asarray(position) - rotation.apply(offset)


@primitive("Move to a clearance pose above an object or target.")
def hover_over(arm, xy=None, obj=None, held=None, height=.20):
    xy = arm.location(obj)[:2] if xy is None else xy
    rotation = top_grasp(arm, obj) if obj and not held else (arm.rotation if held else TOP_DOWN)
    return arm.line([*xy, TABLE_Z + height], rotation, held=held)


@primitive("Lower the open jaws around a grasp point.")
def descend_to_grasp(arm, obj):
    return arm.line(arm.location(obj) + [0, 0, .003], top_grasp(arm, obj))


@primitive("Close the jaws, checking opposing contacts when an object is requested.")
def close_gripper(arm, obj=None):
    return arm.path([arm.position], grip=1,
                    effect=lambda: arm.holding(obj) if obj else arm.aperture < .012)


@primitive("Open the jaws and check the named object was released.")
def open_gripper(arm, obj=None):
    return arm.path([arm.position], grip=-1,
                    effect=lambda: arm.aperture > .070 and (obj is None or not arm.holding(obj)))


@primitive("Raise the hand vertically, optionally retaining a grasp.")
def raise_by(arm, height, held=None):
    return arm.line(arm.position + [0, 0, height], held=held)


@primitive("Lower the hand to an absolute height without specifying contact.")
def lower_to(arm, height, held=None):
    return arm.line([*arm.position[:2], height], held=held)


@primitive("Move to the clearance pose behind an object relative to a push direction.")
def move_behind(arm, obj, direction, gap=.064):
    xy = arm.location(obj)[:2] - gap * unit(direction)
    return arm.line([*xy, TABLE_Z + .18], TOP_DOWN)


@primitive("Descend to contact height at the current push staging point.")
def descend_behind(arm, obj, height=None):
    z = arm.location(obj)[2] + .003 if height is None else height
    return arm.line([*arm.position[:2], z], TOP_DOWN)


@primitive("Displace a free object in a specified planar direction using unilateral contact.")
def push_along(arm, obj, direction, distance):
    axis = unit(direction)
    start = arm.location(obj)
    target = start[:2] + axis * (distance - .024)
    return arm.line([*target, arm.position[2]], steps=28,
                    effect=lambda: np.dot(arm.location(obj)[:2] - start[:2], axis) >= distance - .023)


@primitive("Rotate an airborne object about the vertical axis while maintaining the grasp.")
def yaw_in_hand(arm, obj, angle):
    start = arm.orientation(obj)
    delta = Rotation.from_rotvec([0, 0, angle])
    goal = delta * start
    return arm.line(arm.position, delta * arm.rotation, held=obj, steps=25,
                    effect=lambda: (goal * arm.orientation(obj).inv()).magnitude() < .13)


@primitive("Tilt a held object about a horizontal axis.")
def tilt_in_hand(arm, obj, angle):
    start = arm.orientation(obj)
    delta = Rotation.from_rotvec([0, angle, 0])
    goal = delta * start
    return arm.line(arm.position, delta * arm.rotation, held=obj, steps=25,
                    effect=lambda: (goal * arm.orientation(obj).inv()).magnitude() < .13)


@primitive("Drag a grasped object along the tabletop without lifting it.")
def drag_on_table(arm, obj, xy):
    z = arm.location(obj)[2]
    target = hand_for_object(arm, obj, [*xy, z])
    return arm.line(target, held=obj, steps=38,
                    effect=lambda: np.linalg.norm(arm.location(obj)[:2] - xy) < .020
                    and arm.contact(obj, "table_collision"))


@primitive("Lower a held object until it touches a named support.")
def seat_on_support(arm, obj, support, half_height):
    model = arm.env.sim.model
    top = arm.location(support)[2] + model.geom_size[model.geom_name2id(support), 2]
    target = arm.location(obj)
    target[2] = top + half_height - .002
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=25,
                    effect=lambda: arm.contact(obj, support), check_pose=False)


@primitive("Push an object until it contacts a physical end stop.")
def push_to_stop(arm, obj, stop):
    target = [arm.location(stop)[0] - .038, arm.location(obj)[1], arm.position[2]]
    return arm.line(target, steps=35, effect=lambda: arm.contact(obj, stop), check_pose=False)


@primitive("Tip a tall object by pushing above its center of mass.")
def topple(arm, obj):
    target = arm.position + [0, .16, 0]
    return arm.line(target, steps=27,
                    effect=lambda: abs(arm.orientation(obj).apply([0, 0, 1])[2]) < .35)


@primitive("Roll a horizontal cylinder and verify both translation and rotation.")
def roll_cylinder(arm, obj, distance):
    start = arm.location(obj)
    rotation = arm.orientation(obj)
    return arm.line([start[0] + distance - .02, arm.position[1], arm.position[2]], steps=26,
                    effect=lambda: arm.location(obj)[0] - start[0] > distance - .024
                    and abs((arm.orientation(obj) * rotation.inv()).as_rotvec()[1]) > .7)


@primitive("Move a free object with a curved planar sweep of the closed jaws.")
def sweep_arc(arm, obj, radius, angle):
    start = arm.location(obj)
    arc = np.linspace(0, angle, 34)
    positions = arm.position + np.column_stack((radius * np.sin(arc),
                                               radius * (1 - np.cos(arc)), np.zeros_like(arc)))
    return arm.path(positions, effect=lambda: arm.location(obj)[0] - start[0] > .045
                    and arm.location(obj)[1] - start[1] > .015)


@primitive("Advance a puck through parallel guides using unilateral contact.")
def slide_in_channel(arm, obj, end_x):
    start = arm.location(obj)
    return arm.line([end_x - .024, .245, arm.position[2]], steps=45,
                    effect=lambda: arm.location(obj)[0] > end_x - .025
                    and abs(arm.location(obj)[1] - .245) < .011
                    and arm.location(obj)[0] - start[0] > .055)


@primitive("Approach a handle horizontally at grasp height with the jaws open.")
def side_approach(arm, obj):
    goal = arm.location(obj) + [0, 0, .003]
    goal[2] = arm.position[2]
    return arm.line(goal, top_grasp(arm, obj),
                    effect=lambda: np.linalg.norm(arm.location(obj) - arm.position) < .014
                    and arm.aperture > .070)


@primitive("Back straight out of a fixture opening at constant height.")
def retract_from_fixture(arm, obj, distance=.065):
    goal = arm.position + [-distance, 0, 0]
    return arm.line(goal, effect=lambda: arm.position[0] < arm.location(obj)[0] - .045)


@primitive("Carry an object along an arched path over a low obstacle.")
def clear_obstacle(arm, obj, xy, obstacle="barrier", clearance=.09):
    t = np.linspace(0, 1, 37)[1:]
    goal = np.array([*xy, arm.position[2]])
    path = arm.position + t[:, None] * (goal - arm.position)
    path[:, 2] += clearance * np.sin(np.pi * t)
    return arm.path(path, held=obj, guard=lambda: not arm.contact(obj, obstacle))


def fixture_push(arm, obj, target, position, sign, tolerance=.009):
    return arm.line(position, steps=30, check_pose=False,
                    effect=lambda: sign * (arm.joint(obj) - target) >= -tolerance)


@primitive("Depress a spring-loaded button and verify its stroke.")
def press_button(arm, obj, target=.022):
    position = arm.location(obj) + [0, 0, -.013]
    return fixture_push(arm, obj, target, position, 1, tolerance=.005)


@primitive("Pull a grasped drawer handle along its slide axis.")
def pull_drawer(arm, obj, target=.093):
    delta = target - arm.joint(obj)
    return arm.line(arm.position + [-delta, 0, 0], held=obj, steps=35,
                    effect=lambda: abs(arm.joint(obj) - target) < .013)


@primitive("Close a drawer by pushing its handle without grasping it.")
def push_drawer(arm, obj, target=.0):
    position = arm.location(obj) + [arm.joint(obj) - target - .019, 0, .003]
    return fixture_push(arm, obj, target, position, -1)


@primitive("Advance a captive slider to a requested position using side contact.")
def slide_slider(arm, obj, target=.060):
    position = arm.location(obj) + [target - arm.joint(obj) - .021, 0, .003]
    return fixture_push(arm, obj, target, position, 1)


@primitive("Lift a grasped captive latch along its guide.")
def lift_latch(arm, obj, target=.047):
    return arm.line(arm.position + [0, 0, target - arm.joint(obj)], held=obj, steps=24,
                    effect=lambda: abs(arm.joint(obj) - target) < .009)


def follow_hinge(arm, obj, axis, target, pitch=0):
    delta = target - arm.joint(obj)
    angles = np.linspace(0, delta, 31)[1:]
    rotations = Rotation.from_rotvec(angles[:, None] * np.asarray(axis))
    pivot = arm.env.sim.data.body_xpos[arm.bodies[obj]].copy()
    path = pivot + rotations.apply(arm.position - pivot)
    path[:, 2] += pitch * angles
    return arm.path(path, rotations * arm.rotation, held=obj,
                    effect=lambda: abs(arm.joint(obj) - target) < .15)


@primitive("Turn a grasped dial about its fixed vertical spindle.")
def turn_dial(arm, obj, target=.72):
    return follow_hinge(arm, obj, [0, 0, 1], target)


@primitive("Swing a gate by following its offset handle around a vertical hinge.")
def swing_gate(arm, obj, target=.83):
    return follow_hinge(arm, obj, [0, 0, 1], target)


@primitive("Flip a toggle by tracing its handle around a horizontal pivot.")
def flip_toggle(arm, obj, target=.58):
    return follow_hinge(arm, obj, [0, 1, 0], target)


@primitive("Align a held keyed peg to a socket centerline and orientation.")
def align_to_socket(arm, obj, socket):
    target = arm.location(socket) + [0, 0, .092]
    rotation = arm.orientation(obj).inv() * arm.rotation
    return arm.line(hand_for_object(arm, obj, target, rotation), rotation, held=obj, steps=28,
                    effect=lambda: np.linalg.norm(arm.location(obj)[:2] - target[:2]) < .005
                    and arm.orientation(obj).magnitude() < .13)


@primitive("Insert a grasped peg into the socket along its centerline.")
def insert_peg(arm, obj, socket):
    target = arm.location(socket) + [0, 0, .041]
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=35,
                    effect=lambda: np.linalg.norm(arm.location(obj) - target) < .008)


@primitive("Extract a grasped peg straight out of the socket until its shaft clears.")
def extract_peg(arm, obj, socket):
    target = arm.location(socket) + [0, 0, .112]
    return arm.line(hand_for_object(arm, obj, target), held=obj, steps=31,
                    effect=lambda: arm.location(obj)[2] > target[2] - .009)


@primitive("Feed a held key horizontally between rails to its insertion depth.")
def slide_key_in(arm, obj, stop):
    limit = arm.location(stop)
    target = np.array([limit[0] - .046, limit[1], arm.location(obj)[2]])
    rotation = arm.orientation(obj).inv() * arm.rotation
    return arm.line(hand_for_object(arm, obj, target, rotation), rotation, held=obj, steps=55,
                    effect=lambda: np.linalg.norm(arm.location(obj) - target) < .008
                    and arm.orientation(obj).magnitude() < .20)


@primitive("Seat a press-fit pin using downward contact from the closed jaws.")
def seat_press_fit_pin(arm, obj, target=.023):
    position = arm.location(obj) + [0, 0, -.013]
    return fixture_push(arm, obj, target, position, 1, tolerance=.005)


@primitive("Screw a captive plug in with a coupled turn and vertical travel.")
def twist_in_socket(arm, obj, target=.79):
    motion = follow_hinge(arm, obj, [0, 0, 1], target, pitch=.014)
    angle_reached = motion.effect
    motion.effect = lambda: angle_reached() and abs(arm.joint("plug_rise") - .014 * target) < .004
    return motion
