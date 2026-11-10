"""Callable Panda skills for a robosuite tabletop environment."""

from dataclasses import dataclass

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from scipy.spatial.transform import Rotation

from .skill_names import MOTION_SKILLS


TOP_DOWN = Rotation.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])


def make_env(seed, environment="Lift", **kwargs):
    config = load_composite_controller_config(robot="Panda")
    config["body_parts"]["right"].update(input_type="absolute", input_ref_frame="world")
    return robosuite.make(
        environment,
        robots="Panda",
        controller_configs=config,
        control_freq=20,
        seed=seed,
        hard_reset=False,
        ignore_done=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        **kwargs,
    )


@dataclass(frozen=True)
class SkillCall:
    skill: str
    position: np.ndarray
    rotation: Rotation
    max_steps: int = 100
    grasp: bool = False
    release: bool = False
    retain: bool = False
    min_push: float = 0.0


@dataclass
class Rollout:
    success: bool
    actions: np.ndarray
    observations: dict
    states: np.ndarray


class PandaArm:
    def __init__(self, env):
        self.env = env
        self.robot = env.robots[0]
        self.gripper = self.robot.gripper["right"]
        self.site = self.robot.eef_site_id["right"]
        self.fingers = [env.sim.model.get_joint_qpos_addr(name) for name in self.gripper.joints]
        self.grip_command = -1.0

    @property
    def position(self):
        return self.env.sim.data.site_xpos[self.site].copy()

    @property
    def rotation(self):
        matrix = self.env.sim.data.site_xmat[self.site].reshape(3, 3)
        return Rotation.from_matrix(matrix.copy())

    def robot_observation(self):
        data = self.env.sim.data
        return {
            "eef_position": self.position,
            "eef_quaternion_xyzw": self.rotation.as_quat(),
            "joint_position": data.qpos[self.robot._ref_joint_pos_indexes].copy(),
            "joint_velocity": data.qvel[self.robot._ref_joint_vel_indexes].copy(),
            "finger_position": data.qpos[self.fingers].copy(),
            "gripper_command": self.grip_command,
        }

    def action_for(self, position, rotation):
        return self.robot.create_action_vector({
            "right": np.concatenate((position, rotation.as_rotvec())),
            "right_gripper": np.array([self.grip_command]),
        })


class PandaSkills(PandaArm):
    @property
    def object_position(self):
        return self.env.sim.data.body_xpos[self.env.cube_body_id].copy()

    @property
    def grasped(self):
        return bool(self.env._check_grasp(self.gripper, self.env.cube))

    @property
    def grasp_rotation(self):
        quat = self.env.sim.data.body_xquat[self.env.cube_body_id]
        yaw = Rotation.from_quat(quat, scalar_first=True).as_euler("xyz")[2]
        return Rotation.from_euler("z", yaw % (np.pi / 2) - np.pi / 4) * TOP_DOWN

    def hover_over(self, xy=None):
        holding = self.grasped
        if xy is None:
            xy = self.object_position[:2]
        rotation = self.rotation if holding else self.grasp_rotation
        return SkillCall("hover_over", np.array([*xy, self.env.table_offset[2] + 0.17]),
                         rotation, retain=holding)

    def descend_to_grasp(self):
        target = self.object_position + [0, 0, 0.004]
        return SkillCall("descend_to_grasp", target, self.grasp_rotation)

    def raise_by(self, height):
        return SkillCall("raise_by", self.position + [0, 0, height], self.rotation,
                         retain=self.grasped)

    def lower_to(self, height):
        target = self.position
        target[2] = height
        return SkillCall("lower_to", target, self.rotation, retain=self.grasped)

    def move_behind(self, direction):
        target = self.object_position + [0, -direction * 0.075, 0.14]
        return SkillCall("move_behind", target, TOP_DOWN)

    def descend_behind(self, direction):
        target = self.object_position + [0, -direction * 0.075, 0.006]
        return SkillCall("descend_behind", target, TOP_DOWN)

    def push_along(self, direction, distance):
        target = self.object_position + [0, direction * distance, 0.006]
        return SkillCall("push_along", target, TOP_DOWN, min_push=distance - 0.025)

    def open_gripper(self, *, release=False):
        return SkillCall("open_gripper", self.position, self.rotation, max_steps=60, release=release)

    def close_gripper(self, *, grasp=False):
        return SkillCall("close_gripper", self.position, self.rotation, max_steps=60, grasp=grasp)

    def observe(self):
        return {
            **self.robot_observation(),
            "object_position": self.object_position,
            "object_quaternion_wxyz": self.env.sim.data.body_xquat[self.env.cube_body_id].copy(),
            "grasped": self.grasped,
        }

    def execute(self, call):
        start_object = self.object_position
        holding = self.grasped
        observations = [self.observe()]
        states = [self.env.sim.get_state().flatten()]
        if call.skill == "open_gripper":
            self.grip_command = -1.0
        elif call.skill == "close_gripper":
            self.grip_command = 1.0
        elif call.skill not in MOTION_SKILLS:
            raise ValueError(f"Unknown skill: {call.skill}")

        target = call.position
        action = self.action_for(target, call.rotation)
        if call.min_push:
            direction = (target - self.position)[:2]
            distance = np.linalg.norm(direction)
            if distance == 0:
                raise ValueError("Push requires a horizontal target displacement")
            direction /= distance
        stable = 0
        success = False
        for _ in range(call.max_steps):
            self.env.step(action)
            observations.append(self.observe())
            states.append(self.env.sim.get_state().flatten())
            reached = np.linalg.norm(self.position - target) < 0.009
            aligned = (call.rotation * self.rotation.inv()).magnitude() < 0.12
            aperture = abs(np.diff(observations[-1]["finger_position"])[0])
            completed = reached and aligned
            if call.skill == "open_gripper":
                completed &= aperture > 0.070 and not self.grasped
                if call.release:
                    resting_height = self.env.table_offset[2] + self.env.cube.size[2]
                    completed &= holding and abs(self.object_position[2] - resting_height) < 0.012
            elif call.skill == "close_gripper":
                completed &= self.grasped if call.grasp else aperture < 0.012
            elif call.retain:
                completed &= holding and self.grasped
                relative_change = (self.object_position - start_object) - (self.position - observations[0]["eef_position"])
                completed &= np.linalg.norm(relative_change) < 0.025
            if call.min_push:
                completed &= np.dot((self.object_position - start_object)[:2], direction) > call.min_push
            stable = stable + 1 if completed else 0
            if stable >= 5:
                success = True
                break
        return Rollout(
            success,
            np.repeat(action[None, :], len(states) - 1, axis=0),
            {key: np.asarray([obs[key] for obs in observations]) for key in observations[0]},
            np.asarray(states),
        )
