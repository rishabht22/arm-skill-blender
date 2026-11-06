"""Run two independent reaching skills in one shared simulation."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .reach import build_reach_observation, map_reach_action, reach_observation_keys
from .two_arm_lift import make_env


class CombinedReachEnv(gym.Env):
    """Both arms act each step and must hold their goals simultaneously."""

    metadata = {"render_modes": []}
    render_mode = None

    def __init__(self, arm_settings, *, seed=1000, horizon=200, hold_seconds=3.0):
        super().__init__()
        if len(arm_settings) != 2:
            raise ValueError("Combined evaluation requires one reaching setup per arm")
        if not np.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive and finite")
        self.arm_settings = [dict(settings) for settings in arm_settings]
        for index, settings in enumerate(self.arm_settings):
            if settings["robot_index"] != index:
                raise ValueError("Policies must be ordered UR5e first, Jaco second")
            if not 0 < settings["action_scale"] <= 1 or settings["success_distance"] <= 0:
                raise ValueError("Invalid reaching action scale or success distance")
            if settings["hold_steps"] < 1 or settings["target_height"] < 0:
                raise ValueError("Invalid saved reaching hold duration or target height")
        self.sim_env = make_env(seed=seed, use_object_obs=True, horizon=horizon)
        self.robot_names = [robot.name for robot in self.sim_env.robots]
        self.required_hold_steps = math.ceil(hold_seconds / self.sim_env.control_timestep)
        self.hold_seconds = self.required_hold_steps * self.sim_env.control_timestep
        if horizon < self.required_hold_steps:
            self.sim_env.close()
            raise ValueError("Episode horizon must allow the full simultaneous hold")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        observation, _ = self.reset(seed=seed)
        self.observation_space = spaces.Dict({
            key: spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float32)
            for key, value in observation.items()
        })

    def policy_metadata(self, robot_index):
        return {
            "task": "reach_handle",
            "robots": self.robot_names,
            "observation_keys": reach_observation_keys(robot_index),
            "observation_dim": self.observation_space[f"robot{robot_index}"].shape[0],
            "action_dim": 3,
            "reach": self.arm_settings[robot_index],
        }

    def _observations(self, observation):
        return {
            f"robot{index}": build_reach_observation(
                self.sim_env, observation, index, self.target_positions[index],
                self.held_steps[index], self.arm_settings[index]["hold_steps"],
            )
            for index in range(2)
        }

    def _distances(self):
        return np.array([
            np.linalg.norm(self.target_positions[index] - getattr(self.sim_env, f"_eef{index}_xpos"))
            for index in range(2)
        ])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        observation = self.sim_env.reset()
        # Both controllers retain their normal active-arm behavior. No arm is frozen.
        self.target_positions = []
        for index, settings in enumerate(self.arm_settings):
            target = getattr(self.sim_env, f"_handle{index}_xpos").copy()
            target[2] += settings["target_height"]
            self.target_positions.append(target)
        self.held_steps = [0, 0]
        self.max_held_steps = [0, 0]
        self.joint_held_steps = 0
        self.max_joint_held_steps = 0
        self._previous_distances = self._distances()
        self._done = False
        return self._observations(observation), {"distances": self._previous_distances.copy()}

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode finished; call reset() before taking another step")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Combined reaching requires three finite XYZ commands per arm")
        action = np.clip(action, -1.0, 1.0)
        full_action = sum(
            map_reach_action(self.sim_env, index, action[3 * index:3 * index + 3], settings["action_scale"])
            for index, settings in enumerate(self.arm_settings)
        )
        # Advance the shared simulation once, with both robots' commands together.
        observation, _, timeout, _ = self.sim_env.step(full_action)
        distances = self._distances()
        inside = [distances[index] <= settings["success_distance"] for index, settings in enumerate(self.arm_settings)]
        for index in range(2):
            self.held_steps[index] = self.held_steps[index] + 1 if inside[index] else 0
            self.max_held_steps[index] = max(self.max_held_steps[index], self.held_steps[index])
        self.joint_held_steps = self.joint_held_steps + 1 if all(inside) else 0
        self.max_joint_held_steps = max(self.max_joint_held_steps, self.joint_held_steps)
        success = self.joint_held_steps >= self.required_hold_steps
        reward = float(10.0 * np.sum(self._previous_distances - distances) - np.sum(distances))
        reward -= 0.005 * float(action @ action)
        if success:
            reward += 5.0
        self._previous_distances = distances
        self._done = success or bool(timeout)
        info = {
            "is_success": success,
            "distances": distances.copy(),
            "joint_hold_seconds": self.joint_held_steps * self.sim_env.control_timestep,
            "max_joint_hold_seconds": self.max_joint_held_steps * self.sim_env.control_timestep,
            "arm_success": [steps >= self.required_hold_steps for steps in self.max_held_steps],
        }
        return self._observations(observation), reward, success, bool(timeout) and not success, info

    def close(self):
        self.sim_env.close()
