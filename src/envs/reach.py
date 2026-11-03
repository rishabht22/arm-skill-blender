"""Train either arm to reach its own handle in the existing two-arm scene."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .two_arm_lift import make_env


def reach_observation_keys(robot_index):
    return [
        f"robot{robot_index}_proprio-state", "reach_target_pos", "eef_to_target", "reach_hold_fraction"
    ]


def build_reach_observation(sim_env, observation, robot_index, target_pos, held_steps, hold_steps):
    """Build the exact per-arm input used during independent policy training."""
    eef_position = getattr(sim_env, f"_eef{robot_index}_xpos")
    return np.concatenate([
        observation[f"robot{robot_index}_proprio-state"],
        target_pos,
        target_pos - eef_position,
        # During combined evaluation a policy may hold longer than it trained for.
        [min(held_steps / hold_steps, 1.0)],
    ]).astype(np.float32)


def map_reach_action(sim_env, robot_index, action, action_scale):
    """Map one policy's world XYZ command into its own robot's action slice."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError("Reaching requires three finite XYZ action values")
    action = np.clip(action, -1.0, 1.0)
    controller = sim_env.robots[robot_index].composite_controller.part_controllers["right"]
    if controller.input_ref_frame == "base":
        action = controller.origin_ori.T @ action
    start = sum(robot.action_dim for robot in sim_env.robots[:robot_index])
    full_action = np.zeros(sim_env.action_dim)
    full_action[start:start + 3] = np.clip(action * action_scale, -1.0, 1.0)
    return full_action


class ReachEnv(gym.Env):
    """Learn three world-frame XYZ commands while the other arm holds still.

    The goal is fixed above the selected arm's pot handle at reset. Holding near that
    point completes the task; neither grasping nor lifting is required.
    """

    metadata = {"render_modes": []}
    render_mode = None
    task = "reach_handle"

    def __init__(
        self, *, seed=0, horizon=200, robot_index=0, action_scale=0.2,
        success_distance=0.03, hold_seconds=3.0, hold_steps=None, target_height=0.03,
    ):
        super().__init__()
        if not 0 < action_scale <= 1 or success_distance <= 0 or target_height < 0:
            raise ValueError("Invalid reaching action scale, distance, or target height")
        if robot_index not in (0, 1):
            raise ValueError("robot_index must be 0 (UR5e) or 1 (Jaco)")
        if not np.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive and finite")
        if hold_steps is not None and (not isinstance(hold_steps, int) or hold_steps < 1):
            raise ValueError("hold_steps must be a positive integer")
        self.robot_index = robot_index
        self.idle_robot_index = 1 - robot_index
        self.action_scale = action_scale
        self.success_distance = success_distance
        self.target_height = target_height
        self.sim_env = make_env(seed=seed, use_object_obs=True, horizon=horizon)
        # Explicit step counts remain supported for existing saved checkpoints.
        self.hold_steps = hold_steps if hold_steps is not None else math.ceil(
            hold_seconds / self.sim_env.control_timestep
        )
        self.hold_seconds = self.hold_steps * self.sim_env.control_timestep
        self.robot_names = [robot.name for robot in self.sim_env.robots]
        self.proprio_key = f"robot{robot_index}_proprio-state"
        self.observation_keys = reach_observation_keys(robot_index)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        observation, _ = self.reset(seed=seed)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=observation.shape, dtype=np.float32
        )

    @property
    def reach_settings(self):
        return {
            "robot_index": self.robot_index,
            "action_scale": self.action_scale,
            "success_distance": self.success_distance,
            "hold_steps": self.hold_steps,
            "target_height": self.target_height,
        }

    @property
    def eef_position(self):
        """Current world position of the controlled arm's gripper."""
        return getattr(self.sim_env, f"_eef{self.robot_index}_xpos")

    def _observation(self, observation):
        return build_reach_observation(
            self.sim_env, observation, self.robot_index,
            self.target_pos, self._held_steps, self.hold_steps,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        observation = self.sim_env.reset()
        # With zero deltas, preserve the idle arm's original goal.
        idle_controller = self.sim_env.robots[self.idle_robot_index].composite_controller.part_controllers["right"]
        # Initialize the goal in the controller's base frame before preserving it.
        idle_controller.set_goal(np.zeros(idle_controller.control_dim))
        idle_controller.set_goal_update_mode("desired")
        self.target_pos = getattr(self.sim_env, f"_handle{self.robot_index}_xpos").copy()
        self.target_pos[2] += self.target_height
        self._held_steps = 0
        self._done = False
        self._previous_distance = float(np.linalg.norm(self.target_pos - self.eef_position))
        return self._observation(observation), {"distance_to_target": self._previous_distance}

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode finished; call reset() before taking another step")
        # Only the selected arm's three position commands come from the policy.
        # Rotation, gripper, and idle-arm commands stay zero.
        full_action = map_reach_action(self.sim_env, self.robot_index, action, self.action_scale)
        action = np.clip(np.asarray(action), -1.0, 1.0)
        observation, _, timeout, _ = self.sim_env.step(full_action)

        distance = float(np.linalg.norm(self.target_pos - self.eef_position))
        progress = self._previous_distance - distance
        self._previous_distance = distance
        self._held_steps = self._held_steps + 1 if distance <= self.success_distance else 0
        success = self._held_steps >= self.hold_steps
        # Reward progress and penalize distance / unnecessarily large commands.
        # A negative running cost avoids rewarding endless hovering outside the goal.
        reward = 10.0 * progress - distance - 0.005 * float(action @ action)
        if success:
            reward += 5.0
        self._done = success or bool(timeout)
        info = {
            "is_success": success,
            "distance_to_target": distance,
            "held_steps": self._held_steps,
        }
        return self._observation(observation), reward, success, bool(timeout) and not success, info

    def close(self):
        self.sim_env.close()
