"""Expose the current two-arm scene as a state-based Gymnasium environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .two_arm_lift import make_env


class PolicyEnv(gym.Env):
    """Legacy joint lift environment, retained for existing saved models."""

    metadata = {"render_modes": []}
    render_mode = None
    task = "two_arm_lift"

    def __init__(self, *, seed=0, horizon=500):
        super().__init__()
        self.sim_env = make_env(
            seed=seed, use_object_obs=True, reward_shaping=True, horizon=horizon
        )
        self.robot_names = [robot.name for robot in self.sim_env.robots]
        self.observation_keys = [
            f"robot{index}_proprio-state" for index in range(len(self.sim_env.robots))
        ] + ["object-state"]

        observation = self._flatten(self.sim_env.reset())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=observation.shape, dtype=np.float32
        )
        low, high = self.sim_env.action_spec
        self.action_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )

    def _flatten(self, observation):
        return np.concatenate(
            [np.asarray(observation[key]).ravel() for key in self.observation_keys]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # robosuite 1.5.1's placement sampler uses NumPy's global RNG.
            np.random.seed(seed)
        return self._flatten(self.sim_env.reset()), {}

    def step(self, action):
        observation, reward, done, info = self.sim_env.step(action)
        info["is_success"] = bool(self.sim_env._check_success())
        # TwoArmLift ends at its time limit, not at an absorbing terminal state.
        # PPO must be allowed to bootstrap its value estimate at this timeout.
        return self._flatten(observation), float(reward), False, bool(done), info

    def close(self):
        self.sim_env.close()


def make_policy_env(*, task="reach_handle", seed=0, horizon=200, **reach_settings):
    """Choose the task explicitly so training and evaluation use the same setup."""
    if task == "reach_handle":
        from .reach import ReachEnv

        return ReachEnv(seed=seed, horizon=horizon, **reach_settings)
    if task == "two_arm_lift":
        return PolicyEnv(seed=seed, horizon=horizon)
    raise ValueError(f"Unknown policy task: {task}")


def describe_policy_env(env):
    description = {
        "task": env.task,
        "robots": env.robot_names,
        "observation_keys": env.observation_keys,
        "observation_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
    }
    if env.task == "reach_handle":
        description["reach"] = env.reach_settings
    else:
        description["reward_shaping"] = True
    return description
