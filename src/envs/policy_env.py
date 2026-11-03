"""Expose the current two-arm scene as a state-based Gymnasium environment."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .two_arm_lift import make_env


class PolicyEnv(gym.Env):
    """One policy observes both robots and the pot, then controls both arms."""

    metadata = {"render_modes": []}
    render_mode = None

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
