"""Check the training interface when optional training dependencies are installed."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from src.envs.policy_env import PolicyEnv


def test_policy_observations_include_object_and_both_robots():
    env = PolicyEnv(seed=0, horizon=2)
    try:
        observation, _ = env.reset(seed=7)
        assert env.robot_names == ["UR5e", "Jaco"]
        assert env.observation_keys == [
            "robot0_proprio-state", "robot1_proprio-state", "object-state"
        ]
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation)
        assert env.sim_env.use_object_obs
        assert env.sim_env.reward_shaping

        _, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert np.isfinite(reward)
        assert not terminated and not truncated
        assert isinstance(info["is_success"], bool)

        observation, _, terminated, truncated, _ = env.step(np.zeros(env.action_space.shape))
        assert not terminated and truncated
        assert env.observation_space.contains(observation)
    finally:
        env.close()
