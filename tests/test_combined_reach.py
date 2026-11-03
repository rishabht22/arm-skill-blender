"""Verify both independent controls act in one shared simulation."""

import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.combined_reach import CombinedReachEnv
from src.envs.policy_env import describe_policy_env, make_policy_env
from src.policies.basic import BasicPolicy
from src.training.combined_evaluation import evaluate_both
from src.training.evaluation import RUN_FILES


ARM_SETTINGS = [
    {"robot_index": 0, "action_scale": 0.2, "success_distance": 0.03, "hold_steps": 10, "target_height": 0.04},
    {"robot_index": 1, "action_scale": 0.2, "success_distance": 0.03, "hold_steps": 60, "target_height": 0.03},
]


def test_both_arms_reach_and_hold_together_with_one_physics_step_per_action():
    env = CombinedReachEnv(ARM_SETTINGS, seed=0)
    try:
        observation, _ = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        targets = np.array(env.target_positions)
        initial_positions = [getattr(env.sim_env, f"_eef{index}_xpos").copy() for index in range(2)]
        joint_hold = 0
        for step in range(env.sim_env.horizon):
            action = np.concatenate([
                np.clip((targets[index] - getattr(env.sim_env, f"_eef{index}_xpos")) / 0.01, -1.0, 1.0)
                for index in range(2)
            ])
            observation, _, terminated, truncated, info = env.step(action)
            distances = [
                np.linalg.norm(targets[index] - getattr(env.sim_env, f"_eef{index}_xpos"))
                for index in range(2)
            ]
            joint_hold = joint_hold + 1 if all(distance <= 0.03 for distance in distances) else 0
            assert env.sim_env.timestep == step + 1
            assert terminated == (joint_hold >= 60)
            assert observation["robot0"][-1] <= 1.0
            assert observation["robot1"][-1] <= 1.0
            assert env.observation_space.contains(observation)
            np.testing.assert_array_equal(env.target_positions, targets)
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["is_success"]
        assert info["joint_hold_seconds"] >= 3.0
        for index in range(2):
            assert np.linalg.norm(getattr(env.sim_env, f"_eef{index}_xpos") - initial_positions[index]) > 0.02
    finally:
        env.close()


def test_two_saved_policies_load_and_evaluate_without_changing_artifacts(tmp_path):
    run_dirs = []
    for index, reach_settings in enumerate(ARM_SETTINGS):
        run_dir = tmp_path / f"robot{index}"
        run_dir.mkdir()
        raw_env = make_policy_env(task="reach_handle", seed=0, horizon=60, **reach_settings)
        env = VecNormalize(DummyVecEnv([lambda: raw_env]), norm_reward=False)
        try:
            env.reset()
            # Save initialized policies; no optimization is performed in this test.
            model = PPO(BasicPolicy, env, n_steps=2, batch_size=2, device="cpu", seed=index)
            model.save(str(run_dir / "policy.zip"))
            env.save(str(run_dir / "vecnormalize.pkl"))
            metadata = describe_policy_env(raw_env)
            if index == 0:
                metadata["reach"].pop("robot_index")  # Original UR5e checkpoint format.
            (run_dir / "environment.json").write_text(json.dumps(metadata))
            (run_dir / "config.json").write_text(json.dumps({"horizon": 60}))
        finally:
            env.close()
        run_dirs.append(run_dir)

    before = {(index, name): (run_dir / name).read_bytes() for index, run_dir in enumerate(run_dirs) for name in RUN_FILES}
    results = evaluate_both(*run_dirs, episodes=1, render=False, seed=0)
    assert len(results) == 1
    assert results[0]["steps"] == 60
    assert not results[0]["success"]
    after = {(index, name): (run_dir / name).read_bytes() for index, run_dir in enumerate(run_dirs) for name in RUN_FILES}
    assert before == after
