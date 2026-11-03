"""Load and evaluate a reaching checkpoint without changing its artifacts."""

import json

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.policy_env import describe_policy_env, make_policy_env
from src.policies.basic import BasicPolicy
from src.training.evaluation import RUN_FILES, evaluate, latest_run


@pytest.mark.parametrize("robot_index,legacy_settings", [(0, False), (1, False), (0, True)])
def test_reach_checkpoint_round_trip_and_arm_selection(tmp_path, robot_index, legacy_settings):
    run_dir = tmp_path / "runs" / "001"
    run_dir.mkdir(parents=True)
    reach_settings = {"robot_index": robot_index}
    if legacy_settings:
        reach_settings.update({"hold_steps": 10, "target_height": 0.04})
    raw_env = make_policy_env(task="reach_handle", seed=0, horizon=2, **reach_settings)
    env = VecNormalize(DummyVecEnv([lambda: raw_env]), norm_reward=False)
    try:
        env.reset()
        # Initialize and save an untrained network. There is no learn() call.
        model = PPO(BasicPolicy, env, n_steps=2, batch_size=2, device="cpu", seed=0)
        model.save(str(run_dir / "policy.zip"))
        env.save(str(run_dir / "vecnormalize.pkl"))
        environment_info = describe_policy_env(raw_env)
        if legacy_settings:
            reach_settings.pop("robot_index")
            environment_info["reach"].pop("robot_index")
        (run_dir / "config.json").write_text(json.dumps({
            "task": "reach_handle", "horizon": 2, "reach": reach_settings,
        }))
        (run_dir / "environment.json").write_text(json.dumps(environment_info))
    finally:
        env.close()

    # A newer reaching run for the other arm must not be selected.
    other_arm_dir = tmp_path / "runs" / "002"
    other_arm_dir.mkdir()
    for name in RUN_FILES:
        (other_arm_dir / name).write_text("{}")
    (other_arm_dir / "environment.json").write_text(json.dumps({
        "task": "reach_handle", "reach": {"robot_index": 1 - robot_index},
    }))

    # A later legacy lift run must not be chosen as a reaching model either.
    legacy_dir = tmp_path / "runs" / "003"
    legacy_dir.mkdir()
    for name in RUN_FILES:
        (legacy_dir / name).write_text("{}")
    assert latest_run(tmp_path / "runs", robot_index=robot_index) == run_dir

    artifacts_before = {name: (run_dir / name).read_bytes() for name in RUN_FILES}
    results = evaluate(run_dir, episodes=1, render=False, seed=1000)
    assert len(results) == 1
    assert results[0]["steps"] == 2
    assert not results[0]["success"]
    assert artifacts_before == {name: (run_dir / name).read_bytes() for name in RUN_FILES}
