"""Exercise reaching, stationary-arm control, and goal semantics in MuJoCo."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from src.envs.reach import ReachEnv


@pytest.mark.parametrize("robot_index", [0, 1])
def test_xyz_control_reaches_and_holds_goal_while_other_arm_stays_put(robot_index):
    env = ReachEnv(seed=0, robot_index=robot_index)
    try:
        observation, _ = env.reset(seed=0)
        assert env.action_space.shape == (3,)
        assert env.observation_space.contains(observation)
        assert env.observation_keys[0] == f"robot{robot_index}_proprio-state"
        assert env.hold_seconds == pytest.approx(3.0)
        assert env.hold_steps == 60
        assert env.target_height == pytest.approx(0.03)
        idle_position = lambda: getattr(env.sim_env, f"_eef{1 - robot_index}_xpos")
        stationary_start = idle_position().copy()
        goal = env.target_pos.copy()
        held_steps = 0
        maximum_idle_displacement = 0.0
        for _ in range(env.sim_env.horizon):
            # This controller only verifies the task is physically reachable;
            # it supplies no training data and performs no policy optimization.
            error = goal - env.eef_position
            action = np.clip(error / 0.01, -1.0, 1.0)
            observation, reward, terminated, truncated, info = env.step(action)
            distance = np.linalg.norm(goal - env.eef_position)
            held_steps = held_steps + 1 if distance <= env.success_distance else 0
            assert terminated == (held_steps >= env.hold_steps)
            assert env.observation_space.contains(observation)
            np.testing.assert_array_equal(env.target_pos, goal)
            maximum_idle_displacement = max(
                maximum_idle_displacement,
                np.linalg.norm(idle_position() - stationary_start),
            )
            if terminated or truncated:
                break
        assert terminated and not truncated
        assert info["is_success"]
        assert reward > 0
        assert maximum_idle_displacement < 0.005
    finally:
        env.close()


def test_reach_seed_is_repeatable_and_timeout_is_truncation():
    env = ReachEnv(seed=0, horizon=2)
    try:
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_array_equal(first, second)
        for index in range(2):
            observation, reward, terminated, truncated, info = env.step(np.zeros(3))
            assert not terminated
            assert truncated == (index == 1)
            assert not info["is_success"]
            assert env.observation_space.contains(observation)
            assert np.isfinite(reward)
    finally:
        env.close()


def test_moving_or_lifting_pot_does_not_move_reach_target_or_count_as_success():
    env = ReachEnv(seed=0, hold_steps=1)
    try:
        env.reset(seed=0)
        target = env.target_pos.copy()
        joint = env.sim_env.pot.joints[0]
        pose = env.sim_env.sim.data.get_joint_qpos(joint).copy()
        pose[2] += 0.3
        env.sim_env.sim.data.set_joint_qpos(joint, pose)
        env.sim_env.sim.forward()
        assert env.sim_env._check_success()
        _, _, terminated, _, info = env.step(np.zeros(3))
        np.testing.assert_array_equal(env.target_pos, target)
        assert not terminated
        assert not info["is_success"]
    finally:
        env.close()
