"""Smoke tests against the installed simulator, without opening a viewer."""

import numpy as np

from arm_skill_blender.environment import make_env


def test_scene_loads_without_advancing_physics():
    env = make_env(seed=0)
    try:
        env.reset()

        assert len(env.robots) == 2
        assert env.action_dim == 14
        assert not env.has_renderer
        assert not env.has_offscreen_renderer
        assert env.sim.data.time == 0.0
        assert np.isfinite(env.sim.data.qpos).all()
        assert np.isfinite(env.sim.data.qvel).all()

        # The static preview passes these native objects to MuJoCo's viewer.
        import mujoco

        assert isinstance(env.sim.model._model, mujoco.MjModel)
        assert isinstance(env.sim.data._data, mujoco.MjData)
    finally:
        env.close()


def initial_pose(seed):
    env = make_env(seed=seed)
    try:
        env.reset()
        return env.sim.data.qpos.copy()
    finally:
        env.close()


def test_initial_placement_is_repeatable():
    pose = initial_pose(seed=7)
    np.testing.assert_array_equal(pose, initial_pose(seed=7))
    assert not np.array_equal(pose, initial_pose(seed=8))
