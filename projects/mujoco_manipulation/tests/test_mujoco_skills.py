"""Physical integration tests for the fixed MuJoCo manipulation skills."""

import importlib.util
import numpy as np
import pytest

from src.runtime import configure_caches

configure_caches()
if importlib.util.find_spec("robosuite") is None:
    pytest.skip("MuJoCo runtime not installed", allow_module_level=True)

from src.mujoco_skills.tabletop import make_tabletop
from src.mujoco_skills.controllers import TabletopSkills


@pytest.mark.parametrize("seed", [0, 2, 7])
def test_stack_then_unstack_and_relocate_in_one_continuous_scene(seed):
    env = make_tabletop(seed)
    try:
        env.reset()
        library = TabletopSkills(env)
        assert np.linalg.norm(library.cube("cubeA")[:2] - library.cube("cubeB")[:2]) > .14
        assert library.stack()
        assert env._check_success()
        first_steps = env.timestep
        assert library.unstack_and_relocate()
        assert env.timestep > first_steps
        assert library.events.count("grasp") == 2
        assert not env._check_success()
        assert abs(library.cube("cubeA")[2] - .82) < .01
    finally:
        env.close()
