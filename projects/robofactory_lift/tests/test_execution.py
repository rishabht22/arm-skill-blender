import numpy as np
import torch

from skill_composer.model import CompositionHead
from skill_composer.sim import LiftSandbox
from skill_composer.skills import FrozenJointSkills


def test_only_the_head_has_trainable_parameters_and_executor_is_immutable(library):
    model = CompositionHead()
    assert model(torch.zeros(3, 48)).shape == (3, 2, 4)
    executor = FrozenJointSkills(library)
    copy = executor.command(0, 0)
    copy[:] = 99
    assert executor.command(0, 0)[0] == 0
    assert not executor.targets[0].flags.writeable
    assert executor.is_complete(0, 0, executor.command(0, 0), False)
    assert not executor.is_complete(0, 2, executor.command(0, 2), False)
    assert executor.is_complete(0, 2, executor.command(0, 2), True)


def test_mujoco_step_and_temporal_grasp_constraint(library):
    env = LiftSandbox(library)
    assert env.decision_observation([-1, -1], [True, True]).shape == (48,)
    executed, blocked = env.step([2, 2], guarded=True)
    np.testing.assert_array_equal(executed, [1, 1])
    assert blocked == 2
    assert np.isfinite(env.data.qpos).all()
    # Close fingers cannot attach when the two grippers are far from the handles.
    env.data.qpos[18:21] += [1, 0, 0]
    env.data.qpos[[7, 8, 16, 17]] = 0
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    env.step([1, 1])
    assert not env.attachments.any()
