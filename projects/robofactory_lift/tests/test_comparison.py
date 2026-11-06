import json

import numpy as np

from skill_composer.execution import Trial
from skill_composer.model import CompositionHead
from skill_composer.sim import LiftSandbox


def test_parked_arm_keeps_its_motor_targets(library):
    env = LiftSandbox(library)
    before = env.data.ctrl[9:18].copy()
    env.step([0, -1], guarded=False)
    np.testing.assert_array_equal(env.data.ctrl[9:18], before)
    assert not env.attachments[1]


def test_hold_cannot_bypass_the_shared_grasp_guard(library):
    env = LiftSandbox(library)
    executed, blocked = env.step([3, 2], guarded=True)
    np.testing.assert_array_equal(executed, [1, 1])
    assert blocked == 2


def test_baselines_and_head_receive_identical_delays_and_initial_states(library):
    first = Trial("independent", library, CompositionHead(), seed=3, delay_arm1=.15)
    second = Trial("head_guarded", library, CompositionHead(), seed=3, delay_arm1=.15)
    np.testing.assert_array_equal(first.env.data.qpos, second.env.data.qpos)
    for trial in (first, second):
        before = trial.env.data.ctrl[9:18].copy()
        for _ in range(3):
            assert trial.step()["executed"][1] == -1
            np.testing.assert_array_equal(trial.env.data.ctrl[9:18], before)


def test_single_arm_trial_can_never_count_a_shared_success(library):
    trial = Trial("arm0", library, CompositionHead())
    for _ in range(30):
        trial.step()
    report = trial.report()
    assert report["enabled_arms"] == [True, False]
    assert not report["shared_task_success"]
    assert not report["final"]["attachments"][1]
    assert json.dumps(report)
