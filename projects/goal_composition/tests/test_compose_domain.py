from dataclasses import replace

import numpy as np

from skill_composer.compose.domain import Action, State, candidates, encode, shortest_plan


def test_same_scene_can_require_different_receiving_arms():
    scene = State('shared', None, {'supply': (0,), 'shared': (0, 1), 'stack': (1,)}, 'stack')
    changed = replace(scene, goal='supply')
    choices = candidates(scene)
    assert choices == candidates(changed)
    np.testing.assert_array_equal(encode(scene, choices)[:, 1], [0, 1])
    np.testing.assert_array_equal(encode(changed, choices)[:, 1], [1, 0])
    assert shortest_plan(scene)[0] == Action('pick', 1, 'shared')
    assert shortest_plan(changed)[0] == Action('pick', 0, 'shared')


def test_candidate_features_depend_on_reach_instead_of_dock_identity():
    scene = State(None, 0, {'dock0': (0, 1), 'dock1': (0,), 'goal': (1,)})
    swapped = replace(scene, reach={'dock0': (0,), 'dock1': (0, 1), 'goal': (1,)})
    assert candidates(scene) == candidates(swapped)
    np.testing.assert_array_equal(encode(scene, candidates(scene))[:, 2], [1, 0])
    np.testing.assert_array_equal(encode(swapped, candidates(swapped))[:, 2], [0, 1])
