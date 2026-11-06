import pytest

from skill_composer.compose.tower_world import rollout


@pytest.mark.parametrize('order', [
    ('part0', 'part1', 'part2'),
    ('part2', 'part0', 'part1'),
])
def test_occupied_dock_composes_requested_tower(order):
    result = rollout('occupied-dock', order=order)
    assert result['success'], result['failure']
    assert result['final_positions'] == dict(zip(order, ('base', *order[:-1])))
    assert not result['final_holders']
    assert {call['skill'] for call in result['skill_calls']} == {'pick', 'place', 'stack'}
    assert {call['arm'] for call in result['skill_calls']} == {0, 1}
