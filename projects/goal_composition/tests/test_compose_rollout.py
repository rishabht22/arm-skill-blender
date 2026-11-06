import pytest

from skill_composer.compose.domain import Action, shortest_plan
from skill_composer.compose.rollout import rollout


@pytest.mark.parametrize(
    "layout,goal,expected",
    [
        ("direct", "goal", [("pick", 1, "dock0"), ("place", 1, "goal")]),
        ("relay", "goal", [("pick", 0, "source"), ("place", 0, "dock0"),
                           ("pick", 1, "dock0"), ("place", 1, "goal")]),
        ("relay-swapped", "goal", [("pick", 0, "source"), ("place", 0, "dock1"),
                                   ("pick", 1, "dock1"), ("place", 1, "goal")]),
        ("direct", "source", [("pick", 0, "dock0"), ("place", 0, "source")]),
    ],
)
def test_reference_executes_goal_dependent_skill_sequences(layout, goal, expected):
    result = rollout(layout=layout, goal=goal)
    assert result["success"], result["failure"]
    assert result["goal_error_m"] < 0.025
    assert [(call["skill"], call["arm"], call["target"]) for call in result["skill_calls"]] == expected


def test_private_dock_is_executable_and_does_not_complete_the_goal():
    def choose_detour(state):
        if state.holder == 0:
            return Action("place", 0, "dock1")
        return shortest_plan(state)[0]

    result = rollout("relay", chooser=choose_detour, max_calls=2)
    assert result["skill_calls"][-1]["resulting_location"] == "dock1"
    assert not result["success"]
    assert result["failure"] == "Goal not achieved within skill budget"
