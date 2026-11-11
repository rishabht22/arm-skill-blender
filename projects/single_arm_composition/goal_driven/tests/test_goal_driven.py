from dataclasses import replace

from goal_driven.planner import ACTIONS, FEED, LATCHED, PROCESS, Goal, State, Target, at, optimal, reached, transition
from goal_driven.sim import rollout, symbolic
from goal_driven.tasks import showcase, tasks


def sequence(goal, state):
    actions = []
    plan = optimal(goal, state)
    while state in plan:
        action = ACTIONS[plan[state][0]]
        actions.append(action)
        state = transition(goal, state, action)
    assert reached(goal, state)
    return actions


def test_same_outcome_changes_first_action():
    feed = showcase()
    loaded = showcase(loaded=True)
    assert feed.goal == loaded.goal
    first = sequence(feed.goal, feed.start)
    second = sequence(loaded.goal, loaded.start)
    assert first == [("load", 1), *second]
    assert second[0] == ("close", 1)


def test_two_cycles_share_one_loading_and_latching():
    task = showcase()
    goal = replace(task.goal, targets=at(task.goal.targets, 1, Target(cycles=2, destination=FEED)))
    plan = sequence(goal, task.start)
    assert plan.count(("index", 1)) == 2
    assert plan.count(("load", 1)) == 1
    assert plan.count(("latch", 1)) == 1


def test_explicit_order_changes_choice_with_latched_carrier():
    goal = Goal((Target(inspect=True, destination=FEED), Target(cycles=1, destination=FEED),
                 Target(), Target()), (0, 1))
    start = State(locations=(FEED, PROCESS, FEED, FEED), enclosure=LATCHED)
    assert transition(goal, start, ("index", 1)) is None
    plan = sequence(goal, start)
    assert plan.index(("read", 0)) < plan.index(("index", 1))


def test_goal_holdouts_and_start_conditions():
    train = list(tasks("train", 1800))
    seen = {task.goal for task in train}
    for suite in ("novel_goal", "more_carriers", "ordering"):
        assert all(task.goal not in seen for task in tasks(suite, 256))
    train_states = {task.start for task in train}
    assert all(task.start not in train_states for task in tasks("partial", 256))


def test_physical_showcase_and_fixed_script_failure():
    feed, loaded = showcase(), showcase(loaded=True)
    for task, first in ((feed, "load"), (loaded, "close")):
        result = rollout(task, symbolic)
        assert result["success"], result
        assert result["actions"][0][:2] == [first, 1]
    fixed = iter(ACTIONS.index(action) for action in sequence(feed.goal, feed.start))
    result = rollout(loaded, lambda *_: next(fixed, None))
    assert not result["success"]
    assert result["reason"] == "precondition"
