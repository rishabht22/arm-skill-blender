"""One-payload scene interface shared by training, inference, and graph search."""

from collections import deque
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

FEATURE_NAMES = (
    'target_is_requested_goal',
    'arm_can_reach_requested_goal',
    'another_arm_can_reach_target',
)
FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class State:
    location: str | None
    holder: int | None
    reach: Mapping[str, Sequence[int]]
    goal: str = 'goal'

    def __post_init__(self):
        if (self.location is None) == (self.holder is None):
            raise ValueError('A payload must be at a location or held by an arm')
        if self.goal not in self.reach or (self.location is not None and self.location not in self.reach):
            raise ValueError('Payload location and goal must refer to sites in the scene')

    @property
    def complete(self):
        return self.location == self.goal


@dataclass(frozen=True)
class Action:
    skill: str
    arm: int
    target: str


def candidates(state: State) -> list[Action]:
    """Enumerate one-step affordances. Detours and unproductive calls stay legal."""
    if state.complete:
        return []
    if state.location is not None:
        return [Action('pick', arm, state.location) for arm in sorted(state.reach[state.location])]
    return [Action('place', state.holder, site) for site, arms in sorted(state.reach.items())
            if state.holder in arms]


def encode(state: State, choices: Sequence[Action]) -> np.ndarray:
    """Score role-relative actions using only goal and current reach relations."""
    return np.asarray([
        [choice.target == state.goal,
         choice.arm in state.reach[state.goal],
         any(arm != choice.arm for arm in state.reach[choice.target])]
        for choice in choices
    ], dtype=np.float32).reshape(-1, FEATURES)


def transition(state: State, action: Action) -> State:
    if action not in candidates(state):
        raise ValueError(f'Action is unavailable in the current scene: {action}')
    if action.skill == 'pick':
        return replace(state, location=None, holder=action.arm)
    return replace(state, location=action.target, holder=None)


def shortest_plan(state: State) -> list[Action]:
    """Reference search. Its choices are excluded from imitation training."""
    queue = deque([(state, [])])
    seen = {(state.location, state.holder)}
    while queue:
        current, plan = queue.popleft()
        if current.complete:
            return plan
        for action in candidates(current):
            following = transition(current, action)
            key = following.location, following.holder
            if key not in seen:
                seen.add(key)
                queue.append((following, [*plan, action]))
    raise ValueError('No sequence reaches the requested goal in this scene')
