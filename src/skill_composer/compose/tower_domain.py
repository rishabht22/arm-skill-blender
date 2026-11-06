"""Goal relations and local affordances for the multi-payload workbench."""

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TowerAction:
    skill: str
    arm: int
    obj: str
    target: str


@dataclass(frozen=True)
class TowerState:
    positions: Mapping[str, str]
    holders: Mapping[str, int]
    reach: Mapping[str, Sequence[int]]
    goals: Mapping[str, str]

    @property
    def complete(self):
        return all(self.positions.get(obj) == support for obj, support in self.goals.items())

    def key(self):
        return tuple(sorted(self.positions.items())), tuple(sorted(self.holders.items()))


def root(support, relations, pads):
    """Resolve a support chain to the underlying pad, if grounded."""
    seen = set()
    while support not in pads:
        if support in seen or support not in relations:
            return None
        seen.add(support)
        support = relations[support]
    return support


def candidates(state: TowerState) -> list[TowerAction]:
    """Generate executable calls, including detours and wrong-order choices."""
    if state.complete:
        return []
    covered = set(state.positions.values())
    busy = set(state.holders.values())
    choices = []
    for obj, support in sorted(state.positions.items()):
        if obj in covered:
            continue
        pad = root(support, state.positions, state.reach)
        if pad is not None:
            choices.extend(TowerAction('pick', arm, obj, support)
                           for arm in sorted(state.reach[pad]) if arm not in busy)
    supports = (set(state.reach) | set(state.positions)) - covered
    for obj, arm in sorted(state.holders.items()):
        for support in sorted(supports):
            pad = root(support, state.positions, state.reach)
            if pad is not None and arm in state.reach[pad]:
                skill = 'place' if support in state.reach else 'stack'
                choices.append(TowerAction(skill, arm, obj, support))
    return choices


def _move(state, action):
    positions, holders = dict(state.positions), dict(state.holders)
    if action.skill == 'pick':
        del positions[action.obj]
        holders[action.obj] = action.arm
    else:
        del holders[action.obj]
        positions[action.obj] = action.target
    return replace(state, positions=positions, holders=holders)


def transition(state: TowerState, action: TowerAction) -> TowerState:
    if action not in candidates(state):
        raise ValueError(f'Infeasible call {action}')
    return _move(state, action)


def _lower_bound(state):
    return sum(1 if obj in state.holders else 2 for obj, support in state.goals.items()
               if state.positions.get(obj) != support)


def shortest_plan(state: TowerState, rank=None, max_expansions=30_000) -> list[TowerAction]:
    """A* in skill-call count. An optional learned score breaks search ties."""
    serial = count()
    queue = [(_lower_bound(state), 0, next(serial), state, ())]
    distances = {state.key(): 0}
    for _ in range(max_expansions):
        if not queue:
            raise ValueError('The goal is unreachable under the current affordances')
        _, _, _, current, plan = heappop(queue)
        if len(plan) != distances[current.key()]:
            continue
        if current.complete:
            return list(plan)
        choices = candidates(current)
        scores = rank(current, choices) if rank is not None and choices else [0.] * len(choices)
        for action, score in zip(choices, scores):
            following = _move(current, action)
            distance = len(plan) + 1
            key = following.key()
            if distance < distances.get(key, float('inf')):
                distances[key] = distance
                heappush(queue, (distance + _lower_bound(following), -float(score), next(serial),
                                 following, (*plan, action)))
    raise ValueError('Search reached its expansion budget')


FEATURE_NAMES = (
    'pick', 'place', 'stack', 'object_has_goal', 'object_relation_satisfied',
    'object_support_chain_satisfied', 'target_matches_goal', 'requested_support_available',
    'acting_arm_reaches_goal_pad', 'another_arm_reaches_target', 'goal_arm_reaches_target',
    'target_reserved_as_goal_for_another_object', 'target_is_block',
    'goal_depth', 'blocking_goal_pad', 'other_arm_holding',
)


def _settled(obj, state):
    current = obj
    seen = set()
    while current not in state.reach:
        if current in seen or current not in state.goals or state.positions.get(current) != state.goals[current]:
            return False
        seen.add(current)
        current = state.goals[current]
    return True


def encode(state: TowerState, choices: Sequence[TowerAction]) -> np.ndarray:
    """Ground each candidate in the scene and requested on-relations, independent of IDs."""
    rows = []
    occupied = set(state.positions.values())
    for action in choices:
        obj, target, arm = action.obj, action.target, action.arm
        requested = state.goals.get(obj)
        goal_pad = root(requested, state.goals, state.reach)
        target_pad = root(target, state.positions, state.reach)
        goal_arms = state.reach.get(goal_pad, ())
        target_arms = state.reach.get(target_pad, ())
        # The desired goal chain is supplied as predicates, not a sequence of calls.
        depth, support, seen = 0, requested, set()
        while support in state.goals and support not in seen:
            seen.add(support)
            depth += 1
            support = state.goals[support]
        rows.append([
            action.skill == 'pick', action.skill == 'place', action.skill == 'stack',
            requested is not None,
            requested is not None and state.positions.get(obj) == requested,
            _settled(obj, state),
            target == requested,
            requested not in occupied and _settled(requested, state),
            arm in goal_arms,
            any(peer != arm for peer in target_arms),
            any(peer in target_arms for peer in goal_arms),
            any(other != obj and site == target for other, site in state.goals.items()),
            target not in state.reach,
            min(depth, 3) / 3,
            action.skill == 'pick' and any(other != obj and site == target for other, site in state.goals.items()),
            any(peer != arm for peer in state.holders.values()),
        ])
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
