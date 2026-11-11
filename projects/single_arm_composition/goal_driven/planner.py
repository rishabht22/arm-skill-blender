"""Search over skill effects from outcome goals and observed cell state."""

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from functools import lru_cache

from larger_scale.scene import CARRIERS
from larger_scale.workcell import PRIMITIVES


FEED, READER, PROCESS, RECOVERY, SORTED, DISPLACED = range(6)
OPEN, CLOSED, LATCHED, UNKNOWN = range(4)
SKILLS = (*[kind for kind in PRIMITIVES if kind not in ("release", "reject")], "sort")
ACTIONS = tuple((kind, job) for kind in SKILLS for job in range(CARRIERS))


@dataclass(frozen=True)
class Target:
    cycles: int = 0
    inspect: bool = False
    destination: int | None = None


@dataclass(frozen=True)
class Goal:
    targets: tuple[Target, ...]
    # Inspection of the first job must precede the first index of the second.
    read_before_index: tuple[int, int] | None = None


@dataclass(frozen=True)
class State:
    locations: tuple[int, ...] = (FEED,) * CARRIERS
    reads: tuple[bool, ...] = (False,) * CARRIERS
    cycles: tuple[int, ...] = (0,) * CARRIERS
    enclosure: int = OPEN


def at(values, index, value):
    return values[:index] + (value,) + values[index + 1:]


def reached(goal, state):
    return state.enclosure == OPEN and all(
        state.cycles[job] == target.cycles
        and (not target.inspect or state.reads[job])
        and (target.destination is None or state.locations[job] == target.destination)
        for job, target in enumerate(goal.targets)
    )


def violates_order(goal, state):
    if goal.read_before_index is None:
        return False
    reader, process = goal.read_before_index
    return state.cycles[process] > 0 and not state.reads[reader]


def transition(goal, state, action):
    """Predict one useful skill effect without stepping MuJoCo."""
    kind, job = action
    where = state.locations[job]
    enclosure = state.enclosure
    target = goal.targets[job]
    needs_index = state.cycles[job] < target.cycles
    needs_read = target.inspect and not state.reads[job]
    if kind == "reseat" and where == RECOVERY and target.destination is not None:
        location = FEED
    elif kind == "read_load" and where == FEED and needs_read and READER not in state.locations:
        location = READER
    elif kind == "read" and where == READER and needs_read:
        return replace(state, reads=at(state.reads, job, True))
    elif kind == "read_return" and where == READER and not needs_read:
        location = FEED
    elif kind == "load" and where == FEED and needs_index and enclosure == OPEN and PROCESS not in state.locations:
        location = PROCESS
    elif kind == "close" and where == PROCESS and needs_index and enclosure == OPEN:
        return replace(state, enclosure=CLOSED)
    elif kind == "latch" and where == PROCESS and needs_index and enclosure == CLOSED:
        return replace(state, enclosure=LATCHED)
    elif kind == "index" and where == PROCESS and needs_index and enclosure == LATCHED:
        next_state = replace(state, cycles=at(state.cycles, job, state.cycles[job] + 1))
        return None if violates_order(goal, next_state) else next_state
    elif kind == "unlatch" and where == PROCESS and enclosure == LATCHED:
        return replace(state, enclosure=CLOSED)
    elif kind == "open" and where == PROCESS and enclosure == CLOSED:
        return replace(state, enclosure=OPEN)
    elif kind == "unload" and where == PROCESS and enclosure == OPEN:
        location = FEED
    elif (kind == "sort" and where == FEED and target.destination == SORTED
          and state.reads[job] and not needs_index):
        location = SORTED
    else:
        return None
    return replace(state, locations=at(state.locations, job, location))


@lru_cache(maxsize=4096)
def optimal(goal, start):
    """Find all equally short first moves along shortest paths from start."""
    if reached(goal, start) or violates_order(goal, start):
        return {}
    queue = deque([start])
    depths = {start: 0}
    parents = defaultdict(list)
    solutions = []
    solution_depth = None
    while queue:
        state = queue.popleft()
        depth = depths[state]
        if solution_depth is not None and depth >= solution_depth:
            break
        for index, action in enumerate(ACTIONS):
            successor = transition(goal, state, action)
            if successor is None:
                continue
            if successor not in depths:
                depths[successor] = depth + 1
                if reached(goal, successor):
                    solutions.append(successor)
                    solution_depth = depth + 1
                else:
                    queue.append(successor)
            if depths[successor] == depth + 1:
                parents[successor].append((state, index))
    choices = defaultdict(set)
    pending = set(solutions)
    visited = set()
    while pending:
        successor = pending.pop()
        if successor in visited:
            continue
        visited.add(successor)
        for state, index in parents[successor]:
            choices[state].add(index)
            pending.add(state)
    return {state: tuple(sorted(indices)) for state, indices in choices.items()}
