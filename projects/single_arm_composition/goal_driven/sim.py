"""Execute and score outcome-conditioned decisions in the frozen workcell."""

from larger_scale.rollout import initial_digest
from larger_scale.scene import SCAN, WORK
from larger_scale.workcell import OPEN as LID_OPEN, Outcome, Workcell

from .planner import (
    ACTIONS, CLOSED, DISPLACED, FEED, LATCHED, OPEN, PROCESS, READER,
    RECOVERY, SORTED, UNKNOWN, State, optimal, reached, transition, violates_order,
)


class StartFailure(RuntimeError):
    pass


def observe(cell):
    locations = []
    for job, tag in enumerate(cell.locations()):
        bin_name = "reject" if cell.grades[job] else "release"
        sorted_slot = f"{bin_name}{cell.destinations[job]}"
        locations.append({
            cell.homes[job]: FEED, SCAN: READER, WORK: PROCESS,
            f"recovery{job}": RECOVERY, sorted_slot: SORTED,
        }.get(tag, DISPLACED))
    lid, bolt = cell.j("lid"), cell.j("bolt")
    if lid > LID_OPEN - .007 and bolt < .006:
        enclosure = OPEN
    elif lid < .007 and bolt < .006:
        enclosure = CLOSED
    elif lid < .007 and bolt > .022:
        enclosure = LATCHED
    else:
        enclosure = UNKNOWN
    return State(tuple(locations), tuple(grade >= 0 for grade in cell.reads),
                 tuple(cell.cycles), enclosure)


def prepare(cell, task):
    recovery = tuple(i for i, where in enumerate(task.start.locations) if where == RECOVERY)
    cell.reset(task.seed, task.binding, task.grades, recovery, task.destinations)
    for job, where in enumerate(task.start.locations):
        staging = ()
        if where == READER:
            staging = ("read_load",)
        elif where == PROCESS:
            staging = ("load",)
            if task.start.enclosure in (CLOSED, LATCHED):
                staging += ("close",)
            if task.start.enclosure == LATCHED:
                staging += ("latch",)
        for kind in staging:
            outcome = cell.execute(kind, job)
            if not outcome.ok:
                raise StartFailure(f"{kind}:{outcome.reason}")
    observed = observe(cell)
    if observed != task.start:
        raise StartFailure(f"expected {task.start}, measured {observed}")


def describe(goal):
    clauses = []
    for job, target in enumerate(goal.targets):
        name = chr(65 + job)
        if target.inspect:
            clauses.append(f"inspect {name}")
        if target.cycles:
            clauses.append(f"index {name} {target.cycles}x")
        if target.destination == FEED:
            clauses.append(f"return {name} to feed")
        elif target.destination == SORTED:
            clauses.append(f"sort {name} by grade")
    if goal.read_before_index:
        a, b = goal.read_before_index
        clauses.append(f"read {chr(65 + a)} before indexing {chr(65 + b)}")
    return ", ".join((*clauses, "leave the enclosure open and unlocked"))


def execute(cell, action):
    kind, job = action
    if kind == "sort":
        grade = cell.reads[job]
        if grade < 0:
            return Outcome(False, "unobserved_grade")
        kind = ("release", "reject")[grade]
    return cell.execute(kind, job)


def symbolic(goal, state):
    choices = optimal(goal, state).get(state, ())
    return choices[0] if choices else None


def rollout(task, choose, *, cell=None, before=None, label=False, reference=True):
    cell = cell if cell is not None else Workcell(task.layout)
    try:
        prepare(cell, task)
    except StartFailure as failure:
        return {"task": task.name, "success": False, "reason": f"staging:{failure}",
                "start": None, "actions": [], "optimal_steps": None, "invalid": 0,
                "decisions": []}
    start = initial_digest(cell)
    state = observe(cell)
    reason = ""
    invalid = 0
    actions = []
    decisions = []
    for _ in range(64):
        if reached(task.goal, state):
            break
        index = choose(task.goal, state)
        if index is None:
            reason = "no_plan"
            break
        kind, job = ACTIONS[index]
        if before:
            before(task.goal, state, index)
        if label:
            decisions.append((state, optimal(task.goal, state).get(state, ())))
        rule = task.goal.read_before_index
        if kind == "index" and rule and job == rule[1] and not state.reads[rule[0]]:
            reason = "ordering"
            invalid += 1
            actions.append([kind, job, False, reason])
            break
        outcome = execute(cell, (kind, job))
        actions.append([kind, job, outcome.ok, outcome.reason])
        if not outcome.ok:
            reason = outcome.reason
            invalid += reason in ("precondition", "unobserved_grade")
            break
        state = observe(cell)
        if any(count > target.cycles for count, target in zip(state.cycles, task.goal.targets)):
            reason = "extra_index"
            invalid += 1
            break
    success = reached(task.goal, state) and not violates_order(task.goal, state) and not reason
    best = None
    if reference:
        cursor = task.start
        plan = optimal(task.goal, cursor)
        if plan or reached(task.goal, cursor):
            best = 0
            while cursor in plan:
                cursor = transition(task.goal, cursor, ACTIONS[plan[cursor][0]])
                best += 1
    return {"task": task.name, "success": bool(success), "start": start,
            "reason": "" if success else (reason or "step_limit"),
            "actions": actions, "optimal_steps": best,
            "invalid": invalid, "decisions": decisions}
