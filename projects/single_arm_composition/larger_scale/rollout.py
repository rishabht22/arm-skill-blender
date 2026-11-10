"""Shared execution, baselines, physical outcome scoring and recording."""
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import mujoco as mj
import numpy as np

from .protocol import actions, digest, start_seed
from .scene import CARRIERS, PANDA, SCAN, WORK, layout
from .workcell import OPEN, Workcell


LOCATION_KEYS = ("displaced", WORK, SCAN, *(f"{kind}{job}" for kind in
                 ("feed", "release", "reject", "recovery") for job in range(CARRIERS)))
LOCATION_IDS = {name: i for i, name in enumerate(LOCATION_KEYS)}


@lru_cache(maxsize=1)
def controller_id():
    directory = Path(__file__).resolve().parent
    files = [directory / "scene.py", directory / "workcell.py", PANDA / "panda.xml",
             *sorted((PANDA / "assets").iterdir())]
    return digest({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files})


@lru_cache(maxsize=1)
def protocol_id():
    path = Path(__file__).resolve().parent
    files = ("protocol.py", "rollout.py")
    return digest({name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in files})


def action_name(action, cell):
    if action.kind != "sort":
        return action.kind
    return ("release", "reject")[cell.reads[action.job]] if cell.reads[action.job] >= 0 else None


def ready(cell, nodes, done):
    result = []
    for i, a in enumerate(nodes):
        if (done >> i) & 1 or any(not ((done >> p) & 1) for p in a.predecessors):
            continue
        kind = action_name(a, cell)
        if kind and cell.available(kind, a.job):
            result.append(i)
    return result


def action_target(cell, action):
    kind, job = action.kind, action.job
    if kind in ("load", "read_load", "sort", "reseat", "unload", "read_return"):
        return cell.position(cell.carrier(job))
    if kind in ("close", "open"):
        return cell.geom("lid_handle")
    if kind in ("latch", "unlatch"):
        return cell.geom("bolt_handle")
    if kind == "index":
        return cell.geom("button_cap")
    return cell.hand()


def observation(cell):
    return {
        "locations": [LOCATION_IDS[tag] for tag in cell.locations()],
        "reads": cell.reads.copy(),
        "cycles": cell.cycles.copy(),
        "mechanism": [round(cell.j("lid") / OPEN, 5), round(cell.j("bolt") / .026, 5),
                      round(cell.j("rotor") / (np.pi / 2), 5)],
        "tcp": np.round(cell.hand(), 5).tolist(),
    }


def expert_choice(cell, nodes, candidates, tried, previous):
    tcp = cell.hand()
    return min(candidates, key=lambda i: (
        tried[i],
        # Finish active transactions before moving to a different station.
        nodes[i].kind in ("load", "read_load", "sort", "reseat"),
        round(float(np.linalg.norm(action_target(cell, nodes[i]) - tcp)), 4),
        nodes[i].job != previous, i,
    ))


def topological_script(nodes):
    remaining = set(range(len(nodes)))
    script = []
    while remaining:
        i = min(i for i in remaining if not any(p in remaining for p in nodes[i].predecessors))
        script.append(i)
        remaining.remove(i)
    return script


class Retrieval:
    """Nearest train DAG with the same route multiset, replayed as a fixed plan."""
    def __init__(self, train_graphs):
        self.routes = defaultdict(list)
        for gid, graph in sorted(train_graphs.items()):
            self.routes[graph.lanes].append((gid, graph, frozenset(graph.links)))

    def script(self, graph, recovery):
        matched = self.routes[graph.lanes]
        if matched:
            query = frozenset(graph.links)
            _, exemplar, _ = min(matched, key=lambda item: (len(query ^ item[2]), item[0]))
            return topological_script(actions(exemplar, recovery))
        return list(range(len(actions(graph, recovery))))


def initial_digest(cell):
    d = cell.data
    raw = d.qpos.tobytes() + d.qvel.tobytes() + d.ctrl.tobytes()
    return hashlib.sha256(raw).hexdigest()


def physical_digest(cell):
    return digest((np.round(cell.data.qpos, 9).tolist(), np.round(cell.data.qvel, 9).tolist()))


def target_slots(item):
    return tuple(f"{'reject' if item.grades & (1 << job) else 'release'}{item.destinations[job]}"
                 for job in range(CARRIERS))


def run(graph, item, repetition, manifest, method, *, cell=None, selector=None,
        retrieval=None, keep_decisions=False, record=False, rank=None):
    cell = cell or Workcell(item.layout)
    seed = start_seed(manifest, item, repetition)
    cell.reset(seed, item.binding, item.grades, item.recovery, item.destinations)
    start = initial_digest(cell)
    if record:
        cell.trace = []
    nodes = actions(graph, item.recovery)
    script = None
    if method == "fixed":
        script = list(range(len(nodes)))
    if method == "retrieval":
        script = retrieval.script(graph, item.recovery)
    done = 0
    violations = 0
    physical_failures = 0
    tried = defaultdict(int)
    reason = ""
    attempts = []
    decisions = []
    previous = -1
    for step in range(len(nodes) + (10 if method == "expert" else 0)):
        if done.bit_count() == len(nodes):
            break
        admissible = ready(cell, nodes, done) if method in ("expert", "symbolic") or keep_decisions else []
        if method == "learned":
            idx = selector(nodes, item, observation(cell), done)
        elif script is not None:
            idx = script[step]
        else:
            candidates = [i for i in admissible if tried[i] < (3 if method == "expert" else 1)]
            if not candidates:
                reason = "no_admissible_action"
                break
            idx = (expert_choice(cell, nodes, candidates, tried, previous) if method == "expert"
                   else min(candidates))
        if not 0 <= idx < len(nodes) or (done >> idx) & 1:
            reason = "invalid_action_index"
            break
        action = nodes[idx]
        kind = action_name(action, cell)
        dep_ok = all((done >> p) & 1 for p in action.predecessors)
        if keep_decisions:
            decisions.append([observation(cell), format(done, "x"),
                              format(sum(1 << i for i in admissible), "x"), idx])
        if not dep_ok:
            # An unsafe scheduler request is logged before physical actuation.
            violations += 1
            reason = "dependency"
            attempts.append([idx, kind, False, reason])
            break
        if kind is None:
            reason = "unobserved_grade"
            attempts.append([idx, kind, False, reason])
            physical_failures += 1
            break
        tried[idx] += 1
        outcome = cell.execute(kind, action.job)
        attempts.append([idx, kind, outcome.ok, outcome.reason])
        if outcome.ok:
            done |= 1 << idx
            reason = ""
            previous = action.job
        else:
            physical_failures += 1
            reason = outcome.reason
            if method != "expert":
                break
    terminal = cell.locations()
    target = target_slots(item)
    planned_cycles = tuple(lane.count("P") for lane in graph.lanes)
    physical = (terminal == target and tuple(cell.cycles) == planned_cycles
                and tuple(cell.reads) == cell.grades and cell.j("lid") > OPEN - .007
                and cell.j("bolt") < .006)
    recovered = sum(1 for idx, node in enumerate(nodes) if node.kind == "reseat" and (done >> idx) & 1)
    log = {
        "task_id": item.task_id, "graph": item.graph_id, "family": item.family,
        "split": item.split, "layout": item.layout, "reset": repetition, "seed": seed,
        "start": start, "final_physics": physical_digest(cell), "rank": rank,
        "manifest": manifest, "controller": controller_id(), "protocol": protocol_id(),
        "method": method, "success": bool(physical and done.bit_count() == len(nodes) and not violations),
        "physical_goal": bool(physical), "completed": done.bit_count(), "subgoals": len(nodes),
        "violations": violations, "skill_failures": physical_failures,
        "reason": reason or ("" if physical else "physical_terminal"),
        "recovered": recovered, "recovery_goals": len(item.recovery),
        "locations": terminal, "goal_locations": target, "cycles": cell.cycles,
        "target_cycles": planned_cycles, "reads": cell.reads, "attempts": attempts,
        "sim_seconds": cell.data.time,
    }
    if keep_decisions:
        log["decisions"] = decisions
    return log, cell.trace


def save_recording(path, log, trace):
    if trace is None:
        raise ValueError("recording requires physics commands")
    ticks, controls, active, eq_data, panels = zip(*trace)
    np.savez_compressed(path, ticks=np.asarray(ticks), controls=np.asarray(controls),
                        active=np.asarray(active), eq_data=np.asarray(eq_data),
                        panels=np.asarray(panels), provenance=json.dumps(log, separators=(",", ":")))


def replay(path, item, manifest, *, frame=None):
    with np.load(path, allow_pickle=False) as recording:
        log = json.loads(recording["provenance"].item())
        if (log["manifest"], log["controller"], log["task_id"]) != (
                manifest, controller_id(), item.task_id):
            raise ValueError("recording provenance mismatch")
        cell = Workcell(item.layout)
        cell.reset(log["seed"], item.binding, item.grades, item.recovery, item.destinations)
        if initial_digest(cell) != log["start"]:
            raise ValueError("recording start mismatch")
        for ticks, control, active, eq_data, panel in zip(
                recording["ticks"], recording["controls"], recording["active"],
                recording["eq_data"], recording["panels"]):
            cell.data.ctrl[:] = control
            cell.data.eq_active[:] = active
            cell.model.eq_data[:] = eq_data
            cell.advance(int(ticks), panel=bool(panel))
            if frame is not None:
                frame(cell)
        if physical_digest(cell) != log["final_physics"]:
            raise ValueError("recording diverged from measured final state")
    return log
