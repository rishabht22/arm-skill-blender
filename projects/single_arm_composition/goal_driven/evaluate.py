"""Resumable, paired physical benchmark for outcome-conditioned policies."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

from larger_scale.collect import atomic_gzip, get_cell, read_rows
from larger_scale.protocol import blob, digest, write_json
from larger_scale.rollout import controller_id

from .learn import load, sha, source_id
from .planner import ACTIONS, State, optimal, transition
from .sim import rollout, symbolic
from .tasks import SUITES, tasks


WORK = {}


def fixed_script(goal):
    state = State()
    choices = optimal(goal, state)
    while state in choices:
        index = choices[state][0]
        yield index
        state = transition(goal, state, ACTIONS[index])


def init(models):
    import torch
    torch.set_num_threads(1)
    WORK.clear()
    WORK.update((name, load(path)) for name, path in models)


def episode_batch(path, scenarios):
    if path.exists():
        return len(scenarios)
    with atomic_gzip(path) as stream:
        for task in scenarios:
            cell = get_cell(task.layout)
            script = fixed_script(task.goal)
            methods = {"planner": symbolic, "fixed_feed_plan": lambda *_: next(script, None), **WORK}
            start = set()
            for name, choose in methods.items():
                result = rollout(task, choose, cell=cell)
                start.add(result["start"])
                del result["decisions"]
                result.update(method=name, scenario=asdict(task))
                stream.write(blob(result) + b"\n")
            if len(start) != 1:
                raise RuntimeError(f"paired start mismatch for {task.name}")
    return len(scenarios)


def evaluate(args):
    if min(args.count, args.workers, args.batch, args.parts) < 1 or not 0 <= args.part < args.parts:
        raise ValueError("invalid evaluation or partition size")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = [Path(path).resolve() for path in args.model]
    models = [(f"learned:{sha(path)[:12]}", path) for path in paths]
    if len(dict(models)) != len(models):
        raise ValueError("model checkpoints must be unique")
    contract = {"controller": controller_id(), "sources": source_id(),
                "suite": args.suite, "count": args.count, "batch": args.batch,
                "models": {name: sha(path) for name, path in models}}
    contract_path = output / f"part-{args.part:05d}-of-{args.parts:05d}.json"
    for existing in output.glob("part-*-of-*.json"):
        if json.loads(existing.read_text()) != contract:
            raise ValueError("output already contains a different benchmark")
    write_json(contract_path, contract)
    scenarios = list(tasks(args.suite, args.count))
    bounds = [(first, min(first + args.batch, len(scenarios)))
              for first in range(0, len(scenarios), args.batch)]
    assigned = [(output / f"{first:07d}-{last:07d}.jsonl.gz", scenarios[first:last])
                for index, (first, last) in enumerate(bounds) if index % args.parts == args.part]
    started = time.monotonic()
    completed = 0
    if args.workers == 1:
        init(models)
        for path, batch in assigned:
            completed += episode_batch(path, batch)
            print({"completed_starts": completed, "seconds": time.monotonic() - started}, flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(models,)) as pool:
            for count in pool.map(episode_batch, (a for a, _ in assigned), (b for _, b in assigned)):
                completed += count
                print({"completed_starts": completed, "seconds": time.monotonic() - started}, flush=True)


def report(directories, output):
    result = {}
    for directory in map(Path, directories):
        contracts = [json.loads(path.read_text()) for path in directory.glob("part-*-of-*.json")]
        if not contracts or any(contract != contracts[0] for contract in contracts):
            raise ValueError(f"missing or inconsistent contract in {directory}")
        contract = contracts[0]
        suite = contract["suite"]
        if suite in result:
            raise ValueError(f"duplicate suite {suite}")
        expected = {task.name: digest(asdict(task)) for task in tasks(suite, contract["count"])}
        methods = {"planner", "fixed_feed_plan", *contract["models"]}
        paired = {}
        for path in directory.glob("*.jsonl.gz"):
            for row in read_rows(path):
                key = row["task"]
                if digest(row["scenario"]) != expected.get(key):
                    raise ValueError(f"unexpected scenario {key}")
                present = paired.setdefault(key, {})
                if row["method"] in present:
                    raise ValueError(f"duplicate method for {key}")
                present[row["method"]] = row
        if set(paired) != set(expected):
            raise ValueError(f"{suite} has {len(paired)} of {len(expected)} starts")
        for key, rows in paired.items():
            if set(rows) != methods or len({row["start"] for row in rows.values()}) != 1:
                raise ValueError(f"unpaired records at {key}")
        statistics = {}
        for method in sorted(methods):
            rows = [match[method] for match in paired.values()]
            successes = [row for row in rows if row["success"]]
            excess = [len(row["actions"]) - row["optimal_steps"] for row in successes
                      if row["optimal_steps"] is not None]
            statistics[method] = {"successes": len(successes), "rollouts": len(rows),
                                  "invalid_requests": sum(row["invalid"] for row in rows),
                                  "failures": dict(Counter(row["reason"] for row in rows if not row["success"])),
                                  "mean_excess_actions": sum(excess) / len(excess) if excess else None,
                                  "paired_success_difference_vs_planner":
                                      len(successes) - sum(match["planner"]["success"] for match in paired.values())}
        result[suite] = {
            "contract": contract, "starts": len(paired),
            "distinct_goals": len({digest(rows["planner"]["scenario"]["goal"]) for rows in paired.values()}),
            "methods": statistics,
        }
    write_json(output, result)
    print(json.dumps(result, indent=2))
