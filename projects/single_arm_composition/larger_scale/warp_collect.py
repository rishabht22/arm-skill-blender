"""Validate and collect Aster GPU episodes in a separate experiment directory."""

import argparse
from collections import Counter, defaultdict
from functools import partial
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

from .checks import controls
from .collect import atomic_gzip, ranks
from .protocol import blob, digest, read_manifest, write_json
from .rollout import controller_id, protocol_id, run
from .warp_sim import WarpBatch
from .workcell import PHASES, PRIMITIVES


def backend():
    import warp as wp

    if wp.get_cuda_device_count() != 1:
        raise ValueError("Select exactly one idle NVIDIA GPU with CUDA_VISIBLE_DEVICES")
    device = wp.get_device("cuda:0")
    folder = Path(__file__).parent
    return {
        "kind": "mujoco-warp-cuda",
        "gpu": device.name,
        "versions": {name: version(name) for name in ("mujoco", "mujoco-warp", "warp-lang", "numpy", "scipy")},
        "sources": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                    for name in ("warp_sim.py", "warp_collect.py", "checks.py")},
        "controller": controller_id(),
        "protocol": protocol_id(),
    }


def positive_trial(cell, job):
    grade = job % 2
    cell.reset(30101 + job, (1, 3, 2, 0), grade << job, (job,))
    kinds = ("reseat", *PHASES[job % 2], *PHASES[1 - job % 2], "reject" if grade else "release")
    result = {}
    for kind in kinds:
        outcome = cell.execute(kind, job)
        result[kind] = {"ok": outcome.ok, "reason": outcome.reason}
        if not outcome.ok:
            break
    return result


def counterexample_trial(cell):
    try:
        return controls(cell)
    except AssertionError:
        return [{"case": "counterexample_setup", "passed": False,
                 "locations": cell.locations(), "lid": cell.j("lid"), "bolt": cell.j("bolt")}]


def check_backend(output, identity):
    start = time.monotonic()
    counterexamples, *trials = WarpBatch(0, 5, "cuda:0").run([
        counterexample_trial, *(lambda cell, job=job: positive_trial(cell, job) for job in range(4)),
    ])
    covered = {kind for trial in trials for kind, result in trial.items() if result["ok"]}
    passed = (covered == set(PRIMITIVES) and all(len(trial) == 12 for trial in trials)
              and all(all(result["ok"] for result in trial.values()) for trial in trials)
              and all(row["passed"] for row in counterexamples))
    report = {"backend": identity, "passed": passed, "counterexamples": counterexamples,
              "primitive_trials": trials, "seconds": time.monotonic() - start}
    write_json(output, report)
    return report


def expert(cell, graph, item, repetition, manifest, rank, identity):
    row, _ = run(graph, item, repetition, manifest, "expert", cell=cell, keep_decisions=True, rank=rank)
    row["controller"] = digest(identity)
    row["physics_backend"] = identity["kind"]
    return row


def collect_gpu(args, identity):
    gate = json.loads(Path(args.gate).read_bytes())
    if not gate["passed"] or gate["backend"] != identity:
        raise ValueError("Run a passing GPU check for this backend before collecting")
    if min(args.count, args.shard_size, args.worlds) < 1:
        raise ValueError("count, shard size and worlds must be positive")
    graphs, items, manifest = read_manifest(args.manifest, args.split)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "warp-contract.json"
    contract = {"backend": identity, "manifest": manifest, "split": args.split, "count": args.count,
                "shard_size": args.shard_size, "worlds": args.worlds}
    if path.exists():
        if json.loads(path.read_bytes()) != contract:
            raise ValueError("existing GPU output has a different contract")
    else:
        if any(output.iterdir()):
            raise ValueError("choose an empty directory for the GPU experiment")
        write_json(path, contract)
    start = time.monotonic()
    total = Counter()
    for first in range(0, args.count, args.shard_size):
        last = min(first + args.shard_size, args.count)
        shard = output / f"{first:08d}-{last:08d}.jsonl.gz"
        if shard.exists():
            continue
        layouts = defaultdict(list)
        for rank, repetition, item in ranks(items, manifest, first, last):
            layouts[item.layout].append(partial(expert, graph=graphs[item.graph_id], item=item,
                                                repetition=repetition, manifest=manifest, rank=rank,
                                                identity=identity))
        with atomic_gzip(shard) as stream:
            for layout, jobs in layouts.items():
                for offset in range(0, len(jobs), args.worlds):
                    cohort = jobs[offset:offset + args.worlds]
                    batch = WarpBatch(layout, len(cohort), "cuda:0")
                    for row in batch.run(cohort):
                        stream.write(blob(row) + b"\n")
                        total["episodes"] += 1
                        total["successes"] += row["success"]
        print(json.dumps(dict(total) | {"durable_rank": last, "seconds": time.monotonic() - start}),
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--out", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--out", required=True)
    collect.add_argument("--gate", required=True)
    collect.add_argument("--split", choices=("train", "dev"), default="train")
    collect.add_argument("--count", type=int, required=True)
    collect.add_argument("--worlds", type=int, default=32)
    collect.add_argument("--shard-size", type=int, default=768)
    args = parser.parse_args()
    identity = backend()
    if args.command == "check":
        result = check_backend(args.out, identity)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit("GPU physics check failed")
    else:
        collect_gpu(args, identity)


if __name__ == "__main__":
    main()
