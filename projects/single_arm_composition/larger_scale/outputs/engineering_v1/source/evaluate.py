"""Locked paired evaluation and family/graph/seed bootstrap estimates."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .collect import atomic_gzip, get_cell, ranks, read_rows
from .learn import checkpoint_sha, load_selector
from .protocol import SPLITS, blob, digest, read_manifest, write_json
from .rollout import Retrieval, controller_id, protocol_id, run


TEST = SPLITS[2:]
BASELINES = ("retrieval", "symbolic", "expert")
EVAL = {}


def model_specs(paths, manifest):
    import torch

    torch.set_num_threads(1)
    result = []
    seen = set()
    for path in paths:
        _, meta = load_selector(path, manifest)
        model = {"sha256": checkpoint_sha(path), "width": meta["width"],
                 "seed": meta["seed"], "parameters": meta["parameters"],
                 "episodes": meta["fit"]["episodes"], "decisions": meta["fit"]["decisions"],
                 "selected_epoch": meta["selection"]["epoch"]}
        key = (model["width"], model["seed"], model["episodes"])
        if key in seen:
            raise ValueError(f"duplicate model configuration and seed: {key}")
        seen.add(key)
        result.append(model)
    return result


def lock(manifest_path, paths, counts, output):
    _, _, manifest = read_manifest(manifest_path, "dev")
    if len(counts) != len(TEST) or min(counts) < 1:
        raise ValueError("provide one positive count per test slice")
    path = Path(output)
    if path.exists():
        raise ValueError("evaluation lock already exists")
    contract = {"manifest": manifest, "controller": controller_id(), "protocol": protocol_id(),
                "counts": dict(zip(TEST, counts)), "baselines": list(BASELINES),
                "models": model_specs(paths, manifest)}
    write_json(path, {"contract": contract, "sha256": digest(contract)})
    return contract


def read_lock(path, manifest):
    if path is None:
        raise ValueError("sealed test requires a frozen evaluation lock")
    lock_file = json.loads(Path(path).read_bytes())
    contract = lock_file["contract"]
    if digest(contract) != lock_file["sha256"] or (
            contract["manifest"], contract["controller"], contract["protocol"]) != (
                manifest, controller_id(), protocol_id()):
        raise ValueError("evaluation lock integrity or provenance failure")
    return lock_file


def evaluate_init(manifest_path, split, paths, specs, output, lock_hash):
    import torch

    torch.set_num_threads(1)
    graphs, instances, manifest = read_manifest(manifest_path, split)
    train, _, _ = read_manifest(manifest_path, "train")
    selectors = [(spec, load_selector(path, manifest)[0]) for path, spec in zip(paths, specs)]
    EVAL.update(graphs=graphs, items=instances, manifest=manifest, selectors=selectors,
                retrieval=Retrieval(train), output=Path(output), lock=lock_hash)


def evaluate_batch(bounds):
    first, last = bounds
    path = EVAL["output"] / f"{first:08d}-{last:08d}.jsonl.gz"
    if path.exists():
        return Counter(row["method"] for row in read_rows(path))
    items = sorted(ranks(EVAL["items"], EVAL["manifest"], first, last), key=lambda pair: pair[2].layout)
    counts = Counter()
    methods = [(name, None, None) for name in BASELINES] + [
        ("learned", spec, selector) for spec, selector in EVAL["selectors"]]
    with atomic_gzip(path) as stream:
        for rank, repetition, item in items:
            cell = get_cell(item.layout)
            for method, spec, selector in methods:
                result, _ = run(EVAL["graphs"][item.graph_id], item, repetition, EVAL["manifest"],
                                method, cell=cell, retrieval=EVAL["retrieval"], selector=selector,
                                rank=rank)
                result["checkpoint"] = spec
                result["lock"] = EVAL["lock"]
                stream.write(blob(result) + b"\n")
                counts[method] += 1
    return counts


def evaluate(args):
    if not 0 <= args.part < args.parts or min(args.count, args.workers) < 1:
        raise ValueError("invalid evaluation bounds")
    # Read the development seal before any sealed examples.
    _, _, manifest = read_manifest(args.manifest, "dev")
    specs = model_specs(args.model, manifest)
    lock_hash = None
    if args.split in TEST:
        frozen = read_lock(args.lock, manifest)
        contract = frozen["contract"]
        if contract["counts"][args.split] != args.count or contract["models"] != specs:
            raise ValueError("count or checkpoint differs from the locked protocol")
        lock_hash = frozen["sha256"]
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    contract = {"manifest": manifest, "controller": controller_id(), "protocol": protocol_id(),
                "split": args.split, "count": args.count, "models": specs, "lock": lock_hash}
    for other in directory.glob("part-*-of-*.json"):
        if json.loads(other.read_bytes()) != contract:
            raise ValueError("evaluation output has different provenance")
    write_json(directory / f"part-{args.part:05d}-of-{args.parts:05d}.json", contract)
    batch = 8
    jobs = [(first, min(first + batch, args.count)) for i, first in enumerate(range(0, args.count, batch))
            if i % args.parts == args.part]
    totals = Counter()
    begin = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=evaluate_init, initargs=(
            args.manifest, args.split, args.model, specs, directory, lock_hash)) as pool:
        for counts in pool.map(evaluate_batch, jobs):
            totals.update(counts)
            print(json.dumps({"attempts": dict(totals), "seconds": round(time.monotonic() - begin, 2)}),
                  flush=True)
    return {"attempts": dict(totals), "seconds": time.monotonic() - begin}


def bootstrap(values, groups, rng, repetitions=1500):
    # Values have one row per paired start and one column per training seed.
    # Each draw samples graph families, then graphs within families, and model
    # seeds globally. All paired comparisons keep the same physical starts.
    seeds = rng.integers(values.shape[1], size=(repetitions, values.shape[1]))
    estimates = []
    for graphs in groups.values():
        means = np.stack([values[indices].mean(axis=0) for indices in graphs.values()])
        chosen = rng.integers(len(means), size=(repetitions, len(means)))
        graph_draws = means[chosen].mean(axis=1)
        estimates.append(np.take_along_axis(graph_draws, seeds, axis=1).mean(axis=1))
    by_family = np.stack(estimates, axis=1)
    chosen_families = rng.integers(len(groups), size=by_family.shape)
    draws = np.take_along_axis(by_family, chosen_families, axis=1).mean(axis=1)
    return [float(v) for v in np.quantile(draws, [.025, .975])]


def summarize(paired, spec_rows):
    model_keys = {model["sha256"]: f"learned-w{model['width']}-n{model['episodes']}"
                  for model in spec_rows}
    expected = set(BASELINES) | set(model_keys)
    split_name = next(iter(next(iter(paired.values())).values()))["split"]
    groups = defaultdict(lambda: defaultdict(list))
    all_seeds = set()
    for index, (start, methods) in enumerate(paired.items()):
        if set(methods) != expected:
            raise ValueError(f"missing matched method for {start}")
        fingerprints = {(r["start"], r["seed"], r["manifest"], r["controller"], r["protocol"], r["lock"],
                         r["graph"], r["family"], r["split"]) for r in methods.values()}
        if len(fingerprints) != 1:
            raise ValueError(f"matched starting states or provenance differ for {start}")
        ref = methods["expert"]
        if ref["seed"] in all_seeds:
            raise ValueError("overlapping physical resets")
        all_seeds.add(ref["seed"])
        groups[ref["family"]][ref["graph"]].append(index)
    rng = np.random.default_rng(9491921)
    result = {}
    for name in sorted(set(BASELINES) | set(model_keys.values())):
        keys = sorted(key for key in expected if model_keys.get(key, key) == name)
        block = [[methods[key] for key in keys] for methods in paired.values()]
        flattened = list(itertools.chain.from_iterable(block))
        success = np.asarray([[r["success"] for r in row] for row in block], dtype=float)
        progress = np.asarray([[r["completed"] / r["subgoals"] for r in row] for row in block])
        skill_attempts, skill_successes = Counter(), Counter()
        for row in flattened:
            for _, kind, ok, reason in row["attempts"]:
                if kind is not None and reason != "dependency":
                    skill_attempts[kind] += 1
                    skill_successes[kind] += ok
        stats = {"rollouts": len(flattened), "successes": int(success.sum()),
                 "micro_success": float(success.mean()),
                 "family_balanced_success": float(np.mean([
                     np.mean([success[indices].mean() for indices in graphs.values()])
                     for graphs in groups.values()])),
                 "family_seed_ci95": bootstrap(success, groups, rng),
                 "mean_subgoal_progress": float(progress.mean()),
                 "dependency_violations": sum(r["violations"] for r in flattened),
                 "skill_failures": sum(r["skill_failures"] for r in flattened),
                 "recovery_completed": sum(r["recovered"] for r in flattened),
                 "recovery_goals": sum(r["recovery_goals"] for r in flattened),
                 "failure_reasons": dict(Counter(r["reason"] for r in flattened if not r["success"])),
                 "primitive_attempts": dict(skill_attempts),
                 "primitive_successes": dict(skill_successes),
                 "seed_successes": {key: int(success[:, k].sum()) for k, key in enumerate(keys)}}
        if name not in BASELINES:
            for baseline in BASELINES:
                baseline_values = np.asarray([[methods[baseline]["success"]]
                                              for methods in paired.values()], dtype=float)
                delta = success - baseline_values
                stats[f"paired_delta_vs_{baseline}"] = {
                    "micro": float(delta.mean()), "family_seed_ci95": bootstrap(delta, groups, rng)}
        result[name] = stats
    return {"split": split_name, "matched_starts": len(paired), "families": len(groups),
            "graphs": sum(map(len, groups.values())), "methods": result}


def report(directories, *, partial=False):
    result = {}
    expected_root = None
    for directory in map(Path, directories):
        files = sorted(directory.glob("part-*-of-*.json"))
        if not files:
            raise ValueError(f"no evaluation contract in {directory}")
        contract = json.loads(files[0].read_bytes())
        if any(json.loads(path.read_bytes()) != contract for path in files[1:]):
            raise ValueError("mixed evaluation contracts")
        root = (contract["manifest"], contract["controller"], contract["protocol"], contract["lock"])
        if expected_root is not None and root != expected_root:
            raise ValueError("report inputs use different experimental locks or controller versions")
        expected_root = root
        split = contract["split"]
        if split in result:
            raise ValueError(f"duplicate evaluation split: {split}")
        paired = {}
        rank_to_start = {}
        for path in sorted(directory.glob("*.jsonl.gz")):
            for row in read_rows(path):
                if (row["manifest"], row["controller"], row["protocol"], row["lock"], row["split"]) != (*root, split):
                    raise ValueError(f"rollout provenance mismatch: {path}")
                start = (row["task_id"], row["reset"])
                if rank_to_start.setdefault(row["rank"], start) != start:
                    raise ValueError("rank has multiple matched starts")
                methods = paired.setdefault(start, {})
                model = row["checkpoint"]
                if model is not None and model not in contract["models"]:
                    raise ValueError("checkpoint provenance differs from evaluation contract")
                key = model["sha256"] if model else row["method"]
                if key in methods:
                    raise ValueError("duplicate rollout")
                methods[key] = row
        declared = set(range(contract["count"]))
        if not paired or set(rank_to_start) - declared or len(paired) != len(rank_to_start):
            raise ValueError("missing data or invalid evaluation ranks")
        complete = set(rank_to_start) == declared
        if not complete and not partial:
            raise ValueError("missing evaluation ranks")
        result[split] = summarize(paired, contract["models"]) | {
            "declared_starts": contract["count"], "complete": complete,
            "checkpoints": contract["models"],
        }
    return {"manifest": expected_root[0], "controller": expected_root[1], "protocol": expected_root[2],
            "lock": expected_root[3], "splits": result}
