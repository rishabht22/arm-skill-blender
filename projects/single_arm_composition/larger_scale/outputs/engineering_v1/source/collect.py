"""Resumable expert episode collection. Workers own whole atomic shards."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
import gzip
import json
from pathlib import Path
import random
import time

from .protocol import blob, read_manifest, write_json
from .rollout import controller_id, protocol_id, run
from .workcell import Workcell


CELL_CACHE = OrderedDict()
WORK = {}


def get_cell(layout):
    if layout not in CELL_CACHE:
        CELL_CACHE[layout] = Workcell(layout)
        if len(CELL_CACHE) > 24:
            CELL_CACHE.popitem(last=False)
    CELL_CACHE.move_to_end(layout)
    return CELL_CACHE[layout]


@contextmanager
def atomic_gzip(path):
    path = Path(path)
    import os
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
                            delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            with gzip.GzipFile(fileobj=temp, mode="wb", compresslevel=2, mtime=0) as stream:
                yield stream
            temp.flush()
            os.fsync(temp.fileno())
            os.replace(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)


def read_rows(path):
    with gzip.open(path, "rb") as source:
        for line in source:
            yield json.loads(line)


@lru_cache(maxsize=4)
def reset_order(count, manifest, split, repetition):
    order = list(range(count))
    random.Random(f"{manifest}:{split}:{repetition}").shuffle(order)
    return order


def ranks(items, manifest, start, stop):
    count = len(items)
    for repetition in range(start // count, (stop - 1) // count + 1):
        order = reset_order(count, manifest, items[0].split, repetition)
        offset = repetition * count
        for rank in range(max(start, offset), min(stop, offset + count)):
            yield rank, repetition, items[order[rank - offset]]


def worker_init(manifest_path, split, output):
    graphs, items, manifest = read_manifest(manifest_path, split)
    WORK.update(graphs=graphs, items=items, manifest=manifest, output=Path(output))


def shard(job):
    start, stop = job
    directory = WORK["output"]
    path = directory / f"{start:08d}-{stop:08d}.jsonl.gz"
    sidecar = path.with_suffix(".stats.json")
    if path.exists() and sidecar.exists():
        return json.loads(sidecar.read_bytes())
    items, graphs, manifest = WORK["items"], WORK["graphs"], WORK["manifest"]
    rows = sorted(ranks(items, manifest, start, stop), key=lambda entry: entry[2].layout)
    successes = decisions = simulation = 0
    begin = time.monotonic()
    with atomic_gzip(path) as stream:
        for rank, repetition, item in rows:
            record, _ = run(graphs[item.graph_id], item, repetition, manifest, "expert",
                            cell=get_cell(item.layout), keep_decisions=True, rank=rank)
            successes += record["success"]
            decisions += len(record["decisions"])
            simulation += record["sim_seconds"]
            stream.write(blob(record) + b"\n")
    stats = {"start": start, "stop": stop, "episodes": stop - start,
             "successes": successes, "decisions": decisions, "simulation_seconds": simulation,
             "seconds": time.monotonic() - begin}
    write_json(sidecar, stats)
    return stats


def collect(manifest_path, output, split, count, workers, batch, part, parts):
    if split not in ("train", "dev"):
        raise ValueError("collect accepts fitting and development splits only")
    if not 0 <= part < parts or min(workers, count, batch) < 1:
        raise ValueError("invalid shard partition")
    _, items, manifest = read_manifest(manifest_path, split)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    contract = {"manifest": manifest, "controller": controller_id(), "protocol": protocol_id(),
                "split": split, "count": count, "batch": batch, "tasks": len(items)}
    # Each distributed partition writes its own contract file.
    contract_path = out / f"part-{part:05d}-of-{parts:05d}.json"
    other = [json.loads(path.read_bytes()) for path in out.glob("part-*-of-*.json")]
    if any(data != contract for data in other):
        raise ValueError("output already contains a different run contract")
    write_json(contract_path, contract)
    assignments = [(start, min(start + batch, count)) for i, start in enumerate(range(0, count, batch))
                   if i % parts == part]
    target = sum(b - a for a, b in assignments)
    start = time.monotonic()
    episodes = successes = decisions = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                             initargs=(str(manifest_path), split, str(out))) as pool:
        for stats in pool.map(shard, assignments):
            episodes += stats["episodes"]
            successes += stats["successes"]
            decisions += stats["decisions"]
            print(json.dumps({"episodes": episodes, "successes": successes, "decisions": decisions,
                              "elapsed_seconds": round(time.monotonic() - start, 2),
                              "target": target}), flush=True)
    return {"episodes": episodes, "successes": successes, "decisions": decisions,
            "seconds": time.monotonic() - start}



def corpus_status(directory):
    directory = Path(directory)
    contracts = [json.loads(path.read_bytes()) for path in directory.glob("part-*-of-*.json")]
    if not contracts or any(contract != contracts[0] for contract in contracts):
        raise ValueError("corpus run contract missing or mixed")
    contract = contracts[0]
    rows = []
    for path in sorted(directory.glob("*.jsonl.stats.json")):
        if path.with_name(path.name.removesuffix(".stats.json") + ".gz").is_file():
            rows.append(json.loads(path.read_bytes()))
    previous = prefix = 0
    for row in rows:
        if row["start"] < previous or row["episodes"] != row["stop"] - row["start"]:
            raise ValueError("invalid corpus shard ranks")
        if row["start"] == prefix:
            prefix = row["stop"]
        previous = row["stop"]
    return {"contract": contract, "shards": len(rows), "contiguous_episodes": prefix,
            "executed_episodes": sum(r["episodes"] for r in rows),
            "successful_episodes": sum(r["successes"] for r in rows),
            "decision_examples": sum(r["decisions"] for r in rows),
            "simulation_seconds": sum(r["simulation_seconds"] for r in rows)}
