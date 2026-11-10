"""Train a shared per-action selector from physical expert decisions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .collect import read_rows
from .protocol import actions, read_manifest, start_seed, write_json
from .rollout import LOCATION_IDS, LOCATION_KEYS, controller_id, protocol_id
from .scene import CARRIERS, layout
from .workcell import PRIMITIVES


KINDS = (*[kind for kind in PRIMITIVES if kind not in ("release", "reject")], "sort")
FEATURES = len(KINDS) + len(LOCATION_KEYS) + 4 * CARRIERS + 20
FEATURE_VERSION = 1
LOCATION_ONEHOT = np.eye(len(LOCATION_KEYS), dtype=np.float32)


def bool_matrix(bits, size):
    return np.unpackbits(np.frombuffer(bits.to_bytes((size + 7) // 8, "little"), np.uint8),
                         bitorder="little")[:size].astype(bool)


class Encoder:
    def __init__(self, nodes, item):
        self.jobs = np.array([node.job for node in nodes])
        self.size = len(nodes)
        self.edges = np.zeros((self.size, self.size), dtype=np.float32)
        self.fixed = np.zeros((self.size, len(KINDS) + 3 * CARRIERS + 5), dtype=np.float32)
        configuration = layout(item.layout)
        for i, node in enumerate(nodes):
            self.edges[i, list(node.predecessors)] = 1
            self.fixed[i, KINDS.index(node.kind)] = 1
            self.fixed[i, len(KINDS) + node.job] = 1
            self.fixed[i, len(KINDS) + CARRIERS + item.binding[node.job]] = 1
            self.fixed[i, len(KINDS) + 2 * CARRIERS + item.destinations[node.job]] = 1
            xy = configuration.slots[f"feed{node.job}"][:2]
            target = configuration.slots[f"release{item.destinations[node.job]}"][:2]
            self.fixed[i, -5:] = [node.visit / 4, *xy, *target]
        self.degrees = self.edges.sum(axis=1)

    def encode(self, obs, done):
        locations = np.asarray(obs["locations"])
        occupied = [(locations == LOCATION_IDS[station]).any() for station in ("reader", "process")]
        finished = bool_matrix(done, self.size)
        own_location = LOCATION_ONEHOT[locations[self.jobs]]
        reads = np.asarray(obs["reads"])[self.jobs]
        cycles = np.asarray(obs["cycles"])[self.jobs] / 4
        global_features = [*obs["mechanism"], *obs["tcp"], *occupied]
        dynamic = np.column_stack((
            own_location, (reads < 0), (reads == 0), (reads == 1), cycles,
            cycles - self.fixed[:, -5], self.degrees / 5,
            (self.degrees - self.edges @ finished.astype(np.float32)) / 5,
            np.broadcast_to(global_features, (self.size, len(global_features))),
        ))
        joint_locations = np.stack([locations[self.jobs] == LOCATION_IDS[f"feed{i}"]
                                    for i in range(CARRIERS)], axis=1)
        value = np.concatenate((self.fixed, dynamic, joint_locations), axis=1, dtype=np.float32)
        if value.shape[1] != FEATURES:
            raise RuntimeError(f"feature schema changed ({value.shape[1]})")
        return value, ~finished


class GraphSelector(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(FEATURES, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1),
        )

    def forward(self, features):
        return self.score(features).squeeze(-1)


def checkpoint_sha(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_selector(path, manifest):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    meta = checkpoint["meta"]
    expected = (manifest, controller_id(), protocol_id(), FEATURE_VERSION, checkpoint_sha(__file__))
    actual = tuple(meta[key] for key in ("manifest", "controller", "protocol", "feature_version",
                                         "selector_code"))
    if actual != expected:
        raise ValueError("checkpoint does not match the frozen rollout protocol")
    model = GraphSelector(meta["width"])
    model.load_state_dict(checkpoint["state"])
    model.eval()
    cache = {}

    @torch.inference_mode()
    def selector(nodes, item, obs, done):
        if item.task_id not in cache:
            cache.clear()
            cache[item.task_id] = Encoder(nodes, item)
        features, unfinished = cache[item.task_id].encode(obs, done)
        logits = model(torch.from_numpy(features))
        return int(logits.masked_fill(~torch.from_numpy(unfinished), -torch.inf).argmax())

    return selector, meta


class Corpus:
    def __init__(self, directory, manifest_path, split, limit):
        if limit < 1:
            raise ValueError("need at least one episode")
        self.graphs, instances, self.manifest = read_manifest(manifest_path, split)
        self.items = {item.task_id: item for item in instances}
        self.split = split
        self.limit = limit
        self.files = []
        end = 0
        for path in sorted(Path(directory).glob("*.jsonl.gz")):
            left, right = (int(v) for v in path.name.split(".", 1)[0].split("-"))
            if end != left:
                raise ValueError(f"noncontiguous episode ranks in {directory}")
            self.files.append((path, left, right))
            end = right
            if end >= limit:
                break
        if end < limit:
            raise ValueError(f"only {end} of {limit} episode ranks available in {directory}")
        self.file_hashes = {path.name: checkpoint_sha(path) for path, _, _ in self.files}
        self.audit = self.validate()

    def episodes(self, rng=None):
        files = list(self.files)
        if rng is not None:
            rng.shuffle(files)
        for path, left, right in files:
            rows = list(read_rows(path))
            if {row["rank"] for row in rows} != set(range(left, right)) or len(rows) != right - left:
                raise ValueError(f"rank mismatch in {path}")
            if rng is not None:
                rng.shuffle(rows)
            for row in rows:
                if row["rank"] < self.limit:
                    yield row

    def validate(self):
        item_ids, graph_ids, families, seeds = set(), set(), set(), set()
        decision_count = success = 0
        primitive_counts = Counter()
        for row in self.episodes():
            item = self.items[row["task_id"]]
            if (row["manifest"], row["controller"], row["protocol"], row["split"], row["method"]) != (
                    self.manifest, controller_id(), protocol_id(), self.split, "expert"):
                raise ValueError("mixed manifest, controller, split or method in expert corpus")
            if row["graph"] != item.graph_id or row["seed"] != start_seed(
                    self.manifest, item, row["reset"]) or row["seed"] in seeds:
                raise ValueError("expert corpus provenance or seed overlap")
            seeds.add(row["seed"])
            item_ids.add(item.task_id)
            graph_ids.add(item.graph_id)
            families.add(item.family)
            decision_count += len(row["decisions"])
            success += row["success"]
            primitive_counts.update(kind for _, kind, ok, _ in row["attempts"] if ok)
        return {"episodes": len(seeds), "successes": success, "decisions": decision_count,
                "instances": len(item_ids), "graphs": len(graph_ids), "families": len(families),
                "executed_primitives": dict(primitive_counts)}

    def batches(self, size, rng=None):
        pending = []
        for row in self.episodes(rng):
            item = self.items[row["task_id"]]
            encoder = Encoder(actions(self.graphs[item.graph_id], item.recovery), item)
            examples = list(row["decisions"])
            if rng is not None:
                rng.shuffle(examples)
            for obs, done, ready, selected in examples:
                positives = bool_matrix(int(ready, 16), encoder.size)
                if not positives.any():
                    continue
                value, unfinished = encoder.encode(obs, int(done, 16))
                pending.append((value, unfinished, positives, selected))
                if len(pending) == size:
                    yield pack(pending)
                    pending.clear()
        if pending:
            yield pack(pending)


def pack(examples):
    width = max(len(x) for x, *_ in examples)
    batch = np.zeros((len(examples), width, FEATURES), np.float32)
    mask = np.zeros((len(examples), width), bool)
    positive = np.zeros_like(mask)
    selected = []
    for i, (value, unfinished, ready, choice) in enumerate(examples):
        n = len(value)
        batch[i, :n] = value
        mask[i, :n] = unfinished
        positive[i, :n] = ready
        selected.append(choice)
    return tuple(torch.from_numpy(a) for a in (batch, mask, positive, np.asarray(selected)))


def pass_epoch(model, batches, optimizer=None, device="cpu"):
    totals = Counter()
    model.train(optimizer is not None)
    for x, mask, positive, teacher in batches:
        flat = x[mask].to(device)
        mask, positive, teacher = (value.to(device) for value in (mask, positive, teacher))
        with torch.set_grad_enabled(optimizer is not None):
            logits = torch.zeros(mask.shape, device=device).masked_scatter(mask, model(flat))
            eligible = logits.masked_fill(~mask, -torch.inf)
            correct = logits.masked_fill(~positive, -torch.inf)
            rank_loss = (torch.logsumexp(eligible, dim=1) - torch.logsumexp(correct, dim=1)).mean()
            negatives = mask & ~positive
            binary = F.softplus(-logits[positive]).mean()
            if negatives.any():
                binary += F.softplus(logits[negatives]).mean()
            loss = rank_loss + .3 * binary
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        choice = eligible.argmax(dim=1)
        totals["eligible_top1"] += int(positive.gather(1, choice[:, None]).sum())
        totals["teacher_top1"] += int((teacher == choice).sum())
        totals["loss"] += loss.item() * len(x)
        totals["decisions"] += len(x)
    if not totals["decisions"]:
        raise ValueError("empty training pass")
    return {key: value / totals["decisions"] if key != "decisions" else value
            for key, value in totals.items()}


def train(args):
    if min(args.epochs, args.width, args.threads, args.batch) < 1:
        raise ValueError("invalid training hyperparameters")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    begin = time.monotonic()
    train_data = Corpus(args.train_data, args.manifest, "train", args.episodes)
    dev_data = Corpus(args.dev_data, args.manifest, "dev", args.dev_episodes)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = GraphSelector(args.width).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
    meta = {
        "manifest": train_data.manifest, "controller": controller_id(), "protocol": protocol_id(),
        "feature_version": FEATURE_VERSION, "selector_code": checkpoint_sha(__file__),
        "device": device, "width": args.width, "seed": args.seed,
        "parameters": parameters, "fit": train_data.audit, "dev": dev_data.audit,
        "fit_hashes": train_data.file_hashes, "dev_hashes": dev_data.file_hashes,
        "epoch_budget": args.epochs, "batch": args.batch,
    }
    best = None
    history = []
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        fit = pass_epoch(model, train_data.batches(args.batch, rng), optimizer, device)
        dev = pass_epoch(model, dev_data.batches(args.batch), device=device)
        record = {"epoch": epoch, "fit": fit, "dev": dev, "seconds": time.monotonic() - start}
        history.append(record)
        print(json.dumps(record), flush=True)
        score = (dev["eligible_top1"], -dev["loss"])
        if best is None or score > best:
            best = score
            temp = path.with_suffix(".tmp.pt")
            torch.save({"state": model.state_dict(), "meta": meta | {"selection": record}}, temp)
            temp.replace(path)
    result = meta | {"checkpoint": str(path), "sha256": checkpoint_sha(path),
                     "seconds": time.monotonic() - begin, "history": history}
    write_json(path.with_suffix(".json"), result)
    return result
