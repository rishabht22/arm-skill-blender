"""Fit outcome-conditioned selectors from planner demonstrations."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from larger_scale.protocol import write_json
from larger_scale.rollout import controller_id

from .model import Selector, encode, network
from .planner import ACTIONS, optimal, reached, transition
from .sim import rollout, symbolic
from .tasks import tasks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_id():
    directory = Path(__file__).parent
    return {name: sha(directory / name) for name in ("model.py", "planner.py", "sim.py", "tasks.py")}


def examples(suite, count):
    result = {}
    starts = set()
    for task in tasks(suite, count):
        key = task.goal, task.start
        if key in starts:
            continue
        starts.add(key)
        for state, choices in optimal(*key).items():
            result[task.goal, state] = choices
    return result, len(starts)


def tensors(rows, device):
    features = []
    targets = np.zeros((len(rows), len(ACTIONS)), dtype=bool)
    for i, ((goal, state), choices) in enumerate(rows.items()):
        features.append(encode(goal, state))
        targets[i, list(choices)] = True
    return torch.from_numpy(np.stack(features)).to(device), torch.from_numpy(targets).to(device)


def loss_and_accuracy(model, features, valid):
    scores = model(features).squeeze(-1)
    best = scores.masked_fill(~valid, -torch.inf)
    ranking = (scores.logsumexp(dim=1) - best.logsumexp(dim=1)).mean()
    binary = F.softplus(-scores[valid]).mean() + F.softplus(scores[~valid]).mean()
    accuracy = valid.gather(1, scores.argmax(dim=1, keepdim=True)).float().mean()
    return ranking + .2 * binary, accuracy


def abstract_success(choose, task):
    state = task.start
    for _ in range(64):
        if reached(task.goal, state):
            return True
        successor = transition(task.goal, state, ACTIONS[choose(task.goal, state)])
        if successor is None:
            return False
        state = successor
    return reached(task.goal, state)


def train(args):
    if min(args.problems, args.dev_problems, args.epochs, args.batch, args.width, args.threads) < 1:
        raise ValueError("training sizes and thread count must be positive")
    if args.physical < 0:
        raise ValueError("physical demonstration count must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    begin = time.monotonic()
    rows, unique_starts = examples("train", args.problems)
    abstract_count = len(rows)
    measured, physical_successes = 0, 0
    for task in tasks("train", args.physical):
        log = rollout(task, symbolic, label=True)
        physical_successes += log["success"]
        for state, choices in log["decisions"]:
            if choices:
                rows[task.goal, state] = choices
                measured += 1
    dev, _ = examples("dev", args.dev_problems)
    overlap = rows.keys() & dev.keys()
    if overlap:
        raise ValueError("training and development share goal-state examples")
    x, y = tensors(rows, device)
    val_x, val_y = tensors(dev, device)
    model = network(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=1e-4)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "controller": controller_id(), "sources": source_id(), "width": args.width,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seed": args.seed, "device": device,
        "requested_symbolic_problems": args.problems, "unique_symbolic_goal_starts": unique_starts,
        "symbolic_states": abstract_count, "fit_states": len(rows), "development_states": len(dev),
        "physical_attempts": args.physical, "physical_successes": physical_successes,
        "physical_decisions": measured, "history": [],
    }
    dev_tasks = list(tasks("dev", min(args.dev_problems, 48)))
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.
        for indices in torch.randperm(len(x), device=device).split(args.batch):
            loss, _ = loss_and_accuracy(model, x[indices], y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(indices)
        model.eval()
        with torch.inference_mode():
            dev_metrics = [loss_and_accuracy(model, bx, by)
                           for bx, by in zip(val_x.split(args.batch), val_y.split(args.batch))]
            dev_loss = float(torch.stack([pair[0] for pair in dev_metrics]).mean())
            dev_accuracy = float(torch.stack([pair[1] for pair in dev_metrics]).mean())
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        predictor = network(args.width)
        predictor.load_state_dict(state)
        choose = Selector(predictor)
        abstract = sum(abstract_success(choose, task) for task in dev_tasks)
        entry = {"epoch": epoch, "fit_loss": total / len(x), "dev_loss": dev_loss,
                 "dev_top1": dev_accuracy, "dev_abstract_completed": abstract,
                 "dev_abstract_starts": len(dev_tasks)}
        report["history"].append(entry)
        print(entry, flush=True)
        score = (abstract, -dev_loss)
        if best is None or score > best:
            best = score
            report["selected_epoch"] = epoch
            temp = path.with_suffix(".tmp.pt")
            torch.save({"state": state, "meta": {key: report[key] for key in (
                "controller", "sources", "width", "seed", "selected_epoch")}}, temp)
            temp.replace(path)
    report.update(seconds=time.monotonic() - begin, checkpoint_sha256=sha(path))
    write_json(path.with_suffix(".json"), report)


def load(path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    meta = checkpoint["meta"]
    if meta["controller"] != controller_id() or meta["sources"] != source_id():
        raise ValueError("checkpoint differs from the goal-policy or frozen controller source")
    model = network(meta["width"])
    model.load_state_dict(checkpoint["state"])
    return Selector(model)
