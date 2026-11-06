import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import PROCESSED, RUN, SKILLS, sha256, write_json
from .model import CompositionHead, load_head


def metrics(predictions, labels):
    confusion = np.zeros((2, 4, 4), dtype=int)
    for a in range(2):
        np.add.at(confusion[a], (labels[:, a], predictions[:, a]), 1)
    recalls = []
    for c in confusion:
        recalls.extend(np.diag(c)[c.sum(1) > 0] / c.sum(1)[c.sum(1) > 0])
    return {"per_arm_accuracy": (predictions == labels).mean(0).tolist(),
            "joint_skill_accuracy": float(np.all(predictions == labels, axis=1).mean()),
            "macro_recall": float(np.mean(recalls)),
            "confusion_true_rows_predicted_columns": confusion.tolist()}


def train(dataset=PROCESSED, run=RUN, epochs=100, seed=7):
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    dataset, run = Path(dataset), Path(run)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = np.load(dataset, allow_pickle=False)
    metadata = json.loads(dataset.with_suffix(".json").read_text())
    masks = {name: np.isin(data["episode"], ids) for name, ids in metadata["splits"].items()}
    x = torch.from_numpy(data["features"].astype(np.float32))
    y = torch.from_numpy(data["labels"].astype(np.int64))
    model = CompositionHead()
    tx, ty = x[masks["train"]], y[masks["train"]]
    model.mean.copy_(tx.mean(0))
    model.scale.copy_(tx.std(0).clamp_min(0.01))
    # Only composition-head parameters are optimized. There is no arm policy optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = torch.bincount(ty.flatten(), minlength=4).float().clamp_min(1)
    loss_fn = nn.CrossEntropyLoss(weight=(counts.sum() / counts).sqrt())
    run.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    dataset_digest = sha256(dataset)
    skills_digest = sha256(dataset.with_suffix(".skills.json"))
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for batch in torch.randperm(len(tx), generator=generator).split(256):
            optimizer.zero_grad()
            loss = loss_fn(model(tx[batch]).reshape(-1, 4), ty[batch].flatten())
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            prediction = model(x[masks["validation"]]).argmax(-1).numpy()
        validation = metrics(prediction, y[masks["validation"]].numpy())
        score = validation["joint_skill_accuracy"]
        history.append({"epoch": epoch, "loss": loss_total / len(tx), "validation": validation})
        if score > best:
            best = score
            torch.save({"architecture": {"dimension": 48, "hidden": 128},
                        "state_dict": model.state_dict(), "seed": seed, "best_epoch": epoch,
                        "dataset_sha256": dataset_digest,
                        "skills_sha256": skills_digest,
                        "skills": list(SKILLS), "splits": metadata["splits"]}, run / "head.pt")
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d} loss={loss_total / len(tx):.4f} val joint={score:.3f}", flush=True)
    write_json(run / "training.json", {"history": history, "best_validation_joint_accuracy": best})
    # Seal the frozen skill library next to the selected head.
    (run / "skills.json").write_bytes(dataset.with_suffix(".skills.json").read_bytes())
    return evaluate(dataset, run)


def evaluate(dataset=PROCESSED, run=RUN):
    dataset, run = Path(dataset), Path(run)
    head, checkpoint = load_head(run / "head.pt")
    if sha256(dataset) != checkpoint["dataset_sha256"]:
        raise ValueError("Checkpoint and dataset do not match")
    if sha256(run / "skills.json") != checkpoint["skills_sha256"]:
        raise ValueError("Frozen skill library changed since training")
    data = np.load(dataset, allow_pickle=False)
    mask = np.isin(data["episode"], checkpoint["splits"]["test"])
    x, y = data["features"][mask], data["labels"][mask]
    with torch.no_grad():
        predictions = head(torch.from_numpy(x)).argmax(-1).numpy()
    train_mask = np.isin(data["episode"], checkpoint["splits"]["train"])
    majority = np.array([np.bincount(data["labels"][train_mask, a], minlength=4).argmax() for a in range(2)])
    # Strong task-specific baseline: always advance the fixed known sequence on
    # completion. It exposes how little branching this first task actually has.
    active = x[:, 36:46].reshape(-1, 2, 5).argmax(-1) - 1
    terminated = x[:, 46:] > .5
    sequence = np.where(terminated, np.minimum(active + 1, 3), active)
    report = {"evaluation": "offline imitation at expert-annotated skill completion events",
              "physical_task_success_evaluated": False,
              "source": "published RoboFactory LiftBarrier", "skills": SKILLS,
              "test_episodes": checkpoint["splits"]["test"], "test_decisions": len(x),
              "best_epoch": checkpoint["best_epoch"], "head": metrics(predictions, y),
              "majority_baseline": metrics(np.broadcast_to(majority, y.shape), y),
              "fixed_sequence_baseline": metrics(sequence, y),
              "limitation": "Fixed-order LiftBarrier is a sequencing sanity check, not compositional generalization."}
    write_json(run / "evaluation.json", report)
    np.savez_compressed(run / "test_predictions.npz", prediction=predictions, labels=y,
                        episode=data["episode"][mask], step=data["step"][mask])
    print(f"Held-out joint skill accuracy: {report['head']['joint_skill_accuracy']:.3f}; "
          f"majority: {report['majority_baseline']['joint_skill_accuracy']:.3f}; "
          f"fixed-sequence: {report['fixed_sequence_baseline']['joint_skill_accuracy']:.3f}")
    return report
