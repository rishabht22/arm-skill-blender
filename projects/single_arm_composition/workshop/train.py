"""Train a goal-conditioned behavior-cloning selector from workshop demos."""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import torch
from torch import nn


PROJECT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "recombination")
STOP = '["STOP",{}]'
OBSERVATION_KEYS = (
    "eef_position", "eef_quaternion_xyzw", "joint_position", "finger_position",
    "gripper_command", "entity_position", "entity_quaternion_xyzw", "grasped", "fixture_position",
)


def features(observation, goals, entities, fixture_joints):
    goal_features = np.zeros((len(entities), 10), dtype=np.float32)
    for index, obj in enumerate(entities):
        goal = goals.get(obj, {})
        if "xy" in goal:
            xyz = np.array([*goal["xy"], goal["z"]])
            goal_features[index, :8] = [1, 0, *xyz, *(xyz - observation["entity_position"][index])]
        elif "joint" in goal:
            target = goal["joint"]
            current = observation["fixture_position"][fixture_joints.index(obj)]
            goal_features[index, [1, 8, 9]] = [1, target, target - current]
    state = [np.ravel(observation[key]) for key in OBSERVATION_KEYS]
    return np.concatenate([*state, goal_features.ravel()]).astype(np.float32)


def call_label(skill, args, goals):
    """Represent goal-derived arguments by references so they can be recombined."""
    args = dict(args)
    goal = goals.get(args.get("obj") or args.get("held"), {})
    if (skill == "hover_over" and args.get("held") and "xy" in goal
            and args.get("xy") == goal["xy"] and args.get("height", .20) >= .20):
        args["xy"] = None
        args.pop("height", None)
    if "support" in goal and args.get("support") == goal["support"]:
        args["support"] = None
    if "joint" in goal and args.get("target") == goal["joint"]:
        args["target"] = None
    return json.dumps([skill, args], sort_keys=True, separators=(",", ":"))


def resolve_call(label, goals):
    """Resolve a label against the current goal. Return None for STOP."""
    skill, args = json.loads(label)
    if skill == "STOP":
        return None
    goal = goals.get(args.get("obj") or args.get("held"), {})
    if "xy" in args and args["xy"] is None:
        args["xy"] = goal["xy"]
        args["height"] = max(.20, goal["z"] - .8 + .11)
    if "support" in args and args["support"] is None:
        args["support"] = goal["support"]
    if "target" in args and args["target"] is None:
        args["target"] = goal["joint"]
    return skill, args


def read_demos(path, demos_per_task):
    rows = {split: defaultdict(list) for split in SPLITS}
    with h5py.File(path, "r") as dataset:
        if dataset.attrs.get("format") != "panda_workshop_v1":
            raise ValueError("Expected a panda_workshop_v1 HDF5 dataset")
        metadata = {
            scene: (tuple(group.attrs["entities"]), tuple(group.attrs["fixture_joints"]))
            for scene, group in dataset["scenes"].items()
        }
        for split in SPLITS:
            for task in dataset[split].values():
                included = 0
                for episode in task.values():
                    if not episode.attrs["success"] or not len(episode["calls"]):
                        continue
                    scene = episode.attrs["scene"]
                    goals = {goal["object"]: goal for goal in json.loads(episode.attrs["goal"])}
                    for call in episode["calls"].values():
                        observed = {key: call["observations"][key][0] for key in OBSERVATION_KEYS}
                        label = call_label(call.attrs["skill"], json.loads(call.attrs["args"]), goals)
                        rows[split][scene].append((features(observed, goals, *metadata[scene]), label))
                    observed = {key: call["observations"][key][-1] for key in OBSERVATION_KEYS}
                    rows[split][scene].append((features(observed, goals, *metadata[scene]), STOP))
                    included += 1
                    if split == "train" and included == demos_per_task:
                        break
    if not rows["train"]:
        raise ValueError("No successful full-task training demos")
    return rows, metadata


def as_batch(rows, classes):
    observations, labels = zip(*rows)
    return torch.from_numpy(np.stack(observations)), torch.tensor([classes.get(label, -1) for label in labels])


class Selector(nn.Module):
    def __init__(self, width, labels, entities, fixture_joints):
        super().__init__()
        self.labels = labels
        self.entities = entities
        self.fixture_joints = fixture_joints
        self.register_buffer("center", torch.zeros(width))
        self.register_buffer("scale", torch.ones(width))
        self.network = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, len(labels)))

    def forward(self, values):
        return self.network((values - self.center) / self.scale)

    @torch.no_grad()
    def choose(self, observation, goal_conditions):
        """Return a (skill, args) pair for Step, or None when the selector predicts STOP."""
        goals = {goal["object"]: goal for goal in goal_conditions}
        values = torch.from_numpy(features(observation, goals, self.entities, self.fixture_joints))
        for index in self(values).argsort(descending=True).tolist():
            try:
                return resolve_call(self.labels[index], goals)
            except KeyError:
                # A goal argument for this action is absent from the current task.
                continue
        raise ValueError("The selector has no executable label")


def load_selector(path, scene):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "panda_workshop_selector_v1":
        raise ValueError("Expected a panda_workshop_selector_v1 checkpoint")
    spec = checkpoint["scenes"][scene]
    selector = Selector(spec["width"], spec["labels"], spec["entities"], spec["fixture_joints"])
    selector.load_state_dict(spec["weights"])
    return selector.eval()


@torch.no_grad()
def accuracy(model, batch):
    values, labels = batch
    return float((model(values).argmax(1) == labels).float().mean())


def fit(batches, labels, metadata, epochs, batch_size, seed):
    train_x, train_y = batches["train"]
    selector = Selector(train_x.shape[1], labels, *metadata)
    selector.center.copy_(train_x.mean(dim=0))
    selector.scale.copy_(train_x.std(dim=0, unbiased=False).clamp_min(.01))
    optimizer = torch.optim.AdamW(selector.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(seed)
    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        for indices in torch.randperm(len(train_y), generator=rng).split(batch_size):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(selector(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
        val = accuracy(selector, batches["validation"])
        if val > best_accuracy:
            best_accuracy, best_epoch = val, epoch
            best_weights = deepcopy(selector.state_dict())
    selector.load_state_dict(best_weights)
    return selector, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data/raw/panda_workshop.h5")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs/workshop_selector.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demos-per-task", type=int, help="Limit successful training demos per task")
    parser.add_argument("--dry-run", action="store_true", help="Validate data and label coverage without training")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.seed < 0:
        parser.error("Epochs and batch size must be positive and seed must be non-negative")
    if args.demos_per_task is not None and args.demos_per_task < 1:
        parser.error("--demos-per-task must be positive")
    if not args.dry_run and args.out.exists():
        parser.error(f"Checkpoint already exists: {args.out}. Choose another --out path.")

    try:
        rows, metadata = read_demos(args.dataset, args.demos_per_task)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    scenes = {}
    for scene, training_rows in rows["train"].items():
        labels = sorted({label for _, label in training_rows})
        classes = {label: index for index, label in enumerate(labels)}
        batches = {split: as_batch(rows[split][scene], classes) for split in SPLITS if rows[split][scene]}
        if "validation" not in batches:
            parser.error(f"No validation decisions for {scene}")
        print(f"{scene} ({len(labels)} call labels)", flush=True)
        for split, (_, targets) in batches.items():
            covered = int((targets >= 0).sum())
            print(f"  {split}: {len(targets)} decisions, {covered} with a training label", flush=True)
        if args.dry_run:
            continue

        model, best_epoch = fit(batches, labels, metadata[scene], args.epochs, args.batch_size, args.seed)
        metrics = {split: accuracy(model, batch) for split, batch in batches.items()}
        for split, score in metrics.items():
            print(f"  {split} next-call accuracy: {score:.1%}", flush=True)
        entities, fixture_joints = metadata[scene]
        scenes[scene] = dict(width=len(model.center), labels=labels, entities=entities,
                             fixture_joints=fixture_joints, weights=model.state_dict(),
                             best_epoch=best_epoch, accuracy=metrics)
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.out.parent) as temporary:
            checkpoint = Path(temporary) / args.out.name
            torch.save({"format": "panda_workshop_selector_v1", "scenes": scenes}, checkpoint)
            args.out.hardlink_to(checkpoint)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
