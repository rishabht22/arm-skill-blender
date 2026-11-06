"""Task-specific weak skill annotations for published LiftBarrier trajectories."""
import json
from pathlib import Path

import h5py
import numpy as np

from .common import ARMS, PROCESSED, RAW, SKILLS, sha256, decision_features, write_json

ANNOTATION = {
    "version": 2,
    "kind": "weak labels inferred offline from expert commands; not provided ground truth",
    "approach": "Before the first gripper command below 0.999",
    "grasp": "From closing onset through the first command below -0.95 (inclusive)",
    "lift": "After full closure until the last joint-target displacement > 0.003 rad",
    "hold": "From the last significant joint-target change to trajectory end",
    "motion_threshold_rad": 0.003,
}


def annotate(actions):
    """Use commands for OFFLINE labels only; fail if the task assumption is false."""
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 8 or len(actions) < 4:
        raise ValueError("Expected a LiftBarrier (T, 8) pd_joint_pos action array")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite actions")
    closing = np.flatnonzero(actions[:, -1] < 0.999)
    closed = np.flatnonzero(actions[:, -1] < -0.95)
    if not len(closing) or not len(closed):
        raise ValueError("No complete grasp; manual skill annotations are required")
    start, end = int(closing[0]), int(closed[0]) + 1
    if np.any(np.diff(actions[start:end, -1]) > 1e-4):
        raise ValueError("Unexpected reopening during grasp")
    movement = np.linalg.norm(np.diff(actions[:, :7], axis=0), axis=1)
    late = np.flatnonzero((movement > 0.003) & (np.arange(len(movement)) >= end))
    if not len(late):
        raise ValueError("No lift motion detected after grasp")
    hold = int(late[-1]) + 1
    if not 0 < start < end < hold < len(actions):
        raise ValueError(f"Invalid skill boundaries: {start, end, hold}")
    labels = np.zeros(len(actions), dtype=np.int64)
    labels[start:end] = 1
    labels[end:hold] = 2
    labels[hold:] = 3
    return labels, (start, end, hold)


def split_episodes(ids, seed=7):
    """Split entire trajectories; adjacent frames can never cross partitions."""
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < 10 or len(np.unique(ids)) != len(ids):
        raise ValueError("Need at least 10 unique episodes")
    order = np.random.default_rng(seed).permutation(ids)
    n_train, n_val = int(0.7 * len(ids)), int(0.15 * len(ids))
    return {"train": sorted(order[:n_train].tolist()),
            "validation": sorted(order[n_train:n_train + n_val].tolist()),
            "test": sorted(order[n_train + n_val:].tolist())}


def prepare(raw=RAW, output=PROCESSED, seed=7):
    raw, output = Path(raw), Path(output)
    metadata = json.loads((raw / "LiftBarrier.json").read_text())
    modes = metadata["env_info"]["env_kwargs"]
    if modes["control_mode"] != "pd_joint_pos":
        raise ValueError("Skill annotation is only defined for pd_joint_pos")
    ids = [int(e["episode_id"]) for e in metadata["episodes"] if e.get("success") is True]
    splits = split_episodes(ids, seed)
    features, labels, episodes, steps, skills, audit = [], [], [], [], [], []
    total_frames = 0
    with h5py.File(raw / "LiftBarrier.h5", "r") as dataset:
        for eid in sorted(ids):
            group = dataset[f"traj_{eid}"]
            actions = [group[f"actions/{arm}"][:] for arm in ARMS]
            if len(actions[0]) != len(actions[1]):
                raise ValueError(f"Unsynchronized action arrays in episode {eid}")
            length = len(actions[0])
            arm_labels, boundaries, targets = [], [], []
            positions, velocities = [], []
            for arm, commands in zip(ARMS, actions):
                y, bounds = annotate(commands)
                q = group[f"obs/agent/{arm}/qpos"][:]
                v = group[f"obs/agent/{arm}/qvel"][:]
                if q.shape != (length + 1, 9) or v.shape != q.shape:
                    raise ValueError(f"Expected T+1 observations in episode {eid}")
                positions.append(q[:length])
                velocities.append(v[:length])
                arm_labels.append(y)
                boundaries.append(list(bounds))
                targets.append({"initial": q[0].tolist(),
                                "approach": q[bounds[0], :7].tolist(),
                                "lift": q[bounds[2], :7].tolist()})
            qpos = np.stack(positions, axis=1)
            qvel = np.stack(velocities, axis=1)
            dense_y = np.stack(arm_labels, axis=1)
            events = np.r_[0, np.flatnonzero(np.any(np.diff(dense_y, axis=0) != 0, axis=1)) + 1]
            for t in events:
                active = dense_y[t - 1] if t else np.array([-1, -1])
                terminated = (active != dense_y[t]) if t else np.array([True, True])
                # Pair obs[t] with next skill label[t], never obs[t+1].
                # Executor status is supplied by offline annotated boundaries.
                features.append(decision_features(qpos[t], qvel[t], active, terminated))
                labels.append(dense_y[t])
                episodes.append(eid)
                steps.append(t)
            total_frames += length
            if eid in splits["train"]:
                skills.append(targets)
            audit.append({"episode": eid, "frames": length, "boundaries": boundaries})
    x, y = np.asarray(features), np.asarray(labels)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite observations")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=x, labels=y, episode=np.asarray(episodes), step=np.asarray(steps))
    # Calibrate a fixed placeholder library from TRAIN episodes only.
    # A real pretrained skill executor can replace this without changing the head.
    library = {"kind": "frozen joint-target placeholder skills calibrated on training demos",
               "arms": [{name: np.median([s[a][name] for s in skills], axis=0).tolist()
                         for name in ("initial", "approach", "lift")} for a in range(2)]}
    write_json(output.with_suffix(".skills.json"), library)
    report = {"source": "FACEONG/RoboFactory_Dataset", "task": "LiftBarrier", "seed": seed,
              "raw_sha256": sha256(raw / "LiftBarrier.h5"), "annotation": ANNOTATION,
              "feature_schema": "joint state(36), active skill onehot incl none(2x5), executor terminated(2)",
              "decision_protocol": "head runs at union of either arm's completion events; non-preemptive skills",
              "skills": SKILLS, "splits": splits, "source_frames": total_frames, "decisions": len(x),
              "class_counts": {name: int((y == i).sum()) for i, name in enumerate(SKILLS)},
              "episodes": audit}
    write_json(output.with_suffix(".json"), report)
    print(f"Prepared {len(ids)} real episodes / {total_frames} frames / {len(x)} decisions, split "
          + "/".join(str(len(s)) for s in splits.values()))
    return report
