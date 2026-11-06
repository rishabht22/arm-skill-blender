import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SKILLS = ("approach", "grasp", "lift", "hold")
ARMS = ("panda-0", "panda-1")
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed/lift_barrier.npz"
RUN = ROOT / "outputs/baseline"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_features(qpos, qvel):
    """Both arms' CURRENT measured joint positions and velocities, in arm order.

    No action, episode ID, clock, success flag, or future state is a policy input.
    """
    return np.concatenate([qpos, qvel], axis=-1).reshape(-1).astype(np.float32)


def decision_features(qpos, qvel, active, terminated):
    """State plus the skill executor's causal completion interface.

    active=-1 means no previous skill. Termination is supplied by the executor,
    not predicted by the head. Training uses the offline annotated boundaries.
    """
    context = np.eye(5, dtype=np.float32)[np.asarray(active) + 1].reshape(-1)
    return np.r_[state_features(qpos, qvel), context, np.asarray(terminated, dtype=np.float32)].astype(np.float32)
