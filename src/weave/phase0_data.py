"""Verify the original RoboFactory data and restrict the audit to training episodes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import h5py
import numpy as np

from skill_composer.data import split_episodes

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/env/robofactory_demos.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(data_dir: Path, download: bool = False) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    for task in manifest["tasks"]:
        for file in task["files"]:
            path = data_dir / file["file"]
            if not path.exists() and download:
                url = (
                    f"https://huggingface.co/datasets/{manifest['source']}/resolve/"
                    f"{manifest['revision']}/{task['task']}/{path.name}"
                )
                data_dir.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix + ".part")
                print(f"Downloading {path.name}", flush=True)
                try:
                    with (
                        urlopen(url, timeout=120) as response,
                        partial.open("wb") as output,
                    ):
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(block)
                    verify_file(partial, file)
                    partial.replace(path)
                finally:
                    partial.unlink(missing_ok=True)
            else:
                verify_file(path, file)
    return manifest


def verify_file(path: Path, file: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts/download_data.sh first.")
    if path.stat().st_size != file["bytes"] or sha256(path) != file["sha256"]:
        raise ValueError(f"Manifest mismatch for {path}")


def training_sample(
    episodes: list[dict[str, Any]], split_seed: int, selection_seed: int, count: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    published_successes = {
        ep["episode_id"]: ep for ep in episodes if ep.get("success") is True
    }
    # Match the v1 filter and preserve publisher order before permuting.
    split = split_episodes(list(published_successes), seed=split_seed)
    train = split["train"]
    if count > len(train):
        raise ValueError(f"Requested {count} episodes but train only has {len(train)}")
    chosen = np.random.default_rng(selection_seed).choice(
        train, size=count, replace=False
    )
    return [published_successes[int(episode_id)] for episode_id in sorted(chosen)], {
        name: len(ids) for name, ids in split.items()
    }


def tasks(
    data_dir: Path, config: dict[str, Any]
) -> Iterator[
    tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, int]]
]:
    replay = config["replay"]
    for name, task in config["tasks"].items():
        metadata = json.loads((data_dir / f"{name}.json").read_text())
        env_info = metadata["env_info"]
        if env_info["env_id"] != task["recorded_id"]:
            raise ValueError(f"{name} has unexpected recorded ID {env_info['env_id']}")
        kwargs = env_info["env_kwargs"]
        if kwargs["control_mode"] != "pd_joint_pos" or kwargs["sim_backend"] != "cpu":
            raise ValueError(f"{name} has unexpected controller or backend")
        episodes, sizes = training_sample(
            metadata["episodes"],
            config["split_seed"],
            replay["selection_seed"],
            replay["episodes_per_task"],
        )
        yield name, task, metadata, episodes, sizes


def delta_actions(
    actions: np.ndarray, qpos: np.ndarray, joint_limit: float
) -> tuple[np.ndarray, int]:
    """Convert targets from recorded pre-action qpos to fixed normalized delta commands."""
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(f"Unexpected action shape {actions.shape}")
    if qpos.shape != (len(actions) + 1, 9):
        raise ValueError(
            f"Unexpected qpos shape {qpos.shape} for {len(actions)} actions"
        )
    if not np.isfinite(qpos).all() or joint_limit <= 0:
        raise ValueError(
            "Conversion requires finite qpos and a positive joint delta limit"
        )
    scaled = (actions[:, :7] - qpos[:-1, :7]) / joint_limit
    delta = actions.copy()
    delta[:, :7] = np.clip(scaled, -1.0, 1.0)
    return delta, int(np.count_nonzero(np.abs(scaled) > 1.0))


def describe_trajectory(group: h5py.Group) -> dict[str, Any]:
    leaves: dict[str, Any] = {}

    def describe(path: str, entry: Any) -> None:
        if isinstance(entry, h5py.Dataset):
            leaves[path] = {"shape": list(entry.shape), "dtype": str(entry.dtype)}

    group.visititems(describe)
    return leaves


def inventory(data_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    manifest = verify_files(data_dir)
    summaries = {}
    for name, task, metadata, episodes, split_sizes in tasks(data_dir, config):
        with h5py.File(data_dir / f"{name}.h5", "r") as archive:
            leaves = describe_trajectory(archive[f"traj_{episodes[0]['episode_id']}"])
        summaries[name] = {
            "recording": metadata["env_info"],
            "recording_maniskill_commit": metadata.get("commit_info"),
            "native_id": task["native_id"],
            "split_sizes": split_sizes,
            "sample_episode": episodes[0]["episode_id"],
            "episode_seeds": [ep["episode_seed"] for ep in episodes],
            "hdf5_datasets": leaves,
        }
    return {"verified_manifest": manifest, "tasks": summaries}
