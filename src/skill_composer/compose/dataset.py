"""Derive two-arm decision labels from a published four-arm delivery recording."""

import json
import shutil
import tempfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from ..cell.dataset import REVISION, _runs
from ..common import ROOT, sha256, write_json
from ..data import split_episodes
from .domain import Action, State, candidates, transition

PROJECT = ROOT / "projects/goal_composition"
RAW = PROJECT / "data/raw"
DATA = PROJECT / "data/processed/decisions.json"
RUN = PROJECT / "outputs"

TASK = "LongPipelineDelivery"
REPOSITORY = "FACEONG/RoboFactory_Dataset"
UPSTREAM = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{TASK}"
ARM_MAP = {0: "panda-1", 1: "panda-0"}
REACH = {"source": (0,), "dock0": (0, 1), "goal": (1,)}

# The large HDF5 file is verified against its upstream Git LFS object ID.
CHECKSUMS = {
    f"{TASK}.json": "4ce0ad69673b632f9c3feb8ccde8e1d1be58f2d7a260a587f9d755a0aa120437",
    f"{TASK}.h5": "b4e123df9bcdb5739c29e44336a992c869ee4d7f44c24c04ca368ecf62fe984c",
}


def _verified_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run `python -m skill_composer compose download`.")
    digest = sha256(path)
    if digest != CHECKSUMS[path.name]:
        raise ValueError(f"{path} differs from pinned RoboFactory revision {REVISION}")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}


def download(raw=RAW):
    raw = Path(raw)
    raw.mkdir(parents=True, exist_ok=True)
    for filename, expected in CHECKSUMS.items():
        path = raw / filename
        if path.is_file():
            _verified_file(path)
            print(f"Verified published {filename}", flush=True)
            continue

        print(f"Downloading {filename} from RoboFactory revision {REVISION}", flush=True)
        with tempfile.TemporaryDirectory(dir=raw, prefix=".download-") as staging:
            partial = Path(staging) / filename
            with urllib.request.urlopen(f"{UPSTREAM}/{filename}", timeout=60) as response:
                with partial.open("wb") as file:
                    shutil.copyfileobj(response, file)
            if sha256(partial) != expected:
                raise ValueError(f"Upstream checksum mismatch for {filename}")
            partial.replace(path)


def _gripper_cycle(commands):
    gripper = commands[:, -1]
    delta = np.diff(gripper, prepend=gripper[0])
    closes = _runs(delta < -.025)
    if len(closes) != 1:
        raise ValueError("Expected one gripper close in the recorded stage")
    close_start, close_end = closes[0]
    releases = [(start, end) for start, end in _runs(delta > .025) if start >= close_end]
    if len(releases) != 1:
        raise ValueError("Expected one gripper release after the close")
    release_start, release_end = releases[0]
    return int(close_start), int(close_end), int(release_start), int(release_end)


def _skill_starts(commands, predecessor_release):
    close_start, close_end, release_start, release_end = _gripper_cycle(commands)
    speed = np.r_[0, np.linalg.norm(np.diff(commands[:, :7], axis=0), axis=1)]
    approach = np.flatnonzero(speed[predecessor_release:close_start] > .003)
    lift = [
        (start, end) for start, end in _runs(speed > .003)
        if close_end <= start < release_start and end - start >= 3
    ]
    if not len(approach) or not lift:
        raise ValueError("Missing recorded approach or post-grasp lift in delivery stage")
    pick_start = predecessor_release + int(approach[0])
    place_start = int(lift[0][1]) + 1
    if not pick_start < close_start < place_start <= release_start < release_end:
        raise ValueError("Inconsistent boundaries for a serial delivery stage")
    return pick_start, place_start, release_end


def _episode_examples(episode_id, group):
    recorded = group["actions"]
    arms = {name: recorded[name][:] for name in ("panda-2", "panda-1", "panda-0")}
    lengths = {len(commands) for commands in arms.values()}
    if len(lengths) != 1 or any(
        commands.ndim != 2 or commands.shape[1] != 8 or not np.isfinite(commands).all()
        for commands in arms.values()
    ):
        raise ValueError("Expected synchronized RoboFactory (T, 8) joint and gripper actions")

    supply_release = _gripper_cycle(arms["panda-2"])[-1]
    pick, stage, shared_release = _skill_starts(arms["panda-1"], supply_release)
    receive, finish, _ = _skill_starts(arms["panda-0"], shared_release)
    source_steps = [
        (pick, Action("pick", 0, "source")),
        (stage, Action("place", 0, "dock0")),
        (receive, Action("pick", 1, "dock0")),
        (finish, Action("place", 1, "goal")),
    ]
    state = State(location="source", holder=None, reach=REACH)
    examples = []
    for frame, action in source_steps:
        if action not in candidates(state):
            raise ValueError(f"Expert action {action} is invalid in projected state {state}")
        examples.append({
            "episode": int(episode_id),
            "frame": frame,
            "source_arm": ARM_MAP[action.arm],
            "state": asdict(state),
            "action": asdict(action),
        })
        state = transition(state, action)
    if state.location != state.goal or state.holder is not None:
        raise ValueError("Projected delivery did not reach the terminal goal")
    return examples, next(iter(lengths))


def prepare(raw=RAW, output=DATA):
    raw, output = Path(raw), Path(output)
    hdf5_path, metadata_path = raw / f"{TASK}.h5", raw / f"{TASK}.json"
    source_files = [_verified_file(path) for path in (hdf5_path, metadata_path)]
    metadata = json.loads(metadata_path.read_text())
    if metadata["env_info"]["env_kwargs"]["control_mode"] != "pd_joint_pos":
        raise ValueError("Skill boundaries require the published joint-position controls")

    episode_ids = sorted(
        int(episode["episode_id"]) for episode in metadata["episodes"] if episode.get("success") is True
    )
    splits = split_episodes(episode_ids)
    examples = []
    frames = 0
    with h5py.File(hdf5_path, "r") as recordings:
        for episode_id in episode_ids:
            try:
                decisions, length = _episode_examples(episode_id, recordings[f"traj_{episode_id}"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Cannot annotate {TASK} episode {episode_id}: {exc}") from exc
            examples.extend(decisions)
            frames += length

    report = {
        "source": REPOSITORY,
        "source_url": UPSTREAM,
        "task": TASK,
        "revision": REVISION,
        "source_files": source_files,
        "source_episodes": len(episode_ids),
        "source_frames": frames,
        "decision_events": len(examples),
        "branching_decision_events": sum(len(candidates(State(**row["state"]))) > 1 for row in examples),
        "distinct_abstract_observations": len({json.dumps(row["state"], sort_keys=True) for row in examples}),
        "source_arm_mapping": ARM_MAP,
        "annotation": (
            "The final two observed delivery stages are projected to two abstract arms. "
            "Recorded gripper ramps and joint-target motion identify frame boundaries. "
            "Skill targets, object location and holder, and site reachability are weak semantic "
            "annotations based on the public task script. Reachability is not measured in the recordings. "
            "The receiving arm's final pick and place are the observed direct-delivery suffix. "
            "All rows from a source episode share one split. No additional expert trajectories are generated."
        ),
        "splits": splits,
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".prepare-") as staging:
        temporary = Path(staging) / output.name
        write_json(temporary, report)
        temporary.replace(output)
    print(f"Prepared {len(episode_ids)} published episodes and {len(examples)} decisions at {output}", flush=True)
    return report
