"""Offline native-scene audit. Simulation dependencies are loaded by the CLI on demand."""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import robofactory
import torch
from mani_skill.utils.structs.pose import Pose

from .phase0_data import delta_actions, sha256, tasks, verify_files

Action = np.ndarray


def runtime(config: dict[str, Any]) -> dict[str, Any]:
    versions = {
        package: version(package) for package in ("mani_skill", "sapien", "torch")
    }
    for package, key in (
        ("mani_skill", "maniskill_version"),
        ("sapien", "sapien_version"),
    ):
        if versions[package] != config[key]:
            raise RuntimeError(
                f"Expected {package} {config[key]}, got {versions[package]}"
            )
    source = Path(robofactory.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if revision != config["robofactory_revision"]:
        raise RuntimeError(
            f"Unexpected RoboFactory source revision {revision} in {source}"
        )
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=source, check=True)
    asset_dir = Path(robofactory.ASSET_DIR)
    asset_pin = asset_dir / ".weave-asset-revision"
    if (
        not asset_pin.is_file()
        or asset_pin.read_text().strip() != config["asset_revision"]
    ):
        raise RuntimeError(
            "RoboFactory asset revision marker missing or wrong. Run the setup script."
        )
    return {
        "platform": platform.platform(),
        "versions": versions,
        "robofactory_revision": revision,
        "asset_revision": config["asset_revision"],
        "asset_sha256": {
            relative: sha256(asset_dir / relative)
            for relative in (
                "scenes/table/table.glb",
                "objects/pot_annotated/mobility.urdf",
            )
        },
    }


def _make(task: dict[str, Any], mode: str, backend: str, count: int = 1) -> gym.Env:
    return gym.make(
        task["native_id"],
        obs_mode="state_dict",
        control_mode=mode,
        reward_mode="none",
        render_mode=None,
        sim_backend=backend,
        num_envs=count,
    )


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().copy()


def _run_episode(
    env: gym.Env,
    task: dict[str, Any],
    episode: dict[str, Any],
    actions: dict[str, Action],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reset_kwargs = dict(episode["reset_kwargs"])
    seed = int(episode["episode_seed"])
    if reset_kwargs.setdefault("seed", seed) != seed:
        raise ValueError(f"Conflicting reset seeds for episode {episode['episode_id']}")
    # The upstream scene builder uses both global NumPy and Torch RNGs.
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs, _ = env.reset(**reset_kwargs)
    samples: dict[str, list[np.ndarray]] = {}
    success = []

    def capture(obs: dict[str, Any]) -> None:
        for arm in actions:
            proprio = obs["agent"][arm] if arm else obs["agent"]
            samples.setdefault(f"qpos/{arm}", []).append(_as_numpy(proprio["qpos"][0]))
        for name in task["pose_entities"]:
            entity = getattr(env.unwrapped, name)
            samples.setdefault(f"pose/{name}", []).append(
                _as_numpy(entity.pose.raw_pose[0])
            )
        for name in task.get("pose_articulations", []):
            joint_positions = getattr(env.unwrapped, name).get_qpos()
            samples.setdefault(f"joints/{name}", []).append(
                _as_numpy(joint_positions[0])
            )

    capture(obs)
    for step in range(len(next(iter(actions.values())))):
        command = {arm: array[step] for arm, array in actions.items()}
        obs, _, _, _, info = env.step(command.get("", command))
        success.append(bool(info["success"].item()))
        capture(obs)
    return np.asarray(success), {
        name: np.stack(trace) for name, trace in samples.items()
    }


def _trace_drift(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, float]:
    drift = {}
    for name, left in first.items():
        right = second[name]
        if name.startswith("pose/"):
            # A quaternion and its negative encode the same rotation.
            signs = np.where(np.sum(left[:, 3:] * right[:, 3:], axis=1) < 0, -1, 1)
            right = right.copy()
            right[:, 3:] *= signs[:, None]
        drift[name] = float(np.max(np.abs(left - right)))
    return drift


def _replay_task(
    data_file: Path,
    task: dict[str, Any],
    episodes: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    raw = _make(task, "pd_joint_pos", "cpu")
    try:
        delta = _make(task, "pd_joint_delta_pos", "cpu")
        try:
            with h5py.File(data_file, "r") as archive:
                rows = [
                    _replay_record(raw, delta, task, episode, archive, cfg)
                    for episode in episodes
                ]
        finally:
            delta.close()
        control_hz = raw.unwrapped.control_freq
        sim_hz = raw.unwrapped.sim_freq
    finally:
        raw.close()

    raw_rate = statistics.mean(row["raw_success"] for row in rows)
    delta_rate = statistics.mean(row["delta_success"] for row in rows)
    gap = abs(raw_rate - delta_rate)
    return {
        "control_hz": control_hz,
        "sim_hz": sim_hz,
        "episodes": rows,
        "raw_success_rate": raw_rate,
        "delta_success_rate": delta_rate,
        "conversion_gap": gap,
        "repeat_agreement_rate": statistics.mean(row["repeat_agrees"] for row in rows),
        "bc_precheck": (
            "pass"
            if raw_rate >= cfg["min_bc_raw_success"]
            and gap <= cfg["max_conversion_gap"] + 1e-12
            else "fail"
        ),
    }


def _replay_record(
    raw: gym.Env,
    delta: gym.Env,
    task: dict[str, Any],
    ep: dict[str, Any],
    archive: h5py.File,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    group = archive[f"traj_{ep['episode_id']}"]
    stored_actions = group["actions"]
    arms = (
        [""]
        if isinstance(stored_actions, h5py.Dataset)
        else sorted(stored_actions.keys())
    )
    if len(arms) != task["arms"]:
        raise ValueError(
            f"Unexpected action keys for episode {ep['episode_id']}: {arms}"
        )
    actions, qpos, converted = {}, {}, {}
    saturated = 0
    for arm in arms:
        array = (stored_actions[arm] if arm else stored_actions)[:].astype(np.float32)
        joints = group[f"obs/agent/{arm}/qpos" if arm else "obs/agent/qpos"][:]
        if len(array) != ep["elapsed_steps"] or not np.isfinite(array).all():
            raise ValueError(f"Invalid actions for episode {ep['episode_id']}")
        actions[arm] = array
        qpos[arm] = joints
        converted[arm], clipped = delta_actions(array, joints, cfg["joint_delta_limit"])
        saturated += clipped

    raw_success, raw_trace = _run_episode(raw, task, ep, actions)
    repeated_success, repeated_trace = _run_episode(raw, task, ep, actions)
    delta_success, delta_trace = _run_episode(delta, task, ep, converted)
    drift = _trace_drift(raw_trace, repeated_trace)
    return {
        "episode_id": ep["episode_id"],
        "episode_seed": ep["episode_seed"],
        "steps": ep["elapsed_steps"],
        "published_final_success": bool(group["success"][-1]),
        "raw_success": bool(raw_success[-1]),
        "delta_success": bool(delta_success[-1]),
        "repeat_success": bool(repeated_success[-1]),
        "repeat_agrees": bool(
            np.array_equal(raw_success, repeated_success)
            and max(drift.values()) <= cfg["repeat_atol"]
        ),
        "repeat_max_abs_drift": drift,
        "raw_qpos_rmse_rad": {
            arm or "panda": float(
                np.sqrt(
                    np.mean((raw_trace[f"qpos/{arm}"][:, :7] - qpos[arm][:, :7]) ** 2)
                )
            )
            for arm in arms
        },
        "delta_qpos_rmse_rad": {
            arm or "panda": float(
                np.sqrt(
                    np.mean((delta_trace[f"qpos/{arm}"][:, :7] - qpos[arm][:, :7]) ** 2)
                )
            )
            for arm in arms
        },
        "delta_saturated_components": saturated,
    }


def replay(data_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    environment = runtime(config)
    verify_files(data_dir)
    summaries = {}
    for name, task, _, episodes, _ in tasks(data_dir, config):
        print(f"Replaying {name}", flush=True)
        try:
            summaries[name] = _replay_task(
                data_dir / f"{name}.h5", task, episodes, config["replay"]
            )
        except Exception as exc:  # noqa: BLE001 - preserve other tasks' results
            summaries[name] = {"error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()
    return {"runtime": environment, "tasks": summaries}


def _gpu_mib() -> int:
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return sum(
        int(value)
        for pid, value in (row.split(", ") for row in rows.splitlines() if row.strip())
        if int(pid) == os.getpid()
    )


def _hold_action(base: gym.Env) -> dict[str, torch.Tensor]:
    return {
        name: torch.cat(
            (
                agent.robot.get_qpos()[:, :7],
                torch.ones((base.num_envs, 1), dtype=torch.float32, device=base.device),
            ),
            dim=1,
        )
        for name, agent in base.agent.agents_dict.items()
    }


def _contact_probe(env: gym.Env, cfg: dict[str, Any]) -> dict[str, Any]:
    base = env.unwrapped
    env.reset(seed=cfg["seed"])
    first, second = base.agent.agents
    # Move one robot's base into deliberate overlap in lane zero during reset only.
    pose = second.robot.pose.raw_pose.clone()
    pose[0] = first.robot.pose.raw_pose[0]
    pose[0, 0] += cfg["contact_probe_offset_m"]
    second.robot.set_pose(Pose.create(pose))
    if base.gpu_sim_enabled:
        base.scene._gpu_apply_all()
        base.scene.px.gpu_update_articulation_kinematics()
        base.scene._gpu_fetch_all()
    actions = _hold_action(base)
    peak = torch.zeros(base.num_envs, device=base.device)
    left_links = first.robot.get_links()
    right_links = second.robot.get_links()
    for _ in range(cfg["contact_probe_steps"]):
        env.step(actions)
        for left in left_links:
            for right in right_links:
                force = base.scene.get_pairwise_contact_forces(left, right)
                if tuple(force.shape) != (base.num_envs, 3):
                    raise RuntimeError(f"Unexpected pairwise force shape {force.shape}")
                peak = torch.maximum(peak, torch.linalg.vector_norm(force, dim=1))
    peaks = _as_numpy(peak)
    other_peak = float(np.max(peaks[1:])) if len(peaks) > 1 else None
    threshold = cfg["contact_threshold_n"]
    return {
        "pair_count": len(left_links) * len(right_links),
        "challenged_peak_n": float(peaks[0]),
        "unmodified_peak_n": other_peak,
        "passed": bool(
            peaks[0] > threshold and (other_peak is None or other_peak <= threshold)
        ),
    }


def benchmark(
    config: dict[str, Any], task_name: str, count: int, backend: str
) -> dict[str, Any]:
    environment = runtime(config)
    cfg = config["benchmark"]
    gpu = backend == "gpu"
    if gpu and (platform.system() != "Linux" or not torch.cuda.is_available()):
        raise RuntimeError(
            "GPU throughput requires Linux and a visible NVIDIA CUDA device."
        )
    if gpu:
        torch.cuda.init()
    before = _gpu_mib() if gpu else None
    env = _make(config["tasks"][task_name], "pd_joint_pos", backend, count)
    try:
        base = env.unwrapped
        if (
            base.gpu_sim_enabled != gpu
            or base.num_envs != count
            or len(base.agent.agents) != 2
        ):
            raise RuntimeError("Unexpected backend, batch size, or arm count")
        samples = []
        for repetition in range(cfg["repeats"]):
            seed = cfg["seed"] + repetition
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.reset(seed=seed)
            actions = _hold_action(base)
            for _ in range(cfg["warmup_steps"]):
                env.step(actions)
            if gpu:
                torch.cuda.synchronize(base.device)
            start = time.perf_counter()
            for _ in range(cfg["measure_steps"]):
                env.step(actions)
            if gpu:
                torch.cuda.synchronize(base.device)
            samples.append(count * cfg["measure_steps"] / (time.perf_counter() - start))
        resident = _gpu_mib() if gpu else None
        return {
            "task": task_name,
            "backend": backend,
            "num_envs": count,
            "runtime": environment,
            "gpu": torch.cuda.get_device_name(base.device) if gpu else None,
            "control_hz": base.control_freq,
            "sim_hz": base.sim_freq,
            "env_steps_per_second": samples,
            "median_env_steps_per_second": statistics.median(samples),
            "process_gpu_mib": resident,
            "incremental_gpu_mib_per_env": (resident - before) / count if gpu else None,
            "contact_probe": _contact_probe(env, cfg),
        }
    finally:
        env.close()
