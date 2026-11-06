"""Evidence runner for the instrumented two-Panda cell."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TextIO

import torch
from mani_skill.utils.structs.pose import Pose

from .envs import two_arm_cell
from .envs.integrity import (
    learned_rollout_guard,
    physics_write_guard,
    scan_env_source,
    scan_rollout_log,
)
from .phase0_data import sha256
from .phase0_sim import runtime


def _check(condition: Any, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def _write_log(stream: TextIO, event: dict[str, Any]) -> None:
    stream.write(json.dumps(event, allow_nan=False) + "\n")
    stream.flush()


def _contact_probe(env: Any, config: dict[str, Any], stream: TextIO) -> dict[str, Any]:
    env.reset(seed=config["seed"])
    first, second = env.agent.agents
    # Deliberate overlap in one lane only, before the first physics step.
    pose = second.robot.pose.raw_pose.clone()
    pose[0] = first.robot.pose.raw_pose[0]
    pose[0, 0] += config["contact_probe_offset_m"]
    second.robot.set_pose(Pose.create(pose))
    env.scene._gpu_apply_all()
    env.scene.px.gpu_update_articulation_kinematics()
    env.scene._gpu_fetch_all()
    _write_log(stream, {"event": "probe_reset", "step": 0, "pose_write_lane": 0})
    actions = {
        key: torch.zeros((env.num_envs, 8), device=env.device)
        for key in env.agent.agents_dict
    }
    for action in actions.values():
        action[:, -1] = 1
    for step in range(1, config["contact_probe_steps"] + 1):
        _guarded_step(env, actions, step, stream)
    peaks = env.contact_monitor.peak_n[:, 0]
    threshold = env.contact_monitor.threshold_n
    _check(
        peaks[0] > threshold, "The deliberately overlapping arms produced no contact"
    )
    _check((peaks[1:] <= threshold).all(), "Contact appeared in an untouched lane")
    return {
        "challenged_peak_n": float(peaks[0]),
        "other_lanes_peak_n": float(peaks[1:].max()),
        "challenged_contact_frames": int(env.contact_monitor.frames[0, 0]),
    }


def _guarded_step(
    env: Any,
    action: dict[str, torch.Tensor],
    step: int,
    stream: TextIO,
) -> tuple[dict[str, Any], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        with physics_write_guard(events, step), learned_rollout_guard():
            obs, _, _, _, info = env.step(action)
    finally:
        for event in events:
            _write_log(stream, event)
    _write_log(stream, {"event": "control_step", "step": step})
    return obs, info


def _snapshot(env: Any, extra: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "objects": extra["objects_world"].clone(),
        "fixtures": extra["fixtures_world"].clone(),
        "lid": extra["lid_angle"].clone(),
        "active": extra["active_objects"].clone(),
        "joint_positions": torch.stack(
            [agent.robot.get_qpos() for agent in env.agent.agents], dim=1
        ),
        "elapsed": env.elapsed_steps.clone(),
        "contact_frames": env.contact_monitor.frames.clone(),
    }


def _run(env: Any, config: dict[str, Any], stream: TextIO) -> dict[str, Any]:
    seed = config["seed"]
    initial, _ = env.reset(seed=seed)
    reset = _snapshot(env, initial["extra"])
    repeated, _ = env.reset(seed=seed)
    for key, value in _snapshot(env, repeated["extra"]).items():
        _check(torch.allclose(value, reset[key]), f"Same-seed reset changed {key}")
    _check(
        torch.unique(reset["objects"].flatten(1), dim=0).size(0) > 1,
        "Object layouts are identical across all lanes",
    )
    _check(
        torch.unique(reset["lid"]).numel() > 1,
        "Lid angle is identical across all lanes",
    )
    _check(
        env.pot.get_qpos().shape == (env.num_envs, 1), "Expected one physical lid joint"
    )

    device = env.device
    lanes = torch.arange(env.num_envs, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    env.begin_footprint(0, lanes)
    start_bounds = env.footprint_bounds[:, 0].clone()
    _write_log(stream, {"event": "random_policy_reset", "step": 0, "seed": seed})
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for step in range(1, config["steps"] + 1):
        with learned_rollout_guard():
            actions = {
                name: (
                    torch.rand((env.num_envs, 8), generator=generator, device=device)
                    * 2
                    - 1
                )
                * config["random_action_scale"]
                for name in env.agent.agents_dict
            }
        obs, info = _guarded_step(env, actions, step, stream)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    bounds = env.end_footprint(0, lanes)
    _check(
        (bounds[:, 0] <= start_bounds[:, 0]).all(),
        "Swept minimum excluded the starting sample",
    )
    _check(
        (bounds[:, 1] >= start_bounds[:, 1]).all(),
        "Swept maximum excluded the starting sample",
    )
    _check(torch.isfinite(bounds).all(), "Non-finite swept footprint")
    _check(torch.isfinite(info["contact_peak_n"]).all(), "Non-finite contact force")
    _check(
        torch.isfinite(obs["extra"]["objects_world"]).all(), "Non-finite object state"
    )
    _check(
        (info["contact_frames"] <= config["steps"]).all(),
        "Contact count exceeded control steps",
    )
    before = _snapshot(env, obs["extra"])
    partial, _ = env.reset(options={"env_idx": lanes[:1]})
    after = _snapshot(env, partial["extra"])
    for key, value in after.items():
        _check(
            torch.allclose(before[key][1:], value[1:]),
            f"Partial reset changed {key} in untouched lanes",
        )
    _check(after["elapsed"][0] == 0, "Selected lane did not reset")
    _check(
        not torch.allclose(after["objects"][0], before["objects"][0]),
        "Selected lane did not resample",
    )
    _write_log(stream, {"event": "partial_reset", "step": 0, "lane": 0})
    return {
        "same_seed_reset": True,
        "lane_randomization": True,
        "partial_reset_isolation": True,
        "swept_footprint": True,
        "random_policy_steps": config["steps"],
        "monitored_env_steps_per_second": env.num_envs * config["steps"] / elapsed,
        "random_policy_peak_n_by_category": info["contact_peak_n"].amax(dim=0).tolist(),
        "positive_contact_probe": _contact_probe(env, config, stream),
    }


def smoke(config: dict[str, Any], results_dir: Path) -> dict[str, Any]:
    environment = runtime(config["runtime"])
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU with CUDA_VISIBLE_DEVICES before Phase 1."
        )
    violations = [
        violation
        for filename in (
            "two_arm_cell.py",
            "objects.py",
            "reset_dists.py",
            "contact_monitor.py",
        )
        for violation in scan_env_source(
            Path(two_arm_cell.__file__).with_name(filename)
        )
    ]
    _check(not violations, f"Environment source scan found: {violations}")
    results_dir.mkdir(parents=True, exist_ok=True)
    smoke_config = config["smoke"]
    if smoke_config["num_envs"] < 2 or smoke_config["steps"] < 1:
        raise ValueError(
            "Phase 1 smoke requires at least two lanes and one control step"
        )
    trace = results_dir / "rollout.jsonl"
    with trace.open("w") as stream:
        env = two_arm_cell.TwoArmCellEnv(config["cell"], smoke_config["num_envs"])
        try:
            checks = _run(env, smoke_config, stream)
        finally:
            env.close()
    violations = scan_rollout_log(trace)
    _check(not violations, f"Rollout trace contained prohibited writes: {violations}")
    return {
        "runtime": environment,
        "gpu": torch.cuda.get_device_name(),
        "num_envs": smoke_config["num_envs"],
        "control_hz": env.control_freq,
        "source_scan": "pass",
        "rollout_physics_writes": len(violations),
        "rollout_log_sha256": sha256(trace),
        **checks,
    }
