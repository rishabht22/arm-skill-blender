"""Evaluate a saved PPO policy with its training observation normalization."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.policy_env import PolicyEnv


RUN_FILES = ("policy.zip", "vecnormalize.pkl", "config.json", "environment.json")


def latest_run(output_root):
    """Find the newest completed run, skipping unfinished training runs."""
    candidates = sorted(
        directory
        for directory in Path(output_root).glob("*")
        if directory.is_dir()
        and all((directory / name).is_file() for name in RUN_FILES)
    )
    if not candidates:
        raise FileNotFoundError(f"No completed policy runs found in {output_root}")
    return candidates[-1]


def evaluate(run_dir, *, episodes=3, seed=1000, render=True, device="cpu"):
    """Run complete episodes with a frozen policy; return per-episode results."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    run_dir = Path(run_dir).resolve()
    for name in RUN_FILES:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Missing saved run file: {run_dir / name}")
    settings = json.loads((run_dir / "config.json").read_text())
    expected = json.loads((run_dir / "environment.json").read_text())

    raw_env = PolicyEnv(seed=seed, horizon=settings["horizon"])
    base_env = None
    env = None
    try:
        actual = {
            "robots": raw_env.robot_names,
            "observation_keys": raw_env.observation_keys,
            "observation_dim": raw_env.observation_space.shape[0],
            "action_dim": raw_env.action_space.shape[0],
        }
        for key, value in actual.items():
            if expected[key] != value:
                raise ValueError(
                    f"Saved policy expects {key}={expected[key]}, but the current scene has {value}. "
                    "Use the same robot and observation setup as the training run."
                )

        base_env = DummyVecEnv([lambda: raw_env])
        env = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), base_env)
        # Freeze the saved observation statistics and report original rewards.
        env.training = False
        env.norm_reward = False
        model = PPO.load(str(run_dir / "policy.zip"), env=env, device=device)
        env.seed(seed)
        observation = env.reset()

        print(f"Loaded policy: {run_dir / 'policy.zip'}", flush=True)
        print(f"Robots: {', '.join(raw_env.robot_names)}. Evaluation only.", flush=True)

        viewer_context = nullcontext(None)
        if render:
            import mujoco
            import mujoco.viewer

            viewer_context = mujoco.viewer.launch_passive(
                raw_env.sim_env.sim.model._model,
                raw_env.sim_env.sim.data._data,
            )

        results = []
        episode_reward = 0.0
        episode_steps = 0
        episode_success = False
        with viewer_context as viewer:
            if viewer is not None:
                with viewer.lock():
                    viewer.opt.geomgroup[0] = 0
                    viewer.opt.geomgroup[1] = 1
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat[:] = (0.0, 0.0, 0.9)
                    viewer.cam.distance = 2.5
                    viewer.cam.azimuth = 90
                    viewer.cam.elevation = -25

            while len(results) < episodes:
                if viewer is not None and not viewer.is_running():
                    break
                started = time.perf_counter()
                action, _ = model.predict(observation, deterministic=True)
                with viewer.lock() if viewer is not None else nullcontext():
                    observation, rewards, dones, infos = env.step(action)
                if not (np.isfinite(observation).all() and np.isfinite(rewards).all()):
                    raise ValueError("Evaluation produced a non-finite observation or reward")

                episode_reward += float(rewards[0])
                episode_steps += 1
                # Count an episode as successful if it achieves the lift at any step.
                episode_success |= bool(infos[0].get("is_success", False))
                if dones[0]:
                    results.append({
                        "reward": episode_reward,
                        "steps": episode_steps,
                        "success": episode_success,
                    })
                    print(
                        f"Episode {len(results)}/{episodes}: reward={episode_reward:.2f}, "
                        f"steps={episode_steps}, lift_success={episode_success}",
                        flush=True,
                    )
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_success = False
                    # DummyVecEnv has already reset the completed episode.

                if viewer is not None:
                    viewer.sync()
                    remaining = raw_env.sim_env.control_timestep - (time.perf_counter() - started)
                    time.sleep(max(0.0, remaining))

        if results:
            successes = sum(result["success"] for result in results)
            mean_reward = np.mean([result["reward"] for result in results])
            print(
                f"Successful lifts: {successes}/{len(results)}. Mean reward: {mean_reward:.2f}.",
                flush=True,
            )
        else:
            print("Viewer closed before an episode finished.", flush=True)
        return results
    finally:
        if env is not None:
            env.close()
        elif base_env is not None:
            base_env.close()
        else:
            raw_env.close()
