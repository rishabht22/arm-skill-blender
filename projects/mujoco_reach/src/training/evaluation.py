"""Evaluate a saved PPO policy with its training observation normalization."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.policy_env import describe_policy_env, make_policy_env
from src.policies.action_filters import slow_near_target


RUN_FILES = ("policy.zip", "vecnormalize.pkl", "config.json", "environment.json")


def latest_run(output_root, *, task="reach_handle", robot_index=0):
    """Find a completed run for the requested task and controlled arm."""
    candidates = []
    for directory in Path(output_root).glob("*"):
        if not directory.is_dir() or not all((directory / name).is_file() for name in RUN_FILES):
            continue
        metadata = json.loads((directory / "environment.json").read_text())
        if metadata.get("task", "two_arm_lift") != task:
            continue
        if task == "reach_handle" and metadata.get("reach", {}).get("robot_index", 0) != robot_index:
            continue
        candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(
            f"No completed {task} runs for arm {robot_index + 1} found in {output_root}. "
            "Train this arm with: python -m experiments.basic_policies "
            f"--arm {'jaco' if robot_index == 1 else 'ur5e'} --train"
        )
    return max(candidates)


def evaluate(run_dir, *, episodes=3, seed=1000, render=True, device="cpu", ur5e_slowdown_radius=0.08):
    """Run complete episodes with a frozen policy; return per-episode results."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    slow_near_target(np.zeros(3), 0.0, radius=ur5e_slowdown_radius)
    run_dir = Path(run_dir).resolve()
    for name in RUN_FILES:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Missing saved run file: {run_dir / name}")
    settings = json.loads((run_dir / "config.json").read_text())
    expected = json.loads((run_dir / "environment.json").read_text())
    # Older lift checkpoints predate the explicit task field.
    task = expected.get("task", "two_arm_lift")
    expected["task"] = task
    reach_settings = expected.get("reach", settings.get("reach", {})).copy()
    if task == "reach_handle":
        # Checkpoints created before arm selection always controlled the first arm.
        reach_settings.setdefault("robot_index", 0)
        expected["reach"] = reach_settings

    raw_env = make_policy_env(
        task=task, seed=seed, horizon=settings["horizon"], **reach_settings
    )
    base_env = None
    env = None
    try:
        actual = describe_policy_env(raw_env)
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
        print(f"Task: {task}. Robots: {', '.join(raw_env.robot_names)}. Evaluation only.", flush=True)
        if task == "reach_handle":
            print(
                f"Controlled arm: {raw_env.robot_names[raw_env.robot_index]}. "
                f"Target height: {raw_env.target_height * 100:.1f} cm; hold: {raw_env.hold_seconds:.1f} s.",
                flush=True,
            )
            if raw_env.robot_index == 0:
                print(f"UR5e command slowdown radius: {100 * ur5e_slowdown_radius:.1f} cm (0 disables).", flush=True)

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
                if task == "reach_handle" and raw_env.robot_index == 0:
                    distance = float(np.linalg.norm(raw_env.target_pos - raw_env.eef_position))
                    action[0] = slow_near_target(action[0], distance, radius=ur5e_slowdown_radius)
                with viewer.lock() if viewer is not None else nullcontext():
                    observation, rewards, dones, infos = env.step(action)
                    if viewer is not None and task == "reach_handle":
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[0],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.full(3, raw_env.success_distance),
                            pos=raw_env.target_pos,
                            mat=np.eye(3).ravel(),
                            rgba=np.array([0.2, 1.0, 0.2, 0.35]),
                        )
                        viewer.user_scn.ngeom = 1
                if not (np.isfinite(observation).all() and np.isfinite(rewards).all()):
                    raise ValueError("Evaluation produced a non-finite observation or reward")

                episode_reward += float(rewards[0])
                episode_steps += 1
                # ReachEnv only reports success after holding inside the goal.
                episode_success |= bool(infos[0].get("is_success", False))
                if dones[0]:
                    results.append({
                        "ur5e_slowdown_radius": ur5e_slowdown_radius if task == "reach_handle" and raw_env.robot_index == 0 else 0.0,
                        "reward": episode_reward,
                        "steps": episode_steps,
                        "success": episode_success,
                    })
                    print(
                        f"Episode {len(results)}/{episodes}: reward={episode_reward:.2f}, "
                        f"steps={episode_steps}, success={episode_success}",
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
                f"Successful {task} episodes: {successes}/{len(results)}. Mean reward: {mean_reward:.2f}.",
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
