"""Evaluate two independent saved policies acting together in one scene."""

import time
from contextlib import nullcontext

import numpy as np

from src.envs.combined_reach import CombinedReachEnv
from src.policies.action_filters import slow_near_target
from src.policies.saved_reach import SavedReachPolicy


def evaluate_both(
    ur5e_run_dir, jaco_run_dir, *, episodes=3, seed=1000, render=True,
    device="cpu", hold_seconds=3.0, ur5e_slowdown_radius=0.08,
):
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    slow_near_target(np.zeros(3), 0.0, radius=ur5e_slowdown_radius)
    policies = [
        SavedReachPolicy(ur5e_run_dir, robot_index=0, device=device),
        SavedReachPolicy(jaco_run_dir, robot_index=1, device=device),
    ]
    env = CombinedReachEnv(
        [policy.reach_settings for policy in policies],
        seed=seed,
        horizon=max(policy.config["horizon"] for policy in policies),
        hold_seconds=hold_seconds,
    )
    try:
        for index, policy in enumerate(policies):
            policy.validate_environment(env.policy_metadata(index))
            print(
                f"{env.robot_names[index]} policy: {policy.run_dir}\n"
                f"  Target: {policy.reach_settings['target_height'] * 100:.1f} cm above its handle.",
                flush=True,
            )
        print(f"Both arms must hold their targets together for {env.hold_seconds:.1f} s. Evaluation only.", flush=True)
        print(f"UR5e command slowdown radius: {100 * ur5e_slowdown_radius:.1f} cm (0 disables).", flush=True)

        viewer_context = nullcontext(None)
        if render:
            import mujoco
            import mujoco.viewer

            viewer_context = mujoco.viewer.launch_passive(env.sim_env.sim.model._model, env.sim_env.sim.data._data)

        results = []
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

            for episode in range(episodes):
                if viewer is not None and not viewer.is_running():
                    break
                with viewer.lock() if viewer is not None else nullcontext():
                    observation, _ = env.reset(seed=seed + episode)
                for step in range(env.sim_env.horizon):
                    if viewer is not None and not viewer.is_running():
                        break
                    started = time.perf_counter()
                    # Both predictions read the same simulation timestep.
                    action = np.concatenate([
                        policy.predict(observation[f"robot{index}"])
                        for index, policy in enumerate(policies)
                    ])
                    distance = float(np.linalg.norm(env.target_positions[0] - env.sim_env._eef0_xpos))
                    action[:3] = slow_near_target(
                        action[:3], distance, radius=ur5e_slowdown_radius
                    )
                    with viewer.lock() if viewer is not None else nullcontext():
                        observation, _, terminated, truncated, info = env.step(action)
                        if viewer is not None:
                            for index, color in enumerate(([0.2, 1.0, 0.2, 0.35], [0.2, 0.6, 1.0, 0.35])):
                                mujoco.mjv_initGeom(
                                    viewer.user_scn.geoms[index], type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=np.full(3, env.arm_settings[index]["success_distance"]),
                                    pos=env.target_positions[index], mat=np.eye(3).ravel(), rgba=np.array(color),
                                )
                            viewer.user_scn.ngeom = 2
                    if not all(np.isfinite(value).all() for value in observation.values()):
                        raise ValueError("Combined evaluation produced a non-finite observation")
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(max(0.0, env.sim_env.control_timestep - (time.perf_counter() - started)))
                    if terminated or truncated:
                        result = {
                            "ur5e_slowdown_radius": ur5e_slowdown_radius,
                            "success": bool(info["is_success"]),
                            "arm_success": info["arm_success"],
                            "distances": info["distances"].tolist(),
                            "max_joint_hold_seconds": info["max_joint_hold_seconds"],
                            "steps": step + 1,
                        }
                        results.append(result)
                        print(
                            f"Episode {episode + 1}/{episodes}: both_success={result['success']}, "
                            f"UR5e_distance={100 * result['distances'][0]:.1f} cm, "
                            f"Jaco_distance={100 * result['distances'][1]:.1f} cm, "
                            f"best_joint_hold={result['max_joint_hold_seconds']:.2f} s",
                            flush=True,
                        )
                        break

        if results:
            successes = sum(result["success"] for result in results)
            print(f"Both-arm successes: {successes}/{len(results)}.", flush=True)
        else:
            print("Viewer closed before an episode finished.", flush=True)
        return results
    finally:
        env.close()
