"""Open the scene and move robot arms up and down.

Run from the repository root:
    mjpython -m experiments.preview
"""

import argparse
import importlib.util
import time
import numpy as np

from src.envs import make_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Seed for initial object placement.")
    args = parser.parse_args()

    missing = [name for name in ("numpy", "mujoco", "robosuite") if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(
            1,
            f"Missing simulation packages: {', '.join(missing)}.\n"
            "Activate your virtual environment and run:\n"
            "  python -m pip install -r requirements.txt\n",
        )

    import mujoco
    import mujoco.viewer

    env = make_env(seed=args.seed)
    try:
        env.reset()
        # robosuite wraps the native MuJoCo model and data. The passive
        # viewer displays them without running its own physics loop.
        model = env.sim.model._model
        data = env.sim.data._data
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                # Draw the robot visuals without overlapping collision meshes.
                viewer.opt.geomgroup[0] = 0
                viewer.opt.geomgroup[1] = 1
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = (0.0, 0.0, 0.9)
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -25

            print("Scene open: UR5e (arm 1), Jaco (arm 2). Close the window to exit.", flush=True)
            step = 0

            while viewer.is_running():
                # Seven commands per arm, fourteen total.
                action = np.zeros(env.action_dim)

                # Reverse direction every 40 steps.
                direction = 1 if (step // 40) % 2 == 0 else -1

                action[2] = 0.9 * direction  # UR5e: move up or down.
                action[9] = 0.6 * direction  # Jaco: move up or down.

                with viewer.lock():
                    _, _, done, _ = env.step(action)

                    step += 1
                    if done:
                        env.reset()
                        step = 0

                viewer.sync()
                time.sleep(1.0 / env.control_freq)
    finally:
        env.close()


if __name__ == "__main__":
    main()
