"""Open the two-arm scene without moving the robots or advancing physics.

Run from the repository root:
    mjpython -m experiments.preview
"""

import argparse
import importlib.util
import time

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
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = (0.0, 0.0, 0.9)
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -25

            print("Scene open. Rotate or zoom the camera; close the window to exit.", flush=True)
            while viewer.is_running():
                # Refresh the display only. No env.step() or mj_step().
                viewer.sync()
                time.sleep(1.0 / 30)
    finally:
        env.close()


if __name__ == "__main__":
    main()
