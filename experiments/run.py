"""Use the same entry point on Mac and Linux, with per-backend environments."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ("demo", "list", "check", "preflight", "train", "evaluate", "baseline")


def command_for(task, arguments, *, platform=sys.platform, current_python=sys.executable):
    if task == "demo":
        executable = ROOT / ".venv-mujoco/bin/python"
        if platform == "darwin" and "--headless" not in arguments:
            executable = executable.with_name("mjpython")
        if not executable.is_file():
            raise FileNotFoundError("Run: python -m experiments.setup_mujoco")
        return [str(executable), "-m", "experiments.mujoco_demo", *arguments]
    managed = ROOT / ".venv-calvin/bin/python"
    executable = str(managed) if managed.is_file() else current_python
    return [executable, "-m", "experiments.skill_composition", task, *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=MODES)
    parser.add_argument("--calvin", action="store_true", help="Explicitly use the previous CALVIN experiment")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.calvin and args.task == "list":
        print("Fixed MuJoCo controller skills (scripted, not neural checkpoints):")
        print("reach, grasp, lift, carry, place, place_on_table, release, retreat")
        return
    if args.task == "train" and not args.calvin:
        parser.exit(1, "The previous training command targets CALVIN. It will not start automatically after the MuJoCo switch.\n"
                       "Run the verified manipulation demo first: python -m experiments.run demo\n"
                       "The MuJoCo learned selector is not connected yet; this demo composes fixed controller skills.\n")
    try:
        task, arguments = args.task, args.arguments
        if not args.calvin:
            if task in ("check", "baseline", "preflight"):
                task, arguments = "demo", ["--headless", "--episodes", "3", *arguments]
            elif task == "evaluate":
                task = "demo"
        command = command_for(task, arguments)
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
