"""Launch the scripted MuJoCo manipulation demo."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command_for(task, arguments, *, platform=sys.platform, current_python=sys.executable):
    if task in ("check", "baseline", "preflight"):
        arguments = ["--headless", "--episodes", "3", *arguments]
    managed = ROOT / ".venv-mujoco/bin/python"
    executable = managed if managed.is_file() else Path(current_python)
    if platform == "darwin" and not {"--headless", "--help", "-h"}.intersection(arguments):
        executable = executable.with_name("mjpython")
        if not executable.is_file():
            raise FileNotFoundError("Run: python -m experiments.setup_mujoco")
    return [str(executable), "-m", "experiments.mujoco_demo", *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("demo", "list", "check", "preflight", "evaluate", "baseline"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.task == "list":
        print("reach, grasp, lift, carry, place, place_on_table, release, retreat")
        return
    try:
        command = command_for(args.task, args.arguments)
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
