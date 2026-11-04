"""Use the same entry point on Mac and Linux, with per-backend environments."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODES = ("list", "check", "preflight", "train", "evaluate", "baseline")


def command_for(task, arguments, *, platform=sys.platform, current_python=sys.executable):
    managed = ROOT / ".venv-calvin/bin/python"
    executable = str(managed) if managed.is_file() else current_python
    return [executable, "-m", "experiments.skill_composition", task, *arguments]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=MODES)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        command = command_for(args.task, args.arguments)
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")
    raise SystemExit(subprocess.call(command, cwd=ROOT))


if __name__ == "__main__":
    main()
