"""One-time machine setup; run the same command separately on Mac and Linux."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    if sys.platform not in ("darwin", "linux"):
        parser.error("This setup supports macOS and Linux")
    conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
    if not conda:
        parser.error("Conda is required for the project's Python 3.10 environment")
    prefix = ROOT / ".venv-calvin"
    python = prefix / "bin/python"
    if not python.is_file():
        environment = os.environ.copy()
        # A project-local prefix does not need a home-directory environment entry.
        environment["CONDA_REGISTER_ENVS"] = "false"
        subprocess.run([
            conda, "create", "--copy", "--override-channels", "-c", "conda-forge", "--prefix", str(prefix),
            "python=3.10", "pip", "pybullet=3.25", "-y",
        ], check=True, cwd=ROOT, env=environment)
    env = os.environ.copy()
    env["PATH"] = str(prefix / "bin") + os.pathsep + env["PATH"]
    for phase in ("sources", "install", "model", "metadata", "language"):
        subprocess.run(["bash", "experiments/setup_calvin.sh", phase], check=True, cwd=ROOT, env=env)
    print("Ready: python -m experiments.run check")


if __name__ == "__main__":
    main()
