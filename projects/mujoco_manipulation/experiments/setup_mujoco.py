"""Create the independent MuJoCo demo runtime on Mac or Linux."""

import argparse
from pathlib import Path
import subprocess
import sys
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Use Python 3.10 or newer for this setup")
    python = ROOT / ".venv-mujoco/bin/python"
    if not python.is_file():
        if sys.version_info < (3, 13):
            subprocess.run([sys.executable, "-m", "venv", "--copies", str(python.parents[1])], check=True)
        else:
            conda = shutil.which("conda") or os.environ.get("CONDA_EXE")
            if not conda:
                parser.error("Run setup using Python 3.10–3.12, or install Conda so setup can create Python 3.12")
            env = os.environ.copy()
            env["CONDA_REGISTER_ENVS"] = "false"
            subprocess.run([conda, "create", "--copy", "--override-channels", "-c", "conda-forge", "-p",
                            str(python.parents[1]), "python=3.12", "-y"], check=True, env=env)
    subprocess.run([str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements-mujoco.txt")], check=True, cwd=ROOT)
    print("Ready: python -m experiments.run demo")


if __name__ == "__main__":
    main()
