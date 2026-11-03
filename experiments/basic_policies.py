"""Check or train a joint PPO policy using the robots in src/envs/two_arm_lift.py.

From the repository root:
    python -m experiments.basic_policies --check
    python -m experiments.basic_policies --train
"""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Check the environment and policy without training (default).")
    mode.add_argument("--train", action="store_true", help="Start PPO training and save logs and checkpoints.")
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/configs/basic_policies.json")
    parser.add_argument("--timesteps", type=int, help="Override the number of training environment steps.")
    parser.add_argument("--device", help="Override the PyTorch device, for example cpu or cuda.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/basic_policies")
    args = parser.parse_args()

    required = ("numpy", "mujoco", "robosuite", "torch", "gymnasium", "stable_baselines3")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(
            1,
            f"Missing training packages: {', '.join(missing)}.\n"
            "Activate your Python 3.12 environment, then run:\n"
            "  python -m pip install -r requirements-training.txt\n",
        )

    from src.training.ppo import PPOConfig, run

    try:
        settings = json.loads(args.config.read_text())
        if args.timesteps is not None:
            settings["total_timesteps"] = args.timesteps
        if args.device is not None:
            settings["device"] = args.device
        config = PPOConfig(**settings)
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"Invalid training config: {error}")

    run(config, train=args.train, output_root=args.output_dir)


if __name__ == "__main__":
    main()
