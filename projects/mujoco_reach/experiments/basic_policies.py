"""Train one arm's reaching policy while the other arm holds still.

From the repository root:
    python -m experiments.basic_policies --check
    python -m experiments.basic_policies --train
    python -m experiments.basic_policies --arm jaco --train
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
    parser.add_argument("--arm", choices=("ur5e", "jaco"), help="Arm to train (default: ur5e, unless --config selects another arm).")
    parser.add_argument("--config", type=Path, help="Override the selected arm's default training config.")
    parser.add_argument("--timesteps", type=int, help="Override the number of training environment steps.")
    parser.add_argument("--device", help="Override the PyTorch device, for example cpu or cuda.")
    parser.add_argument("--output-dir", type=Path, help="Override the selected arm's output folder.")
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
        default_config = "jaco_reach.json" if args.arm == "jaco" else "basic_policies.json"
        config_path = args.config or ROOT / "experiments/configs" / default_config
        settings = json.loads(config_path.read_text())
        if args.timesteps is not None:
            settings["total_timesteps"] = args.timesteps
        if args.device is not None:
            settings["device"] = args.device
        config = PPOConfig(**settings)
        robot_index = config.reach.get("robot_index", 0)
        if args.arm is not None and robot_index != (1 if args.arm == "jaco" else 0):
            raise ValueError("--arm does not match robot_index in the training config")
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"Invalid training config: {error}")

    default_output = "outputs/jaco_reach" if robot_index == 1 else "outputs/basic_policies"
    run(config, train=args.train, output_root=args.output_dir or ROOT / default_output)


if __name__ == "__main__":
    main()
