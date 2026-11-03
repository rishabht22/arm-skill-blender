"""Watch or measure a trained policy. Defaults to the latest completed run.

From the repository root:
    mjpython -m experiments.evaluate_policy
    python -m experiments.evaluate_policy --headless --episodes 10
"""

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Saved run directory; defaults to the latest completed run.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes (default: 3).")
    parser.add_argument("--seed", type=int, default=1000, help="Initial placement seed for evaluation.")
    parser.add_argument("--headless", action="store_true", help="Test without a viewer, as fast as possible.")
    parser.add_argument("--device", default="cpu", help="PyTorch inference device (default: cpu).")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    required = ("numpy", "mujoco", "robosuite", "torch", "gymnasium", "stable_baselines3")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.exit(
            1,
            f"Missing policy packages: {', '.join(missing)}.\n"
            "Activate your training environment, then run:\n"
            "  python -m pip install -r requirements-training.txt\n",
        )

    from src.training.evaluation import evaluate, latest_run

    try:
        run_dir = args.run_dir if args.run_dir is not None else latest_run(ROOT / "outputs/basic_policies")
        evaluate(
            run_dir,
            episodes=args.episodes,
            seed=args.seed,
            render=not args.headless,
            device=args.device,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Could not evaluate policy: {error}\n")


if __name__ == "__main__":
    main()
