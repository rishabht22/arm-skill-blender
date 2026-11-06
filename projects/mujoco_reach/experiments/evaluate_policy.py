"""Watch both independent reaching policies together, or test one arm.

From the repository root:
    mjpython -m experiments.evaluate_policy
    mjpython -m experiments.evaluate_policy --arm jaco
    python -m experiments.evaluate_policy --headless --episodes 10
"""

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Evaluate a single saved policy; uses its saved arm.")
    parser.add_argument("--arm", choices=("both", "ur5e", "jaco"), help="Arms to evaluate (default: both).")
    parser.add_argument("--ur5e-run-dir", type=Path, help="Override the UR5e checkpoint for combined evaluation.")
    parser.add_argument("--jaco-run-dir", type=Path, help="Override the Jaco checkpoint for combined evaluation.")
    parser.add_argument("--episodes", type=int, default=3, help="Number of test episodes (default: 3).")
    parser.add_argument("--seed", type=int, default=1000, help="Initial placement seed for evaluation.")
    parser.add_argument("--headless", action="store_true", help="Test without a viewer, as fast as possible.")
    parser.add_argument("--device", default="cpu", help="PyTorch inference device (default: cpu).")
    parser.add_argument(
        "--ur5e-slowdown-radius", type=float, default=0.08,
        help="Slow UR5e near its goal within this radius in meters (default: 0.08; 0 disables).",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    arm = args.arm or ("ur5e" if args.run_dir is not None else "both")
    if arm == "both" and args.run_dir is not None:
        parser.error("Use --ur5e-run-dir and --jaco-run-dir when evaluating both arms")
    if arm != "both" and (args.ur5e_run_dir is not None or args.jaco_run_dir is not None):
        parser.error("--ur5e-run-dir and --jaco-run-dir are for combined evaluation")

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
        if arm == "both":
            from src.training.combined_evaluation import evaluate_both

            ur5e_run = args.ur5e_run_dir or latest_run(ROOT / "outputs/basic_policies", robot_index=0)
            jaco_run = args.jaco_run_dir or latest_run(ROOT / "outputs/jaco_reach", robot_index=1)
            evaluate_both(
                ur5e_run, jaco_run, episodes=args.episodes, seed=args.seed,
                render=not args.headless, device=args.device,
                ur5e_slowdown_radius=args.ur5e_slowdown_radius,
            )
            return
        robot_index = 1 if arm == "jaco" else 0
        output_folder = "outputs/jaco_reach" if robot_index == 1 else "outputs/basic_policies"
        run_dir = args.run_dir if args.run_dir is not None else latest_run(
            ROOT / output_folder, robot_index=robot_index
        )
        evaluate(
            run_dir,
            episodes=args.episodes,
            seed=args.seed,
            render=not args.headless,
            device=args.device,
            ur5e_slowdown_radius=args.ur5e_slowdown_radius,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Could not evaluate policy: {error}\n")


if __name__ == "__main__":
    main()
