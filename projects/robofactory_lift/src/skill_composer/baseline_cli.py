import argparse
from pathlib import Path

from .common import PROCESSED, RAW, RUN


def main():
    parser = argparse.ArgumentParser(description="RoboFactory-data skill composition baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download published LiftBarrier demonstrations (~988 MB)")
    download.add_argument("--raw", type=Path, default=RAW)
    prepare = commands.add_parser("prepare", help="Validate HDF5, weakly label skills, split by episode")
    prepare.add_argument("--raw", type=Path, default=RAW)
    prepare.add_argument("--output", type=Path, default=PROCESSED)
    prepare.add_argument("--seed", type=int, default=7)
    train = commands.add_parser("train", help="Behavior clone the coordination head only")
    train.add_argument("--data", type=Path, default=PROCESSED)
    train.add_argument("--run", type=Path, default=RUN)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--seed", type=int, default=7)
    evaluate = commands.add_parser("evaluate", help="Offline held-out label imitation metrics")
    evaluate.add_argument("--data", type=Path, default=PROCESSED)
    evaluate.add_argument("--run", type=Path, default=RUN)
    preview = commands.add_parser("preview", help="Annotate actual held-out demo frames")
    preview.add_argument("--raw", type=Path, default=RAW)
    preview.add_argument("--run", type=Path, default=RUN)
    preview.add_argument("--episode", type=int)
    rollout = commands.add_parser("rollout", help="Run learned head in the separate MuJoCo proxy sandbox")
    rollout.add_argument("--run", type=Path, default=RUN)
    rollout.add_argument("--steps", type=int, default=160)
    rollout.add_argument("--viewer", action="store_true")
    rollout.add_argument("--gif", type=Path)
    rollout.add_argument("--no-guard", action="store_true")
    rollout.add_argument("--oracle", action="store_true", help="Diagnostic scripted scheduler instead of learned head")
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--randomization", type=float, default=0.0, help="Object XY offset in meters (transfer stress test)")
    skills = commands.add_parser("skills", help="Inspect one arm executing its frozen skills with the other parked")
    skills.add_argument("--run", type=Path, default=RUN)
    skills.add_argument("--arm", type=int, choices=[0, 1], default=0)
    skills.add_argument("--steps", type=int, default=100)
    skills.add_argument("--viewer", action="store_true")
    skills.add_argument("--gif", type=Path)
    comparison = commands.add_parser("compare", help="Compare individual skills, independent queues, learned head, and fixed coordinator")
    comparison.add_argument("--run", type=Path, default=RUN)
    comparison.add_argument("--steps", type=int, default=120)
    comparison.add_argument("--gif", type=Path)
    comparison.add_argument("--output", type=Path, help="Results directory; defaults to RUN/comparison")
    comparison.add_argument("--seed", type=int, default=0)
    comparison.add_argument("--delay-arm1", type=float, default=0., help="Delay arm 1 motor commands by this many seconds, identically in every trial")
    args = parser.parse_args()
    if args.command == "download":
        from .download import download
        download(args.raw)
    elif args.command == "prepare":
        from .data import prepare
        prepare(args.raw, args.output, args.seed)
    elif args.command == "train":
        from .train import train
        train(args.data, args.run, args.epochs, args.seed)
    elif args.command == "evaluate":
        from .train import evaluate
        evaluate(args.data, args.run)
    elif args.command == "preview":
        from .preview import preview
        preview(args.run, args.raw, args.episode)
    elif args.command == "rollout":
        from .execution import run_episode
        scenario = ("fixed_guarded" if args.oracle else "head_guarded")
        if args.no_guard:
            scenario = "independent" if args.oracle else "head"
        run_episode(args.run, scenario, args.steps, args.viewer, args.gif, args.seed, args.randomization)
    elif args.command == "skills":
        from .execution import run_episode
        run_episode(args.run, f"arm{args.arm}", args.steps, args.viewer, args.gif)
    elif args.command == "compare":
        from .comparison import compare
        compare(args.run, args.steps, args.gif, args.seed, args.delay_arm1, args.output)


if __name__ == "__main__":
    main()
