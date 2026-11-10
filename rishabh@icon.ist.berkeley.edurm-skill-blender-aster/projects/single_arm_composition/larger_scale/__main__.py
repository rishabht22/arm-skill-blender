"""Command line entry point. Run from projects/single_arm_composition."""
import argparse
import json

from .protocol import SPLITS, audit, generate, write_json
from .scene import LAYOUTS


TIERS = {
    "pilot": (320, 64, 96, 48, 48, 48, 48),
    "large": (24_000, 2400, 2400, 1200, 1200, 1200, 1200),
}


def parser():
    root = argparse.ArgumentParser(prog="python -m larger_scale")
    sub = root.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="generate and seal a task manifest")
    manifest.add_argument("--out", required=True)
    manifest.add_argument("--tier", choices=TIERS, default="pilot")
    manifest.add_argument("--seed", type=int)
    sub.add_parser("audit", help="verify manifest provenance and partition invariants").add_argument("manifest")

    check = sub.add_parser("check", help="exercise primitives and interlocks")
    check.add_argument("--out", required=True)
    check.add_argument("--layouts", type=int, default=LAYOUTS)
    check.add_argument("--repetitions", type=int, default=2)
    check.add_argument("--workers", type=int, default=1)

    collect = sub.add_parser("collect", help="collect physical expert demonstrations")
    collect.add_argument("--manifest", required=True)
    collect.add_argument("--out", required=True)
    collect.add_argument("--split", choices=("train", "dev"), default="train")
    collect.add_argument("--count", type=int, required=True)
    collect.add_argument("--workers", type=int, default=1)
    collect.add_argument("--batch", type=int, default=64)
    collect.add_argument("--part", type=int, default=0)
    collect.add_argument("--parts", type=int, default=1)

    train = sub.add_parser("train", help="fit a learned selector on a frozen expert corpus")
    train.add_argument("--manifest", required=True)
    train.add_argument("--train-data", required=True)
    train.add_argument("--dev-data", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--episodes", type=int, required=True)
    train.add_argument("--dev-episodes", type=int, default=128)
    train.add_argument("--width", type=int, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch", type=int, default=48)
    train.add_argument("--threads", type=int, default=1)
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")

    evaluate = sub.add_parser("evaluate", help="run paired closed-loop comparisons")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--split", choices=SPLITS[1:], default="dev")
    evaluate.add_argument("--count", type=int, required=True)
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--model", action="append", default=[])
    evaluate.add_argument("--lock")
    evaluate.add_argument("--part", type=int, default=0)
    evaluate.add_argument("--parts", type=int, default=1)

    lock = sub.add_parser("lock", help="lock checkpoints and evaluation counts before opening test")
    lock.add_argument("--manifest", required=True)
    lock.add_argument("--out", required=True)
    lock.add_argument("--model", action="append", required=True)
    lock.add_argument("--counts", nargs=5, type=int, required=True, metavar="COUNT")

    report = sub.add_parser("report", help="verify paired records and compute uncertainty")
    report.add_argument("--out", required=True)
    report.add_argument("--partial", action="store_true",
                        help="summarize only completed matched starts and mark incomplete slices")
    report.add_argument("rollouts", nargs="+")

    status = sub.add_parser("corpus-status", help="summarize durable physical episode shards")
    status.add_argument("corpus")

    view = sub.add_parser("view", help="inspect a live learned run or replay recorded expert controls")
    view.add_argument("--manifest", required=True)
    view.add_argument("--task", required=True)
    view.add_argument("--split", choices=SPLITS[1:], required=True)
    view.add_argument("--reset", type=int, default=0)
    view.add_argument("--lock")
    view.add_argument("--recordings", required=True)
    mode = view.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model")
    mode.add_argument("--real-demo", action="store_true")
    mode.add_argument("--record-demo", action="store_true")
    view.add_argument("--show", action="store_true")
    view.add_argument("--video")
    view.add_argument("--speed", type=float, default=2.)
    return root


def main():
    args = parser().parse_args()
    if args.command == "manifest":
        seed = args.seed if args.seed is not None else {"pilot": 748732, "large": 9076842}[args.tier]
        result = generate(args.out, dict(zip(SPLITS, TIERS[args.tier])), seed,
                          3 if args.tier == "large" else 2)
    elif args.command == "audit":
        result = audit(args.manifest)
    elif args.command == "check":
        from .checks import validate
        result = validate(args.layouts, args.repetitions, args.workers)
        write_json(args.out, result)
        if result["failures"] or not result["all_primitives_exercised"] or not all(
                row["passed"] for row in result["counterexamples"]):
            raise SystemExit(f"physical checks failed; see {args.out}")
    elif args.command == "collect":
        from .collect import collect
        result = collect(args.manifest, args.out, args.split, args.count, args.workers,
                         args.batch, args.part, args.parts)
    elif args.command == "train":
        from .learn import train
        result = train(args)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        result = evaluate(args)
    elif args.command == "lock":
        from .evaluate import lock
        result = lock(args.manifest, args.model, args.counts, args.out)
    elif args.command == "report":
        from .evaluate import report
        result = report(args.rollouts, partial=args.partial)
        write_json(args.out, result)
    elif args.command == "corpus-status":
        from .collect import corpus_status
        result = corpus_status(args.corpus)
    else:
        from .view import view
        result = view(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
