"""Train, benchmark, and view outcome-conditioned skill composition."""

import argparse
from contextlib import ExitStack
import json

from larger_scale.scene import CARRIERS

from .tasks import SUITES


def show(args):
    from larger_scale.view import frames
    from larger_scale.workcell import Workcell

    from .learn import load
    from .planner import ACTIONS
    from .sim import describe, rollout, symbolic
    from .tasks import showcase, tasks

    if args.speed <= 0:
        raise ValueError("speed must be positive")
    if args.goal:
        from .goal_file import read_goal
        task = read_goal(args.goal)
    elif args.suite:
        task = list(tasks(args.suite, args.index + 1))[-1]
    else:
        task = showcase(args.carrier, args.start == "loaded")
    choose = load(args.model) if args.model else symbolic
    cell = Workcell(task.layout)
    goal_text = describe(task.goal)
    print(f"Goal — {goal_text}", flush=True)
    decision = ""

    def trace(goal, state, choice):
        nonlocal decision
        kind, job = ACTIONS[choice]
        line = f"Choose {kind} for carrier {chr(65 + job)}"
        if args.model:
            scores = choose.scores(goal, state)
            top = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:3]
            labels = [f"{ACTIONS[index][0]} {chr(65 + ACTIONS[index][1])} ({scores[index]:.2f})"
                      for index in top]
            line += f" | top scores {' / '.join(labels)}"
        decision = line
        print(line, flush=True)

    with ExitStack() as stack:
        video = None
        if args.video:
            from .video import video_frames
            video = stack.enter_context(video_frames(cell, args.video, args.speed, goal_text))
        draw = stack.enter_context(frames(argparse.Namespace(
            show=True, video=None, speed=args.speed), cell)) if args.show else None
        if draw or video:
            advance = cell.advance

            def rendered(ticks, *, panel=False):
                advance(ticks, panel=panel)
                if decision:
                    if draw:
                        draw(cell)
                    if video:
                        video(decision)

            cell.advance = rendered
        result = rollout(task, choose, cell=cell, before=trace, reference=False)
        if video:
            final = "Goal reached" if result["success"] else f"Stopped | {result['reason']}"
            video(final, hold=30)
    print(json.dumps(result, indent=2))
    if not result["success"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--out", required=True)
    fit.add_argument("--problems", type=int, default=1800)
    fit.add_argument("--dev-problems", type=int, default=96)
    fit.add_argument("--physical", type=int, default=0)
    fit.add_argument("--epochs", type=int, default=28)
    fit.add_argument("--batch", type=int, default=192)
    fit.add_argument("--width", type=int, default=64)
    fit.add_argument("--seed", type=int, default=17)
    fit.add_argument("--threads", type=int, default=2)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    bench = sub.add_parser("evaluate")
    bench.add_argument("--model", action="append", required=True)
    bench.add_argument("--out", required=True)
    bench.add_argument("--suite", choices=("engineering", *SUITES), required=True)
    bench.add_argument("--count", type=int, required=True)
    bench.add_argument("--workers", type=int, default=1)
    bench.add_argument("--batch", type=int, default=8)
    bench.add_argument("--part", type=int, default=0)
    bench.add_argument("--parts", type=int, default=1)

    summary = sub.add_parser("report")
    summary.add_argument("--out", required=True)
    summary.add_argument("rollouts", nargs="+")

    view = sub.add_parser("show")
    view.add_argument("--model")
    view.add_argument("--start", choices=("feed", "loaded"), default="feed")
    view.add_argument("--carrier", type=int, choices=range(CARRIERS), default=1)
    goal_input = view.add_mutually_exclusive_group()
    goal_input.add_argument("--suite", choices=SUITES)
    goal_input.add_argument("--goal", help="JSON file specifying desired outcomes")
    view.add_argument("--index", type=int, default=0)
    view.add_argument("--show", action="store_true")
    view.add_argument("--video")
    view.add_argument("--speed", type=float, default=4.)

    args = parser.parse_args()
    if args.command == "train":
        from .learn import train
        train(args)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(args)
    elif args.command == "report":
        from .evaluate import report
        report(args.rollouts, args.out)
    else:
        if args.index < 0:
            parser.error("--index must be nonnegative")
        show(args)


if __name__ == "__main__":
    main()
