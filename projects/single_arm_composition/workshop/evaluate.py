"""Evaluate a workshop selector closed-loop on fresh Panda resets."""

import argparse
from collections import Counter
from contextlib import closing, nullcontext
from pathlib import Path

import numpy as np

from individual_skill_generation.viewer import ViewerClosed, live_frames

from .__main__ import run_case, start_task, view
from .control import WorkshopArm
from .environment import make_workshop
from .examples import Step
from .tasks import TASKS
from .train import PROJECT, load_selector


def rollout(env, task, selector, seed, max_calls, on_frame=None):
    env.rng = np.random.default_rng(seed)
    env.reset()
    arm = WorkshopArm(env, on_frame)
    episode = start_task(arm)
    if not episode.success:
        return episode, False

    for index in range(max_calls + 1):
        choice = selector.choose(arm.observe(), task.goals)
        if choice is None:
            if on_frame is not None:
                print("  STOP", flush=True)
            if not task.succeeded(arm):
                episode.failure = "premature STOP"
            break
        if index == max_calls:
            episode.failure = "call budget"
            break
        call = Step(*choice)
        if on_frame is not None:
            print(f"  {index + 1:02d} {call.skill} {call.args}", flush=True)
        result = call.execute(arm)
        episode.calls.append((call, result))
        if not result.success:
            episode.failure = f"failed {call.skill}"
            break
    return episode, task.succeeded(arm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "outputs/workshop_selector.pt")
    parser.add_argument("--task", choices=TASKS, action="append", help="Defaults to all held-out tasks")
    parser.add_argument("--episodes", type=int, default=5, help="Fresh resets per task")
    parser.add_argument("--seed", type=int, default=1001, help="First reset seed")
    parser.add_argument("--max-calls", type=int, default=40)
    parser.add_argument("--trace", action="store_true", help="Print primitive calls with their arguments")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--view", action="store_true", help="Replay a single rollout in MuJoCo")
    display.add_argument("--live", action="store_true", help="Watch the selector run in MuJoCo (mjpython on macOS)")
    parser.add_argument("--speed", type=float, default=.8, help="Viewer speed relative to real time")
    args = parser.parse_args()
    if args.seed < 0 or min(args.episodes, args.max_calls) < 1:
        parser.error("Seed must be non-negative and episodes and max-calls must be positive")
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    tasks = args.task or [name for name, task in TASKS.items() if task.held_out]
    if args.view and (len(tasks) != 1 or args.episodes != 1):
        parser.error("--view requires one --task and --episodes 1")

    selectors = {}
    for name in tasks:
        task = TASKS[name]
        if task.scene not in selectors:
            selectors[task.scene] = load_selector(args.checkpoint, task.scene)
        counts = Counter()
        with (
            closing(make_workshop(task.scene, args.seed)) as env,
            (live_frames(env, args.speed) if args.live else nullcontext()) as on_frame,
        ):
            for seed in range(args.seed, args.seed + args.episodes):
                if args.live:
                    print(f"Running {name} (seed {seed})", flush=True)
                episode, goal = rollout(env, task, selectors[task.scene], seed, args.max_calls, on_frame)
                if args.view:
                    view(env, episode, args.speed, isolated=False)
                expert = run_case(env, task, seed)
                if not np.array_equal(episode.reset_state, expert.reset_state):
                    raise RuntimeError("Selector and expert started from different simulator states")
                counts["selector"] += episode.success
                counts["goal"] += goal
                counts["expert"] += expert.success
                status = "PASS" if episode.success else episode.failure
                print(f"{name} seed={seed} selector={status}, goal={goal}, expert={expert.success},"
                      f" calls={len(episode.calls)}", flush=True)
                if args.trace and not args.live:
                    for i, (call, result) in enumerate(episode.calls, 1):
                        print(f"  {i:02d} {call.skill} {call.args} {'PASS' if result.success else 'FAIL'}", flush=True)
            print(f"{name} | selector {counts['selector']}/{args.episodes} | final goal {counts['goal']}/{args.episodes}"
                  f" | expert {counts['expert']}/{args.episodes}", flush=True)



if __name__ == "__main__":
    try:
        main()
    except ViewerClosed:
        print("Live viewer closed. Evaluation stopped.", flush=True)
        raise SystemExit(130)
