"""Run and record the interlocked single-Panda packing cell."""

import argparse
from contextlib import closing, nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import mujoco
import numpy as np
import robosuite

from baseline.individual_skill_generation.viewer import ViewerClosed, live_frames
from baseline.workshop.__main__ import report, write_episode
from baseline.workshop.examples import grasp, step
from .cell.environment import DRAWER_OPEN, SCENE, make_cell
from .cell.runs import TRIALS, run_job, run_trial, setup_run
from .cell.skills import CELL_SKILLS, execute
from .cell.tasks import JOBS, MODES


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/raw/panda_packing.h5"


def interlock_check(env, seed):
    arm, episode = setup_run(env, seed, "locked", (), None)
    if not episode.success:
        return False
    for call in grasp("drawer"):
        if not execute(arm, call).success:
            return False
    result = execute(arm, step("slide_drawer", obj="drawer", target=DRAWER_OPEN))
    drawer_index = env.fixture_joints.index("drawer")
    return not result.success and float(result.observations["fixture_position"][:, drawer_index].max()) < .04


def generate(env, args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    known = {name: job for name, job in JOBS.items() if not job.held_out}
    held_out = {name: job for name, job in JOBS.items() if job.held_out}
    partitions = (
        ("trials", TRIALS, args.trials),
        ("train", known, args.train),
        ("validation", known, args.validation),
        ("recombination", held_out, args.recombination),
    )
    with TemporaryDirectory(dir=args.out.parent) as directory:
        path = Path(directory) / args.out.name
        with h5py.File(path, "x") as dataset:
            dataset.attrs.update(format="panda_workshop_v1", experiment="packing_cell",
                                 base_seed=args.seed, control_hz=env.control_freq,
                                 mujoco_version=mujoco.__version__, robosuite_version=robosuite.__version__)
            scene = dataset.create_group(f"scenes/{SCENE}")
            scene.attrs["entities"] = env.entities
            scene.attrs["fixture_joints"] = env.fixture_joints
            scene.create_dataset("xml", data=env.model.get_xml())
            for split_id, (split, cases, count) in enumerate(partitions):
                partition = dataset.create_group(split)
                if not count:
                    continue
                for case_id, (name, case) in enumerate(cases.items()):
                    group = partition.create_group(name)
                    successes = 0
                    for index in range(count):
                        seed = np.random.SeedSequence([args.seed, split_id, case_id, index]).generate_state(
                            1, dtype=np.uint64
                        )[0]
                        if split == "trials":
                            episode = run_trial(env, case, seed)
                            goals, mode, stocked = None, case.mode, case.stocked
                        else:
                            mode = tuple(MODES)[index % len(MODES)]
                            episode, goals, mode, stocked = run_job(env, case, seed, mode)
                        record = group.create_group(f"{index:05d}")
                        record.attrs.update(start_mode=mode, stocked=json.dumps(stocked))
                        write_episode(record, episode, SCENE, seed, goals)
                        successes += episode.success
                    print(f"{split}/{name} {successes}/{count} passed", flush=True)
        args.out.hardlink_to(path)
    print(args.out)


def evaluate(env, args, on_frame):
    from baseline.workshop.train import load_selector

    selector = load_selector(args.checkpoint, SCENE)
    names = args.task or [name for name, job in JOBS.items() if job.held_out]
    for name in names:
        job = JOBS[name]
        learned_total = expert_total = 0
        for seed in range(args.seed, args.seed + args.episodes):
            if args.live:
                print(f"Running learned selector on {name} (seed {seed})", flush=True)
            learned, goals, mode, stocked = run_job(env, job, seed, on_frame=on_frame, selector=selector)
            expert, reference_goals, reference_mode, reference_stock = run_job(env, job, seed)
            if ((goals, mode, stocked) != (reference_goals, reference_mode, reference_stock)
                    or not np.array_equal(learned.reset_state, expert.reset_state)):
                raise RuntimeError(f"Selector and expert received different starts for {name}, seed {seed}")
            learned_total += learned.success
            expert_total += expert.success
            print(f"{name} seed={seed} mode={mode} selector={learned.failure or 'PASS'} "
                  f"expert={expert.failure or 'PASS'}", flush=True)
        print(f"{name} | selector {learned_total}/{args.episodes} | expert {expert_total}/{args.episodes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the new primitives and jobs")
    for command, choices in (("skill", TRIALS), ("task", JOBS)):
        run = commands.add_parser(command)
        run.add_argument("name", choices=choices)
        run.add_argument("--seed", type=int, default=42)
        run.add_argument("--live", action="store_true")
        run.add_argument("--speed", type=float, default=1)
        if command == "task":
            run.add_argument("--mode", choices=MODES)
            run.add_argument("--checkpoint", type=Path, help="Run the trained selector in place of the expert")
    check = commands.add_parser("check", help="Run the interlock, isolated motions, and job checks")
    check.add_argument("--seed", type=int, default=42)
    export = commands.add_parser("generate", help="Generate separate HDF5 training and evaluation demos")
    export.add_argument("--seed", type=int, default=2026)
    export.add_argument("--out", type=Path, default=DATASET)
    for split, count in (("trials", 2), ("train", 6), ("validation", 3), ("recombination", 3)):
        export.add_argument(f"--{split}", type=int, default=count, help="Episodes per skill or job")
    compare = commands.add_parser("evaluate", help="Compare the selector and expert on identical fresh starts")
    compare.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/packing_selector.pt")
    compare.add_argument("--live", action="store_true", help="Watch the learned selector in MuJoCo")
    compare.add_argument("--speed", type=float, default=1, help="Viewer speed relative to real time")
    compare.add_argument("--task", choices=JOBS, action="append")
    compare.add_argument("--episodes", type=int, default=5)
    compare.add_argument("--seed", type=int, default=21001)
    args = parser.parse_args()

    if args.command == "list":
        for name, build in CELL_SKILLS.items():
            print(f"{name:<25} {build.__doc__}")
        for name, job in JOBS.items():
            print(f"{name:<25} {'held-out' if job.held_out else 'train/validation'}")
        return 0

    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.command == "evaluate" and args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.command == "generate":
        counts = [args.trials, args.train, args.validation, args.recombination]
        if min(counts) < 0 or not any(counts):
            parser.error("Counts must be non-negative with at least one episode")
        if args.out.exists():
            parser.error(f"Dataset already exists: {args.out}")
    if args.command in ("skill", "task", "evaluate") and (not np.isfinite(args.speed) or args.speed <= 0):
        parser.error("--speed must be positive and finite")

    with closing(make_cell(args.seed)) as env:
        if args.command == "evaluate":
            frames = live_frames(env, args.speed) if args.live else nullcontext()
            with frames as on_frame:
                evaluate(env, args, on_frame)
            return 0
        if args.command == "generate":
            generate(env, args)
            return 0
        if args.command == "check":
            blocked = interlock_check(env, args.seed)
            print(f"locked_drawer_interlock {'PASS' if blocked else 'FAIL'}", flush=True)
            successes = int(blocked)
            for name, trial in TRIALS.items():
                episode = run_trial(env, trial, args.seed)
                report(name, episode)
                successes += episode.success
            for name, job in JOBS.items():
                episode, _, _, _ = run_job(env, job, args.seed, mode="locked")
                report(name, episode)
                successes += episode.success
            total = 1 + len(TRIALS) + len(JOBS)
            print(f"{successes}/{total} passed")
            return 0 if successes == total else 1
        frames = live_frames(env, args.speed) if args.live else nullcontext()
        with frames as on_frame:
            if args.command == "skill":
                episode = run_trial(env, TRIALS[args.name], args.seed, on_frame)
            else:
                selector = None
                if args.checkpoint:
                    from baseline.workshop.train import load_selector
                    selector = load_selector(args.checkpoint, SCENE)
                episode, _, mode, stocked = run_job(env, JOBS[args.name], args.seed, args.mode, on_frame, selector)
                print(f"Start mode {mode}, stocked {stocked}", flush=True)
            report(args.name, episode, verbose=True)
        return 0 if episode.success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ViewerClosed:
        print("Live viewer closed. Test stopped.", flush=True)
        raise SystemExit(130)
