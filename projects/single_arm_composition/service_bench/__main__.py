"""Run, watch, and record single-arm connector servicing."""

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
from .checks import rejects_reversed_key, retains_locked_sensor
from .environment import KEEPER_OPEN, SCENE, make_bench
from .jobs import JOBS
from .runs import TRIALS, run_job, run_trial
from .skills import SERVICE_SKILLS
from .test_cases import list_tests, show_test


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data/raw/panda_service_composition.h5"
KEEPER_MODES = {"locked": 0., "open": KEEPER_OPEN}


def generate(env, args):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    known = {name: job for name, job in JOBS.items() if not job.held_out}
    held_out = {name: job for name, job in JOBS.items() if job.held_out}
    splits = (
        ("trials", TRIALS, args.trials),
        ("train", known, args.train),
        ("validation", known, args.validation),
        ("test", held_out, args.test),
    )
    with TemporaryDirectory(dir=args.out.parent) as temporary:
        staged = Path(temporary) / args.out.name
        with h5py.File(staged, "x") as data:
            data.attrs.update(format="panda_workshop_v1", experiment="service_bench",
                              base_seed=args.seed, control_hz=env.control_freq,
                              mujoco_version=mujoco.__version__, robosuite_version=robosuite.__version__)
            scene = data.create_group(f"scenes/{SCENE}")
            scene.attrs["entities"] = env.entities
            scene.attrs["fixture_joints"] = env.fixture_joints
            scene.create_dataset("xml", data=env.model.get_xml())
            for split_id, (split, cases, count) in enumerate(splits):
                partition = data.create_group(split)
                for case_id, (name, case) in enumerate(cases.items()):
                    if not count:
                        break
                    group = partition.create_group(name)
                    successes = 0
                    for i in range(count):
                        seed = np.random.SeedSequence([args.seed, split_id, case_id, i]).generate_state(
                            1, dtype=np.uint64
                        )[0]
                        if split == "trials":
                            episode = run_trial(env, case, seed)
                            goals, keeper, installed = None, case.keeper, case.installed
                        else:
                            episode, goals, keeper, installed = run_job(
                                env, case, seed, keeper=(KEEPER_OPEN if i % 2 else 0.)
                            )
                        record = group.create_group(f"{i:05d}")
                        record.attrs.update(start_keeper=keeper, installed=json.dumps(installed))
                        write_episode(record, episode, SCENE, seed, goals)
                        successes += episode.success
                    print(f"{split}/{name} {successes}/{count} passed", flush=True)
        args.out.hardlink_to(staged)
    print(args.out)


def check(env, seed):
    physical = {
        "reversed_key_rejected": rejects_reversed_key(env, seed),
        "locked_sensor_retained": retains_locked_sensor(env, seed),
    }
    for name, passed in physical.items():
        print(f"{name:<27} {'PASS' if passed else 'FAIL'}", flush=True)
    outcomes = list(physical.values())
    for name, trial in TRIALS.items():
        episode = run_trial(env, trial, seed)
        report(name, episode)
        outcomes.append(episode.success)
    for name, job in JOBS.items():
        episode, _, _, _ = run_job(env, job, seed, keeper=0.)
        report(name, episode)
        outcomes.append(episode.success)
    print(f"{sum(outcomes)}/{len(outcomes)} passed")
    return 0 if all(outcomes) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the isolated motions and jobs")
    for name, cases in (("skill", TRIALS), ("task", JOBS)):
        run = commands.add_parser(name)
        run.add_argument("name", choices=cases)
        run.add_argument("--seed", type=int, default=42)
        run.add_argument("--live", action="store_true")
        run.add_argument("--speed", type=float, default=1.)
        if name == "task":
            run.add_argument("--keeper", choices=KEEPER_MODES)
            run.add_argument("--checkpoint", type=Path, help="Run a learned selector in place of the expert")
    test = commands.add_parser("check", help="Check physical constraints, isolated motions, and jobs")
    test.add_argument("--seed", type=int, default=42)
    tests = commands.add_parser("tests", help="List saved test episode IDs")
    tests.add_argument("--dataset", type=Path, default=DATASET)
    compare = commands.add_parser("test", help="Watch the model or replay one recorded test demo")
    compare.add_argument("id", help="For example test_1 or recommission_panel/00000")
    mode = compare.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model", action="store_true", help="Run the learned selector live from the recorded start")
    mode.add_argument("--real-demo", "--real_demo", action="store_true", help="Replay the recorded expert states")
    compare.add_argument("--guarded", action="store_true", help="Focus on one pending goal and filter invalid calls")
    compare.add_argument("--dataset", type=Path, default=DATASET)
    compare.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/service_composition_selector.pt")
    compare.add_argument("--speed", type=float, default=1.)
    compare.add_argument("--headless", action="store_true", help="Print the calls and result without opening a window")
    export = commands.add_parser("generate", help="Record separate training and held-out job demonstrations")
    export.add_argument("--seed", type=int, default=2402)
    export.add_argument("--out", type=Path, default=DATASET)
    for split, count in (("trials", 2), ("train", 6), ("validation", 3), ("test", 4)):
        export.add_argument(f"--{split}", type=int, default=count, help="Attempts per skill or job")
    args = parser.parse_args()

    if args.command == "list":
        for name, skill in SERVICE_SKILLS.items():
            print(f"{name:<26} {skill.__doc__}")
        for name, job in JOBS.items():
            print(f"{name:<26} {'held-out' if job.held_out else 'train/validation'}")
        return 0
    if args.command in ("tests", "test"):
        try:
            if args.command == "tests":
                list_tests(args.dataset)
                return 0
            if args.guarded and not args.model:
                parser.error("--guarded requires --model")
            if not np.isfinite(args.speed) or args.speed <= 0:
                parser.error("--speed must be positive and finite")
            return show_test(args)
        except (OSError, ValueError, KeyError) as exc:
            parser.error(str(exc))
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.command == "generate":
        counts = [args.trials, args.train, args.validation, args.test]
        if min(counts) < 0 or not any(counts):
            parser.error("Counts must be non-negative with at least one episode")
        if args.out.exists():
            parser.error(f"Dataset already exists: {args.out}")
    if args.command in ("skill", "task") and (not np.isfinite(args.speed) or args.speed <= 0):
        parser.error("--speed must be positive and finite")

    with closing(make_bench(args.seed)) as env:
        if args.command == "generate":
            generate(env, args)
            return 0
        if args.command == "check":
            return check(env, args.seed)
        frames = live_frames(env, args.speed) if args.live else nullcontext()
        with frames as on_frame:
            if args.command == "skill":
                episode = run_trial(env, TRIALS[args.name], args.seed, on_frame)
            else:
                selector = None
                if args.checkpoint:
                    from baseline.workshop.train import load_selector

                    selector = load_selector(args.checkpoint, SCENE)
                print(f"Running {'learned selector' if selector is not None else 'scripted expert'} on {args.name}",
                      flush=True)
                episode, _, keeper, installed = run_job(env, JOBS[args.name], args.seed,
                                                         KEEPER_MODES.get(args.keeper), on_frame, selector)
                print(f"Start keeper {keeper:+.2f}, installed {installed}", flush=True)
            report(args.name, episode, verbose=True)
        return 0 if episode.success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ViewerClosed:
        print("Live viewer closed. Test stopped.", flush=True)
        raise SystemExit(130)
