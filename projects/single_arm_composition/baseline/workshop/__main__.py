"""Run and record scripted single-arm skill and composition rollouts."""

import argparse
from contextlib import ExitStack, closing, nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robosuite

from ..individual_skill_generation.skills import Rollout
from ..individual_skill_generation.viewer import ViewerClosed, live_frames, play_clips

from .control import WorkshopArm
from .environment import SCENES, make_workshop
from .examples import EXAMPLES, Example, Step, prepare_example, step
from .primitives import PRIMITIVES
from .tasks import TASKS


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data/raw/panda_workshop.h5"


@dataclass
class Episode:
    reset_state: np.ndarray
    setup_actions: np.ndarray
    calls: list[tuple[Step, Rollout]]
    failure: str = ""

    @property
    def success(self):
        return not self.failure


def start_task(arm):
    reset = arm.env.sim.get_state().flatten()
    opened = step("open_gripper").execute(arm)
    return Episode(reset, opened.actions, [], "" if opened.success else "setup/open_gripper")


def run_case(env, case, seed, on_frame=None):
    env.rng = np.random.default_rng(seed)
    env.reset()
    arm = WorkshopArm(env, on_frame)
    if isinstance(case, Example):
        reset, setup, failure = prepare_example(arm, case)
        episode = Episode(reset, np.asarray(setup).reshape(-1, env.action_dim), [])
        if failure:
            episode.failure = f"setup/{failure}"
        calls = (case.call,)
    else:
        episode = start_task(arm)
        calls = case.plan
    if not episode.success:
        return episode

    for index, call in enumerate(calls):
        if on_frame is not None:
            print(f"  {index + 1:02d} {call.skill} {call.args}", flush=True)
        result = call.execute(arm)
        episode.calls.append((call, result))
        if not result.success:
            episode.failure = f"call/{index}/{call.skill}"
            break
    else:
        if not isinstance(case, Example) and not case.succeeded(arm):
            episode.failure = "final_goal"
    return episode


def report(name, episode, verbose=False):
    if verbose:
        for index, (call, result) in enumerate(episode.calls, 1):
            status = "PASS" if result.success else "FAIL"
            print(f"  {index:02d} {call.skill:<22} {status} ({len(result.actions)} steps)")
    status = "PASS" if episode.success else f"FAIL ({episode.failure})"
    print(f"{name:<25} {status}", flush=True)


def view(env, episode, speed, isolated):
    if not episode.calls:
        return
    if isolated:
        clips = [(call.skill, rollout.states) for call, rollout in episode.calls]
    else:
        states = np.concatenate([result.states if i == 0 else result.states[1:]
                                 for i, (_, result) in enumerate(episode.calls)])
        clips = [("Task rollout", states)]
    play_clips(env, clips, speed, env.control_freq)


def write_episode(group, episode, scene, seed, goals=None):
    group.attrs.update(scene=scene, seed=seed, success=episode.success, failure=episode.failure)
    if goals is not None:
        group.attrs["goal"] = json.dumps(sorted(goals, key=lambda goal: goal["object"]))
    group.create_dataset("reset_state", data=episode.reset_state)
    group.create_dataset("setup_actions", data=episode.setup_actions)
    calls = group.create_group("calls")
    for index, (call, result) in enumerate(episode.calls):
        record = calls.create_group(f"{index:03d}")
        record.attrs.update(skill=call.skill, args=json.dumps(call.args), success=result.success)
        record.create_dataset("actions", data=result.actions, compression="gzip")
        record.create_dataset("states", data=result.states, compression="gzip")
        observations = record.create_group("observations")
        for name, values in result.observations.items():
            observations.create_dataset(name, data=values, compression="gzip")


def generate(args, environments):
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "x") as dataset:
        dataset.attrs.update(format="panda_workshop_v1", base_seed=args.seed, control_hz=20,
                             robosuite_version=robosuite.__version__, mujoco_version=mujoco.__version__)
        models = dataset.create_group("scenes")
        for name, env in environments.items():
            model = models.create_group(name)
            model.attrs["entities"] = env.entities
            model.attrs["fixture_joints"] = np.asarray(env.fixture_joints, dtype=h5py.string_dtype())
            model.create_dataset("xml", data=env.model.get_xml())

        known_tasks = {name: task for name, task in TASKS.items() if not task.held_out}
        held_out_tasks = {name: task for name, task in TASKS.items() if task.held_out}
        partitions = (
            ("trials", EXAMPLES, args.trials_per_skill),
            ("train", known_tasks, args.episodes_per_task),
            ("validation", known_tasks, args.episodes_per_task),
            ("recombination", held_out_tasks, args.episodes_per_task),
        )
        for split_id, (split, cases, count) in enumerate(partitions):
            partition = dataset.create_group(split)
            if not count:
                continue
            for case_id, (name, case) in enumerate(cases.items()):
                trials = partition.create_group(name)
                successes = 0
                for index in range(count):
                    seed = np.random.SeedSequence([args.seed, split_id, case_id, index]).generate_state(1, dtype=np.uint64)[0]
                    episode = run_case(environments[case.scene], case, seed)
                    goals = None if isinstance(case, Example) else case.goals
                    write_episode(trials.create_group(f"{index:05d}"), episode, case.scene, seed, goals)
                    successes += episode.success
                print(f"{split}/{name}: {successes}/{count} passed", flush=True)
    print(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the skills, workcells, and expert tasks")
    for command, cases in (("skill", EXAMPLES), ("task", TASKS)):
        run = commands.add_parser(command, help=f"Run one {command} from a fresh reset")
        run.add_argument("name", choices=cases)
        run.add_argument("--seed", type=int, default=42)
        display = run.add_mutually_exclusive_group()
        display.add_argument("--view", action="store_true", help="Replay the execution in MuJoCo")
        display.add_argument("--live", action="store_true", help="Watch the test run in MuJoCo (mjpython on macOS)")
        run.add_argument("--speed", type=float, default=.65, help="Viewer speed relative to real time")
    check = commands.add_parser("check", help="Run every isolated skill and expert task")
    check.add_argument("--seed", type=int, default=42)
    check.add_argument("--live", action="store_true", help="Watch tests in MuJoCo (mjpython on macOS)")
    check.add_argument("--speed", type=float, default=1.0, help="Viewer speed relative to real time")
    export = commands.add_parser("generate", help="Write an HDF5 dataset of isolated trials and expert episodes")
    export.add_argument("--seed", type=int, default=42, help="Base seed used to derive separate reset seeds")
    export.add_argument("--out", type=Path, default=DEFAULT_DATASET)
    export.add_argument("--trials-per-skill", type=int, default=2)
    export.add_argument("--episodes-per-task", type=int, default=2, help="Episodes per task in each applicable split")
    args = parser.parse_args()

    if args.command == "list":
        for scene in SCENES:
            print(f"\n{scene}")
            for name, example in EXAMPLES.items():
                if example.scene == scene:
                    print(f"  {name:<25} {PRIMITIVES[name].description}")
            for name, task in TASKS.items():
                if task.scene == scene:
                    split = "recombination" if task.held_out else "train/validation"
                    print(f"  task {name:<20} {len(task.plan)} calls, {split}")
        return 0

    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.command in ("skill", "task", "check") and (not np.isfinite(args.speed) or args.speed <= 0):
        parser.error("--speed must be positive and finite")
    if args.command in ("skill", "task"):
        cases = EXAMPLES if args.command == "skill" else TASKS
        case = cases[args.name]
        with closing(make_workshop(case.scene, args.seed)) as env:
            frames = live_frames(env, args.speed) if args.live else nullcontext()
            with frames as on_frame:
                if args.live:
                    print(f"Running {args.name} (seed {args.seed})", flush=True)
                episode = run_case(env, case, args.seed, on_frame)
                report(args.name, episode, verbose=not args.live)
            if args.view:
                view(env, episode, args.speed, isolated=args.command == "skill")
        return 0 if episode.success else 1

    if args.command == "generate" and (
        min(args.trials_per_skill, args.episodes_per_task) < 0
        or not (args.trials_per_skill or args.episodes_per_task)
    ):
        parser.error("Dataset counts must be non-negative with at least one positive count")

    with ExitStack() as stack:
        environments = {name: stack.enter_context(closing(make_workshop(name, args.seed))) for name in SCENES}
        if args.command == "generate":
            try:
                generate(args, environments)
            except FileExistsError:
                parser.error(f"Dataset already exists: {args.out}")
            return 0
        successes = 0
        cases = (*EXAMPLES.items(), *TASKS.items())
        scenes = SCENES if args.live else (None,)
        for scene in scenes:
            frames = live_frames(environments[scene], args.speed) if args.live else nullcontext()
            with frames as on_frame:
                for name, case in cases:
                    if args.live:
                        if case.scene != scene:
                            continue
                        print(f"Running {name} (seed {args.seed})", flush=True)
                    episode = run_case(environments[case.scene], case, args.seed, on_frame)
                    report(name, episode)
                    successes += episode.success
    total = len(EXAMPLES) + len(TASKS)
    print(f"{successes}/{total} passed")
    return 0 if successes == total else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ViewerClosed:
        print("Live viewer closed. Test stopped.", flush=True)
        raise SystemExit(130)
