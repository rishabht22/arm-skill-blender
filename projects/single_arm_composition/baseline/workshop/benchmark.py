"""Paired closed-loop tests of composition, binding, and partially completed goals."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from hashlib import file_digest
import json
from math import sqrt
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robosuite
import torch

from .__main__ import start_task
from .control import WorkshopArm
from .environment import make_workshop
from .examples import Step
from .tasks import DRAWER, KEY, LATCH, PEG, TASKS, Task, fixtures, goal_satisfied, sorting
from .train import PROJECT, call_label, load_selector, resolve_call


@dataclass(frozen=True)
class Case:
    name: str
    condition: str
    task: Task
    done: str | None = None
    continuation: tuple[Step, ...] = ()


def make_cases():
    yield from (Case(name, "familiar", TASKS[name])
                for name in ("sort_red_blue", "latch_and_drawer", "fit_peg"))
    yield from (Case(name, "held_out", task) for name, task in TASKS.items() if task.held_out)
    # Each pair has identical objects, order, and call count. One assignment changes.
    pairs = (
        ((("red", "pedestal"), ("green", "buffer")), (("red", "pedestal"), ("green", "mat"))),
        ((("blue", "buffer"), ("red", "mat")), (("blue", "pedestal"), ("red", "mat"))),
        ((("green", "pedestal"), ("red", "mat")), (("green", "pedestal"), ("red", "buffer"))),
    )
    for index, (seen, unseen) in enumerate(pairs, 1):
        yield Case(f"binding_pair_{index}_seen", "new_pair_seen_bindings", sorting(seen))
        yield Case(f"binding_pair_{index}_unseen", "new_pair_unseen_binding", sorting(unseen))

    red_on_pedestal = sorting((("red", "pedestal"),)).plan
    partials = (
        ("sort_red_blue", "red", sorting((("blue", "mat"),)).plan),
        ("sort_red_blue", "blue", red_on_pedestal),
        ("latch_and_drawer", "latch", fixtures((DRAWER,)).plan),
        ("latch_and_drawer", "drawer", fixtures((LATCH,)).plan),
        ("fit_key_and_peg", "key", PEG),
        ("fit_key_and_peg", "peg", KEY),
        ("stack_blue_on_red", "red", TASKS["stack_blue_on_red"].plan[len(red_on_pedestal):]),
    )
    for name, done, continuation in partials:
        yield Case(f"{name}_{done}_done", "partial_start", TASKS[name], done, continuation)


def read_training_plans(path):
    """Read successful training templates and all recorded reset seeds."""
    plans = defaultdict(Counter)
    seeds = set()
    with h5py.File(path, "r") as data:
        if data.attrs.get("format") != "panda_workshop_v1":
            raise ValueError("Expected a panda_workshop_v1 dataset")
        for split in ("train", "validation", "recombination"):
            for name, task in data[split].items():
                for ep in task.values():
                    seeds.add(int(ep.attrs["seed"]))
                    if split != "train" or not ep.attrs["success"]:
                        continue
                    goals = json.loads(ep.attrs["goal"])
                    by_object = {goal["object"]: goal for goal in goals}
                    calls = tuple(call_label(call.attrs["skill"], json.loads(call.attrs["args"]), by_object)
                                  for call in ep["calls"].values())
                    plans[ep.attrs["scene"], name, ep.attrs["goal"]][calls] += 1
    return [
        (scene, name, json.loads(goal), counts.most_common(1)[0][0])
        for (scene, name, goal), counts in sorted(plans.items())
    ], seeds


def verify_cases(cases, templates):
    def signature(goals):
        return json.dumps(sorted(goals, key=lambda goal: goal["object"]), sort_keys=True)

    seen = {(scene, signature(goals)) for scene, _, goals, _ in templates}
    bindings = {(goal["object"], goal.get("support")) for scene, _, goals, _ in templates
                if scene == "tabletop" for goal in goals}
    for case in cases:
        present = (case.task.scene, signature(case.task.goals)) in seen
        if case.condition == "familiar" and not present:
            raise ValueError(f"Missing training goal for familiar case {case.name}")
        if case.condition != "partial_start" and case.condition != "familiar" and present:
            raise ValueError(f"Evaluation goal found in training data: {case.name}")
        if case.condition.startswith("new_pair"):
            missing = sum((goal["object"], goal.get("support")) not in bindings for goal in case.task.goals)
            expected = int(case.condition == "new_pair_unseen_binding")
            if missing != expected:
                raise ValueError(f"Unexpected training binding coverage for {case.name}")


def fixed_plan(task, templates):
    """Select one whole training script by goal overlap and bind its goal arguments once."""
    goals = {goal["object"]: goal for goal in task.goals}

    def overlap(template):
        ref = {goal["object"]: goal for goal in template[2]}
        return (len(ref.keys() & goals.keys()), -len(ref.keys() - goals.keys()),
                sum(goals.get(obj) == goal for obj, goal in ref.items()))

    _, name, _, labels = max((item for item in templates if item[0] == task.scene), key=overlap)
    try:
        return name, tuple(resolve_call(label, goals) for label in labels)
    except KeyError:
        return name, None


def completed(arm, task):
    return [goal["object"] for goal in task.goals if goal_satisfied(arm, goal)]


def reset(env, case, seed):
    env.rng = np.random.default_rng(seed)
    env.reset()
    if case.done is not None:
        goal = next(goal for goal in case.task.goals if goal["object"] == case.done)
        if "joint" in goal:
            env.sim.data.set_joint_qpos(case.done, goal["joint"])
        else:
            env.sim.data.set_joint_qpos(case.done, [*goal["xy"], goal["z"], 1, 0, 0, 0])
        env.sim.forward()
    arm = WorkshopArm(env)
    episode = start_task(arm)
    initial = completed(arm, case.task)
    expected = [case.done] if case.done else []
    if not episode.success or initial != expected:
        raise RuntimeError(f"Invalid start for {case.name} seed {seed}: {initial}, expected {expected}")
    return arm, env.sim.get_state().flatten(), initial


def run_policy(arm, task, choose, initial, budget):
    current = initial
    trace = []
    failure = ""
    for index in range(budget + 1):
        call = choose(index, arm)
        if call is None:
            if len(current) != len(task.goals):
                failure = "premature_stop"
            break
        if index == budget:
            failure = "budget"
            break
        skill, args = call
        result = Step(skill, args).execute(arm)
        current = completed(arm, task)
        trace.append(dict(skill=skill, args=args, success=result.success, completed=current))
        if not result.success:
            failure = f"primitive_failed/{skill}"
            break
    targets = [call["args"].get("obj") or call["args"].get("held") for call in trace]
    first_target = next((target for target in targets if target is not None), None)
    return dict(success=not failure, failure=failure, final_completed=current, first_target=first_target,
                peak_completed=max([len(initial), *(len(call["completed"]) for call in trace)]), trace=trace)


def evaluate_case(case, seeds, checkpoint, templates, budget):
    torch.set_num_threads(1)
    selector = load_selector(checkpoint, case.task.scene)
    expert = case.continuation if case.done else case.task.plan
    expert_calls = [(call.skill, call.args) for call in expert]
    template, fixed = fixed_plan(case.task, templates)
    rows = []
    with closing(make_workshop(case.task.scene, seeds[0])) as env:
        for seed in seeds:
            arm, reference, initial = reset(env, case, seed)
            learned = run_policy(arm, case.task, lambda _, arm: selector.choose(arm.observe(), case.task.goals),
                                 initial, budget)
            arm, state, _ = reset(env, case, seed)
            if not np.array_equal(reference, state):
                raise RuntimeError(f"Unmatched expert reset for {case.name}, seed {seed}")
            scripted = run_policy(arm, case.task, lambda i, _: expert_calls[i] if i < len(expert_calls) else None,
                                  initial, budget)
            if fixed is None:
                replay = dict(success=False, failure="unbound_goal_argument", final_completed=initial,
                              first_target=None, peak_completed=len(initial), trace=[])
            else:
                arm, state, _ = reset(env, case, seed)
                if not np.array_equal(reference, state):
                    raise RuntimeError(f"Unmatched fixed-script reset for {case.name}, seed {seed}")
                replay = run_policy(arm, case.task, lambda i, _: fixed[i] if i < len(fixed) else None,
                                    initial, budget)
            rows.append(dict(seed=seed, initial_completed=initial, selector=learned, fixed=replay, expert=scripted))
    print(f"{case.name}: " + " | ".join(f"{method} {sum(row[method]['success'] for row in rows)}/{len(rows)}"
                                       for method in ("selector", "fixed", "expert")), flush=True)
    return dict(name=case.name, condition=case.condition, goals=case.task.goals, fixed_template=template, episodes=rows)


def wilson(success, n):
    rate = success / n
    z = 1.96
    scale = 1 + z*z/n
    center = (rate + z*z/(2*n)) / scale
    radius = z * sqrt(rate*(1-rate)/n + z*z/(4*n*n)) / scale
    return [max(0., center - radius), min(1., center + radius)]


def summarize(results):
    groups = defaultdict(list)
    for case in results:
        for row in case["episodes"]:
            groups[case["condition"]].append((len(case["goals"]), row))
        n = len(case["episodes"])
        case["summary"] = {}
        for method in ("selector", "fixed", "expert"):
            successes = sum(row[method]["success"] for row in case["episodes"])
            case["summary"][method] = dict(success=successes, n=n, wilson95=wilson(successes, n))

    summary = {}
    for condition, episodes in groups.items():
        counts = {}
        for method in ("selector", "fixed", "expert"):
            failures = Counter(result["failure"] for _, row in episodes if (result := row[method])["failure"])
            # Count only newly satisfied goals; completed-at-reset goals do not contribute.
            progress = [len(set(row[method]["final_completed"]) - set(row["initial_completed"]))
                        / (n - len(row["initial_completed"])) for n, row in episodes]
            counts[method] = dict(success=sum(row[method]["success"] for _, row in episodes),
                                  n=len(episodes), mean_pending_goals_completed=sum(progress) / len(progress),
                                  failures=dict(failures))
            if condition == "partial_start":
                counts[method]["first_target_already_done"] = sum(
                    row[method]["first_target"] in row["initial_completed"] for _, row in episodes
                )
        counts["paired_selector_vs_fixed"] = {
            "selector_only": sum(row["selector"]["success"] and not row["fixed"]["success"] for _, row in episodes),
            "fixed_only": sum(row["fixed"]["success"] and not row["selector"]["success"] for _, row in episodes),
        }
        summary[condition] = counts
    return summary


def main():
    cases = tuple(make_cases())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "outputs/workshop_selector.pt")
    parser.add_argument("--dataset", type=Path, default=PROJECT / "data/raw/panda_workshop.h5")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs/composition_benchmark.json")
    parser.add_argument("--episodes", type=int, default=16, help="Resets per case")
    parser.add_argument("--seed", type=int, default=26001, help="First reset seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case", choices=[case.name for case in cases], action="append")
    args = parser.parse_args()
    if args.seed < 0 or min(args.episodes, args.workers) < 1:
        parser.error("Seed must be non-negative, and counts must be positive")
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}")

    templates, recorded_seeds = read_training_plans(args.dataset)
    seeds = tuple(range(args.seed, args.seed + args.episodes))
    if recorded_seeds.intersection(seeds):
        parser.error("Evaluation seeds overlap with recorded dataset seeds")
    selected = [case for case in cases if not args.case or case.name in args.case]
    try:
        verify_cases(selected, templates)
    except ValueError as exc:
        parser.error(str(exc))
    budget = 40
    settings = (seeds, args.checkpoint, templates, budget)
    if args.workers == 1:
        results = [evaluate_case(case, *settings) for case in selected]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_case, case, *settings) for case in selected]
            results = [future.result() for future in futures]

    with args.checkpoint.open("rb") as file:
        checkpoint_digest = file_digest(file, "sha256").hexdigest()
    with args.dataset.open("rb") as file:
        dataset_digest = file_digest(file, "sha256").hexdigest()
    report = dict(format="panda_composition_benchmark_v1", checkpoint_sha256=checkpoint_digest,
                  dataset_sha256=dataset_digest, mujoco=mujoco.__version__, robosuite=robosuite.__version__,
                  seeds=seeds, call_budget=budget, summary=summarize(results), cases=results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as file:
        json.dump(report, file, indent=2)
    print(args.out)


if __name__ == "__main__":
    main()
