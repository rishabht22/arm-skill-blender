"""Collect goal-conditioned composition demos from scripted Panda experts."""

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from individual_skill_generation.skills import PandaSkills, make_env


SPLITS = {
    "train": (100_000, ("lift", "place", "push")),
    "validation": (200_000, ("lift", "place", "push")),
    "test": (300_000, ("lift", "place", "push")),
    "recombination": (400_000, ("push_then_place",)),
}


def task_calls(arm, task, goal, start_mode):
    """Yield one primitive at a time so arguments use live geometry."""
    direction, distance, x, y, lift_height = goal
    if task in ("push", "push_then_place"):
        if start_mode == "open":
            yield arm.close_gripper()
        if start_mode != "behind":
            yield arm.move_behind(direction)
            yield arm.descend_behind(direction)
        yield arm.push_along(direction, distance)
        if task == "push":
            return
        yield arm.raise_by(0.13)
        yield arm.open_gripper()
    elif start_mode == "closed":
        yield arm.open_gripper()

    if not arm.grasped:
        yield arm.hover_over()
        yield arm.descend_to_grasp()
        yield arm.close_gripper(grasp=True)
    yield arm.raise_by(lift_height)
    if task == "lift":
        return
    yield arm.hover_over([x, y])
    yield arm.lower_to(arm.env.table_offset[2] + arm.env.cube.size[2] + 0.05)
    yield arm.open_gripper(release=True)
    yield arm.raise_by(0.12)


def sample_goal(rng, split, task):
    side = rng.choice([-1, 1])
    direction = side if task in ("push", "push_then_place") else 0
    distance = rng.uniform(0.073, 0.10) if direction else 0
    x = y = lift_height = 0.0
    if task in ("place", "push_then_place"):
        # Reserve the farther target band for evaluation.
        x = rng.uniform(-0.06, 0.09)
        band = (0.155, 0.18) if split in ("test", "recombination") else (0.085, 0.15)
        y = side * rng.uniform(*band)
    if task != "push":
        lift_height = rng.uniform(0.10, 0.14)
    return np.array([direction, distance, x, y, lift_height])


def goal_success(arm, task, goal, start, pushed):
    direction, distance, x, y, lift_height = goal
    displacement = arm.object_position - start
    if task == "lift":
        return arm.grasped and displacement[2] >= lift_height - 0.025
    if task == "push":
        return pushed and displacement[1] * direction >= distance - 0.025
    resting_height = arm.env.table_offset[2] + arm.env.cube.size[2]
    return (
        (task == "place" or pushed)
        and not arm.grasped
        and abs(arm.object_position[2] - resting_height) < 0.012
        and np.linalg.norm(arm.object_position[:2] - [x, y]) < 0.035
    )


def record_episode(env, episode, split, task, seed):
    rng = np.random.default_rng(seed)
    env.rng.bit_generator.state = rng.bit_generator.state
    env.reset()
    arm = PandaSkills(env)
    episode.create_dataset("reset_state", data=env.sim.get_state().flatten())
    goal = sample_goal(rng, split, task)
    push_task = task in ("push", "push_then_place")
    start_mode = str(rng.choice(("open", "closed", "behind" if push_task else "grasped")))
    start_call = arm.close_gripper() if start_mode in ("closed", "behind") else arm.open_gripper()
    preparation = [arm.execute(start_call)]
    if preparation[-1].success and start_mode in ("behind", "grasped"):
        if start_mode == "behind":
            setup_calls = (arm.move_behind(goal[0]), arm.descend_behind(goal[0]))
        else:
            setup_calls = (arm.hover_over(), arm.descend_to_grasp())
        for call in setup_calls:
            preparation.append(arm.execute(call))
            if not preparation[-1].success:
                break
        if preparation[-1].success and start_mode == "grasped":
            preparation.append(arm.execute(arm.close_gripper(grasp=True)))
    episode.create_dataset("setup_actions", data=np.concatenate([rollout.actions for rollout in preparation]))
    start = arm.object_position
    episode.attrs.update(task=task, seed=seed, goal=goal, start_mode=start_mode)
    calls = episode.create_group("calls")
    sequence = []
    pushed = False
    success = preparation[-1].success
    if success:
        for index, call in enumerate(task_calls(arm, task, goal, start_mode)):
            rollout = arm.execute(call)
            sequence.append(call.skill)
            entry = calls.create_group(f"{index:02d}")
            entry.attrs.update(skill=call.skill, success=rollout.success,
                               target_position=call.position, target_quaternion_xyzw=call.rotation.as_quat(),
                               grasp=call.grasp, release=call.release, retain=call.retain, min_push=call.min_push)
            entry.create_dataset("states", data=rollout.states)
            entry.create_dataset("actions", data=rollout.actions)
            observations = entry.create_group("observations")
            for name, values in rollout.observations.items():
                observations.create_dataset(name, data=values)
            success = rollout.success
            if not success:
                break
            pushed |= call.skill == "push_along"
    episode.attrs["success"] = success and goal_success(arm, task, goal, start, pushed)
    episode.attrs["sequence"] = json.dumps(sequence)
    return bool(episode.attrs["success"]), tuple(sequence)


def generate(output, train, validation, test, recombination):
    env = make_env(2026)
    try:
        with h5py.File(output, "x") as file:
            file.attrs.update(schema_version=1, model_seed=2026, robot="Panda",
                              environment="Lift", control_hz=20,
                              goal_fields="push_direction,push_distance,destination_x,destination_y,lift_height")
            file.create_dataset("model_xml", data=env.sim.model.get_xml())
            for split, count in zip(SPLITS, (train, validation, test, recombination)):
                base_seed, tasks = SPLITS[split]
                group = file.create_group(split)
                for task_id, task in enumerate(tasks):
                    successes = 0
                    sequences = Counter()
                    for index in range(count):
                        episode = group.create_group(f"{task}/{index:05d}")
                        success, sequence = record_episode(env, episode, split, task, base_seed + task_id * 10_000 + index)
                        if success:
                            successes += 1
                            sequences[sequence] += 1
                    print(f"{split}/{task}: {successes}/{count} successful, {len(sequences)} sequences", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/panda_composition.h5"))
    for name, default in (("train", 60), ("validation", 15), ("test", 20), ("recombination", 20)):
        parser.add_argument(f"--{name}", type=int, default=default, help="Episodes per task")
    args = parser.parse_args()
    counts = (args.train, args.validation, args.test, args.recombination)
    if any(not 1 <= count <= 10_000 for count in counts):
        parser.error("Split counts must be between 1 and 10000.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.out, *counts)


if __name__ == "__main__":
    main()
