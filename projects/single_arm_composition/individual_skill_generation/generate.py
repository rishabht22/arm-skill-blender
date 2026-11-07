"""Generate independent Panda skill trials with seeded resets and replay data."""

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np
import robosuite

from .scenarios import CONTEXTS, SetupFailure, prepare_trial
from .skills import make_env


def generate(output, per_context, seed, contexts):
    env = make_env(seed)
    try:
        with h5py.File(output, "x") as dataset:
            dataset.attrs.update(
                schema_version=1, environment="Lift", robot="Panda",
                controller="OSC_POSE absolute world", control_hz=20, model_seed=seed,
                robosuite_version=robosuite.__version__, mujoco_version=mujoco.__version__,
            )
            dataset.create_dataset("model_xml", data=env.sim.model.get_xml())
            data = dataset.create_group("trials")
            stats = {}
            for context in contexts:
                counts = {"success": 0, "skill_failure": 0, "setup_failure": 0}
                failures = []
                for offset in range(per_context):
                    trial_seed = seed + CONTEXTS.index(context) * 1_000_000 + offset
                    try:
                        trial, call = prepare_trial(env, context, trial_seed)
                    except SetupFailure as exc:
                        counts["setup_failure"] += 1
                        failures.append({"seed": trial_seed, "reason": str(exc)})
                        continue
                    rollout = trial.arm.execute(call)
                    counts["success" if rollout.success else "skill_failure"] += 1
                    group = data.create_group(f"{context}/{offset:05d}")
                    group.attrs.update(
                        skill=call.skill, seed=trial_seed, success=rollout.success,
                        target_position=call.position, target_quaternion_xyzw=call.rotation.as_quat(),
                        grasp=call.grasp, release=call.release, retain=call.retain, min_push=call.min_push,
                        max_steps=call.max_steps,
                    )
                    group.create_dataset("reset_state", data=trial.reset_state)
                    group.create_dataset("setup_actions", data=np.asarray(trial.setup_actions).reshape(-1, env.action_dim))
                    group.create_dataset("actions", data=rollout.actions)
                    group.create_dataset("states", data=rollout.states, compression="gzip")
                    obs = group.create_group("observations")
                    for name, array in rollout.observations.items():
                        obs.create_dataset(name, data=array)
                stats[context] = counts
                data.require_group(context).attrs["setup_failures"] = json.dumps(failures)
                print(context, counts, flush=True)
            dataset.attrs["counts"] = json.dumps(stats)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/panda_semantic_skills.h5"))
    parser.add_argument("--per-context", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--contexts", nargs="+", choices=CONTEXTS, default=CONTEXTS)
    args = parser.parse_args()
    if not 0 < args.per_context <= 1_000_000 or args.seed < 0:
        parser.error("per-context must be 1–1000000 and seed must be nonnegative")
    if len(set(args.contexts)) != len(args.contexts):
        parser.error("contexts must be unique")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.out, args.per_context, args.seed, args.contexts)


if __name__ == "__main__":
    main()
