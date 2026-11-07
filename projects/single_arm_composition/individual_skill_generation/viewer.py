"""Play one successful recording per primitive in a single MuJoCo window."""

import argparse
from pathlib import Path
from time import monotonic, sleep

import h5py
import mujoco
import mujoco.viewer
import numpy as np

from .skill_names import MOTION_SKILLS


DATA_DIR = Path(__file__).resolve().parents[1] / "data/raw"
DEFAULT_DATASET = DATA_DIR / "panda_composition.h5"
STANDALONE_DATASET = DATA_DIR / "panda_semantic_skills.h5"


PRIMITIVE_EXAMPLES = {
    "hover_over": "approach",
    "descend_to_grasp": "descend",
    "raise_by": "lift",
    "lower_to": "lower",
    "move_behind": "move_behind",
    "descend_behind": "descend_behind",
    "push_along": "push",
    "open_gripper": "open_empty",
    "close_gripper": "close_empty",
}


TASK_SKILLS = (*MOTION_SKILLS, "open_gripper", "close_gripper")


def composition_examples(dataset, skills):
    examples = {}
    for task in dataset["train"].values():
        for episode in task.values():
            if episode.attrs["success"]:
                for call in episode["calls"].values():
                    examples.setdefault(call.attrs["skill"], call)
            if all(skill in examples for skill in skills):
                return [(skill, examples[skill]) for skill in skills]
    missing = sorted(set(skills) - examples.keys())
    raise ValueError(f"Missing successful recordings for: {', '.join(missing)}")


def playlist(dataset, selected_skill=None):
    if "train" in dataset:
        yield from composition_examples(dataset, (selected_skill,) if selected_skill else TASK_SKILLS)
        return
    skills = (selected_skill,) if selected_skill else PRIMITIVE_EXAMPLES
    for skill in skills:
        context = PRIMITIVE_EXAMPLES[skill]
        trials = dataset["trials"].get(context)
        recordings = () if trials is None else trials.values()
        trial = next(
            (trial for trial in recordings if trial.attrs["success"] and trial.attrs["skill"] == skill),
            None,
        )
        if trial is not None:
            yield skill, trial
        elif selected_skill:
            raise ValueError(f"No successful {skill} example in the {context} context.")


def show_frames(window, model, data, states, interval):
    deadline = monotonic()
    for state in states:
        if not window.is_running():
            return False
        with window.lock():
            mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, data)
        window.sync()
        deadline += interval
        sleep(max(0, deadline - monotonic()))
    return window.is_running()


def play(trials, speed, model_seed, control_hz):
    from .skills import make_env

    env = make_env(model_seed)
    try:
        model = env.sim.model._model
        data = mujoco.MjData(model)
        interval = 1 / (control_hz * speed)
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as window:
            with window.lock():
                # Match robosuite's visual meshes and hide debug sites.
                window.opt.geomgroup[0] = 0
                window.opt.sitegroup[:] = 0
                window.cam.lookat[:] = [-0.15, 0.0, 0.91]
                window.cam.distance = 1.25
                window.cam.azimuth = 130
                window.cam.elevation = -28
            for index, (skill, trial) in enumerate(trials, 1):
                states = trial["states"][:]
                print(f"[{index}/{len(trials)}] {skill}", flush=True)
                if not show_frames(window, model, data, states[:1], 0.5):
                    return
                if not show_frames(window, model, data, states[1:], interval):
                    return
                sleep(0.6)
            print(f"Played {len(trials)} primitive clip(s).", flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--skill", choices=TASK_SKILLS,
                        help="Play just one skill; defaults to all skills in the dataset")
    parser.add_argument("--speed", type=float, default=0.75, help="Playback speed relative to real time")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    path = args.dataset
    if path is None:
        path = DEFAULT_DATASET if DEFAULT_DATASET.is_file() else STANDALONE_DATASET
    if not path.is_file():
        parser.error(f"Dataset not found: {path}. Run composition.generate first.")
    with h5py.File(path, "r") as dataset:
        try:
            trials = list(playlist(dataset, args.skill))
        except ValueError as exc:
            parser.error(str(exc))
        play(trials, args.speed, int(dataset.attrs["model_seed"]), int(dataset.attrs["control_hz"]))


if __name__ == "__main__":
    main()
