"""Replay a recorded reset, setup, and single skill and check the saved states."""

import argparse

import h5py
import numpy as np

from .skills import make_env


def reset_trial(env, trial):
    seed = int(trial.attrs["seed"])
    env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    env.reset()
    np.testing.assert_allclose(env.sim.get_state().flatten(), trial["reset_state"][:], atol=1e-8)


def replay(env, trial):
    reset_trial(env, trial)
    for action in trial["setup_actions"]:
        env.step(action)

    states = [env.sim.get_state().flatten()]
    for action in trial["actions"]:
        env.step(action)
        states.append(env.sim.get_state().flatten())
    np.testing.assert_allclose(states, trial["states"][:], rtol=1e-6, atol=1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("trials", nargs="+", help="Trial keys such as grasp/00000")
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as dataset:
        env = make_env(int(dataset.attrs["model_seed"]))
        try:
            for key in args.trials:
                replay(env, dataset["trials"][key])
                print(f"{key}: replay matched")
        finally:
            env.close()


if __name__ == "__main__":
    main()
