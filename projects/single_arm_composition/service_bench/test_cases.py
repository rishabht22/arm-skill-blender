"""Replay a saved test trace or evaluate a selector from its exact start."""

from contextlib import closing, nullcontext
import json
from time import sleep

import h5py
import mujoco
import numpy as np

from baseline.individual_skill_generation.viewer import live_frames, show_frames, simulation_window
from baseline.workshop.__main__ import report
from .environment import SCENE, make_bench
from .guarded import GuardedSelector
from .jobs import satisfied
from .runs import run_episode, start


def test_records(data):
    if data.attrs.get("experiment") != "service_bench":
        raise ValueError("Expected a service-bench HDF5 dataset")
    split = data["test" if "test" in data else "recombination"]
    for job, group in sorted(split.items()):
        for index, episode in sorted(group.items()):
            yield f"{job}/{index}", episode


def list_tests(path):
    with h5py.File(path, "r") as data:
        for i, (name, episode) in enumerate(test_records(data), 1):
            status = "PASS" if episode.attrs["success"] else "FAIL"
            print(f"test_{i:<3} {name:<32} seed={episode.attrs['seed']} "
                  f"calls={len(episode['calls'])} expert={status}")


def choose_test(data, requested):
    for i, (name, episode) in enumerate(test_records(data), 1):
        alias = f"test_{i}"
        if requested in (alias, name):
            return alias, name, episode
    raise ValueError(f"Unknown test {requested}. Run python -m service_bench tests to list IDs.")


def replay_calls(record, xml, control_hz, speed, headless):
    calls = record["calls"].values()
    if headless:
        for i, call in enumerate(calls, 1):
            print(f"  {i:02d} {call.attrs['skill']} {call.attrs['args']}")
        return
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    interval = 1 / (control_hz * speed)
    with simulation_window(model, data) as window:
        for i, call in enumerate(calls, 1):
            print(f"  {i:02d} {call.attrs['skill']} {call.attrs['args']}", flush=True)
            states = call["states"][:]
            # Each call begins at the previous call's final state.
            if not show_frames(window, model, data, states if i == 1 else states[1:], interval):
                return
        sleep(.6)


def run_saved_model(record, dataset, checkpoint, xml, speed, headless, guarded):
    from baseline.workshop.train import dataset_fingerprint, load_selector

    selector = load_selector(checkpoint, SCENE, dataset_sha256=dataset_fingerprint(dataset))
    seed = int(record.attrs["seed"])
    keeper = float(record.attrs["start_keeper"])
    installed = tuple(json.loads(record.attrs["installed"]))
    goals = json.loads(record.attrs["goal"])
    with closing(make_bench(seed)) as env:
        if env.model.get_xml() != xml:
            raise ValueError("Current scene XML differs from the recorded scene")
        arm, episode = start(env, seed, keeper, installed)
        first_state = next(iter(record["calls"].values()))["states"][0]
        if (not episode.success or not np.array_equal(episode.reset_state, record["reset_state"][:])
                or not np.array_equal(episode.setup_actions, record["setup_actions"][:])
                or not np.array_equal(env.sim.get_state().flatten(), first_state)):
            raise ValueError("Could not reproduce the saved test's starting state")
        print("Verified the same start and gripper setup as the recorded demo.", flush=True)
        if guarded:
            selector = GuardedSelector(selector, arm)
        with (nullcontext() if headless else live_frames(env, speed)) as on_frame:
            arm.on_frame = on_frame
            if on_frame is not None:
                on_frame(first_state)
            run_episode(arm, episode, goals, selector=selector)
            report("Model", episode, verbose=headless)
            for goal in goals:
                print(f"  goal {goal['object']:<12} {'PASS' if satisfied(arm, goal) else 'FAIL'}", flush=True)
            if on_frame is not None:
                sleep(.6)
        print(f"Model calls {len(episode.calls)} | recorded expert calls {len(record['calls'])}", flush=True)
    return 0 if episode.success else 1


def show_test(args):
    with h5py.File(args.dataset, "r") as data:
        alias, name, record = choose_test(data, args.id)
        mode = "Learned selector" if args.model else "Recorded expert demo"
        if args.guarded:
            mode = "Learned selector with goal focus and call filters"
        print(f"{mode} | {alias} | {name} | seed={record.attrs['seed']}", flush=True)
        success = bool(record.attrs["success"])
        status = "PASS" if success else f"FAIL ({record.attrs['failure']})"
        print(f"Recorded expert {status}", flush=True)
        if not len(record["calls"]):
            raise ValueError("No recorded calls in this episode")
        xml = data[f"scenes/{SCENE}/xml"].asstr()[()]
        if args.model:
            return run_saved_model(record, args.dataset, args.checkpoint, xml, args.speed, args.headless,
                                   args.guarded)
        replay_calls(record, xml, int(data.attrs["control_hz"]), args.speed, args.headless)
        return 0
