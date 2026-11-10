"""Run individual skills or a complete job with the packing-cell controllers."""

from dataclasses import dataclass

import numpy as np

from baseline.workshop.__main__ import start_task
from baseline.workshop.control import WorkshopArm
from baseline.workshop.examples import Step, grasp, lift, step
from .environment import BOLT_UP, DRAWER_OPEN, SCENE
from .skills import execute
from .tasks import job_start, plan_job, reset_cell, satisfied


@dataclass(frozen=True)
class Trial:
    mode: str
    stocked: tuple
    setup: tuple
    call: Step
    scene: str = SCENE


RED_HELD = lift("red", height=.16)
CENTER = step("center_over_slot", obj="red", slot="slot_a")
TRIALS = {
    "set_bolt": Trial("locked", (), grasp("bolt"), step("set_bolt", obj="bolt", target=BOLT_UP)),
    "slide_drawer": Trial("unlocked", (("red", "slot_a"),), grasp("drawer"),
                          step("slide_drawer", obj="drawer", target=DRAWER_OPEN)),
    "center_over_slot": Trial("open", (), RED_HELD, CENTER),
    "seat_in_slot": Trial("open", (), (*RED_HELD, CENTER), step("seat_in_slot", obj="red", slot="slot_a")),
    "lift_from_slot": Trial("open", (("red", "slot_a"),), grasp("red"),
                            step("lift_from_slot", obj="red", slot="slot_a")),
}


def setup_run(env, seed, mode, stocked, on_frame):
    reset_cell(env, seed, mode, stocked)
    arm = WorkshopArm(env, on_frame)
    return arm, start_task(arm)


def run_trial(env, trial, seed, on_frame=None):
    arm, episode = setup_run(env, seed, trial.mode, trial.stocked, on_frame)
    for call in trial.setup:
        if not episode.success:
            return episode
        result = execute(arm, call)
        episode.setup_actions = np.concatenate((episode.setup_actions, result.actions))
        if not result.success:
            episode.failure = f"setup/{call.skill}"
    if episode.success:
        result = execute(arm, trial.call)
        episode.calls.append((trial.call, result))
        if not result.success:
            episode.failure = f"call/{trial.call.skill}"
    return episode


def run_job(env, job, seed, mode=None, on_frame=None, selector=None):
    rng = np.random.default_rng(seed)
    mode, stocked = job_start(job, rng, mode)
    arm, episode = setup_run(env, seed, mode, stocked, on_frame)
    goals = job.goals(arm)
    if not episode.success:
        return episode, goals, mode, stocked

    def selected_calls():
        for index in range(51):
            chosen = selector.choose(arm.observe(), goals)
            if chosen is None:
                if on_frame:
                    print("  STOP", flush=True)
                return
            if index == 50:
                episode.failure = "call_budget"
                return
            yield Step(*chosen)

    calls = plan_job(arm, goals, rng) if selector is None else selected_calls()
    for index, call in enumerate(calls):
        if on_frame:
            print(f"  {index + 1:02d} {call.skill} {call.args}", flush=True)
        result = execute(arm, call)
        episode.calls.append((call, result))
        if not result.success:
            episode.failure = f"call/{index}/{call.skill}"
            break
    if episode.success and not all(satisfied(arm, goal) for goal in goals):
        episode.failure = "final_goal" if selector is None else "premature_STOP"
    return episode, goals, mode, stocked
