"""Live and headless runs for the service bench."""

from dataclasses import dataclass

import numpy as np

from baseline.workshop.__main__ import start_task
from baseline.workshop.control import WorkshopArm
from baseline.workshop.examples import Step, grasp, lift, step
from .environment import KEEPER_OPEN
from .guarded import NoFeasibleCall
from .jobs import PORT, RAIL, grasp_keeper, grasp_sensor, plan_job, reset_bench, satisfied
from .skills import execute


@dataclass(frozen=True)
class Trial:
    setup: tuple[Step, ...]
    call: Step
    installed: tuple[str, ...] = ()
    keeper: float = KEEPER_OPEN


SENSOR_HELD = (*grasp_sensor("sensor"), step("raise_by", height=.14, held="sensor"))
POWER_HELD = lift("cartridge", height=.14)
SENSOR_ALIGNED = (*SENSOR_HELD, step("align_sensor", obj="sensor"))
POWER_ALIGNED = (*POWER_HELD, step("align_cartridge", obj="cartridge"))
TRIALS = {
    "approach_keeper": Trial(grasp_keeper()[:1], grasp_keeper()[1], keeper=0),
    "swing_keeper": Trial(grasp_keeper(), step("swing_keeper", obj="keeper", target=KEEPER_OPEN), keeper=0),
    "descend_to_sensor": Trial(grasp_sensor("sensor")[:1], step("descend_to_sensor", obj="sensor"),
                                installed=("sensor",)),
    "align_sensor": Trial(SENSOR_HELD, step("align_sensor", obj="sensor")),
    "seat_sensor": Trial(SENSOR_ALIGNED, step("seat_sensor", obj="sensor")),
    "extract_sensor": Trial(grasp_sensor("sensor"), step("extract_sensor", obj="sensor"), installed=("sensor",)),
    "align_cartridge": Trial(POWER_HELD, step("align_cartridge", obj="cartridge")),
    "feed_cartridge": Trial(POWER_ALIGNED, step("feed_cartridge", obj="cartridge")),
    "seat_cartridge_stop": Trial((*POWER_ALIGNED, step("feed_cartridge", obj="cartridge")),
                                 step("seat_cartridge_stop", obj="cartridge")),
    "orient_for_inspection": Trial(SENSOR_HELD, step("orient_for_inspection", obj="sensor", yaw=-.70)),
    "withdraw_cartridge": Trial(grasp("cartridge"), step("withdraw_cartridge", obj="cartridge"),
                                 installed=("cartridge",)),
    "seat_in_nest": Trial((*SENSOR_HELD, step("hover_over", held="sensor", xy=[-.053, -.255], height=.22)),
                           step("seat_in_nest", obj="sensor", nest="quarantine")),
}


def start(env, seed, keeper, installed, on_frame=None):
    reset_bench(env, seed, keeper, installed)
    arm = WorkshopArm(env, on_frame)
    return arm, start_task(arm)


def run_trial(env, trial, seed, on_frame=None):
    arm, episode = start(env, seed, trial.keeper, trial.installed, on_frame)
    for call in trial.setup:
        if not episode.success:
            break
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


def run_job(env, job, seed, keeper=None, on_frame=None, selector=None):
    rng = np.random.default_rng(seed)
    keeper = (KEEPER_OPEN if rng.integers(2) else 0.) if keeper is None else keeper
    installed = list(job.installed)
    # Prefill at most one component when the job has two independent installation goals.
    installs = [obj for obj, station in job.placements if station in (PORT, RAIL)]
    if not installed and len(installs) == 2 and rng.random() < .4:
        installed.append(installs[int(rng.integers(2))])
    arm, episode = start(env, seed, keeper, installed, on_frame)
    goals = job.goals(arm)
    run_episode(arm, episode, goals, plan=plan_job(arm, goals, rng), selector=selector)
    return episode, goals, keeper, installed


def run_episode(arm, episode, goals, *, plan=None, selector=None):
    if not episode.success:
        return

    def selected():
        for index in range(61):
            try:
                choice = selector.choose(arm.observe(), goals)
            except NoFeasibleCall:
                episode.failure = "no_feasible_call"
                return
            if choice is None:
                if arm.on_frame is not None:
                    print("  STOP", flush=True)
                return
            if index == 60:
                episode.failure = "call_budget"
                return
            yield Step(*choice)

    calls = plan if selector is None else selected()
    for i, call in enumerate(calls, 1):
        if arm.on_frame is not None:
            print(f"  {i:02d} {call.skill} {call.args}", flush=True)
        result = execute(arm, call)
        episode.calls.append((call, result))
        if not result.success:
            episode.failure = f"call/{i}/{call.skill}"
            break
    if episode.success and not all(satisfied(arm, goal) for goal in goals):
        episode.failure = "final_goal" if selector is None else "premature_STOP"
