"""Matched physics probes for keying and mechanical retention."""

from dataclasses import replace

from baseline.workshop.examples import grasp

from .environment import KEEPER_OPEN
from .runs import TRIALS, run_trial, start
from .skills import sensor_pose


def rejects_reversed_key(env, seed):
    results = []
    for quaternion in ([1, 0, 0, 0], [0, 0, 0, 1]):
        arm, episode = start(env, seed, KEEPER_OPEN, ())
        if not episode.success:
            return False
        target = sensor_pose(arm)
        env.sim.data.set_joint_qpos("sensor", [*target[:2], target[2] + .041, *quaternion])
        env.sim.forward()
        action = arm.action_for(arm.position, arm.rotation)
        rib_blocked = False
        for _ in range(25):
            env.step(action)
            rib_blocked |= arm.contact("sensor_rib", "port_west")
        results.append((arm.contact("sensor_stem", "sensor_floor"), rib_blocked))
    return results == [(True, False), (False, True)]


def retains_locked_sensor(env, seed):
    episode = run_trial(env, replace(TRIALS["extract_sensor"], keeper=0, setup=grasp("sensor")), seed)
    if episode.success or not episode.calls:
        return False
    trace = episode.calls[0][1].observations
    heights = trace["entity_position"][:, env.entities.index("sensor"), 2]
    keeper = trace["fixture_position"][:, env.fixture_joints.index("keeper")]
    return float(max(heights) - heights[0]) < .035 and float(min(keeper)) > -.25
