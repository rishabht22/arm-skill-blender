"""Standalone skill trials. Setup actions are tracked separately for replay."""

from dataclasses import dataclass, field

import numpy as np

from .skills import PandaSkills


CONTEXTS = (
    "approach", "descend", "open_empty",
    "close_empty", "grasp", "lift", "carry", "lower", "release", "retreat", "move_behind", "descend_behind", "push",
)


@dataclass
class Trial:
    arm: PandaSkills
    reset_state: np.ndarray
    setup_actions: list = field(default_factory=list)

    def setup(self, call):
        rollout = self.arm.execute(call)
        self.setup_actions.extend(rollout.actions)
        if not rollout.success:
            raise SetupFailure(f"{call.skill} failed at {call.position.round(3).tolist()}")


class SetupFailure(RuntimeError):
    pass


def prepare_trial(env, context, seed):
    env.rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    env.reset()
    rng = np.random.default_rng(seed + 1)
    arm = PandaSkills(env)
    trial = Trial(arm, env.sim.get_state().flatten())
    if context not in CONTEXTS:
        raise ValueError(f"Unknown context: {context}")
    if context == "open_empty":
        trial.setup(arm.close_gripper())
        return trial, arm.open_gripper()
    trial.setup(arm.open_gripper())
    if context == "close_empty":
        return trial, arm.close_gripper()

    if context in ("move_behind", "descend_behind", "push"):
        trial.setup(arm.close_gripper())
        direction = rng.choice([-1, 1])
        if context == "move_behind":
            return trial, arm.move_behind(direction)
        trial.setup(arm.move_behind(direction))
        if context == "descend_behind":
            return trial, arm.descend_behind(direction)
        trial.setup(arm.descend_behind(direction))
        return trial, arm.push_along(direction, rng.uniform(0.073, 0.10))

    if context == "approach":
        return trial, arm.hover_over()
    trial.setup(arm.hover_over())
    if context == "descend":
        return trial, arm.descend_to_grasp()
    trial.setup(arm.descend_to_grasp())
    if context == "grasp":
        return trial, arm.close_gripper(grasp=True)
    trial.setup(arm.close_gripper(grasp=True))

    height = rng.uniform(0.10, 0.14)
    if context == "lift":
        return trial, arm.raise_by(height)
    trial.setup(arm.raise_by(height))

    destination = [rng.uniform(-0.07, 0.10), rng.choice([-1, 1]) * rng.uniform(0.08, 0.17)]
    if context == "carry":
        return trial, arm.hover_over(destination)
    trial.setup(arm.hover_over(destination))

    release_height = env.table_offset[2] + env.cube.size[2] + 0.05
    if context == "lower":
        return trial, arm.lower_to(release_height)
    trial.setup(arm.lower_to(release_height))
    if context == "release":
        return trial, arm.open_gripper(release=True)
    trial.setup(arm.open_gripper(release=True))
    return trial, arm.raise_by(0.12)
