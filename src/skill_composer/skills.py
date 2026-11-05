"""Fixed low-level executors. Replace this interface with pretrained arm skills."""
from typing import Protocol

import numpy as np


class SkillExecutor(Protocol):
    def command(self, arm: int, skill: int) -> np.ndarray:
        """Return seven joint position targets and two finger-opening targets."""
        ...

    def is_complete(self, arm: int, skill: int, qpos: np.ndarray, attached: bool) -> bool:
        """Report termination from the executor's measured state."""
        ...


class FrozenJointSkills:
    """Panda position-control primitives; no torch module or trainable weights.

    These modest placeholders use median training-demo endpoint configurations.
    They are not independently trained grasp policies or a motion planner.
    """

    def __init__(self, library):
        self.targets = []
        for arm in library["arms"]:
            approach, lift = np.asarray(arm["approach"]), np.asarray(arm["lift"])
            targets = np.stack([np.r_[approach, .04, .04],
                                np.r_[approach, 0., 0.],
                                np.r_[lift, 0., 0.],
                                np.r_[lift, 0., 0.]])
            targets.flags.writeable = False
            self.targets.append(targets)

    def command(self, arm, skill):
        if arm not in (0, 1) or skill not in range(4):
            raise ValueError("Expected arm 0/1 and skill 0..3")
        return self.targets[arm][skill].copy()

    def is_complete(self, arm, skill, qpos, attached):
        if skill == -1:
            return True
        if skill in (0, 2):
            reached = np.linalg.norm(np.asarray(qpos)[:7] - self.command(arm, skill)[:7]) < .035
            return bool(reached and (skill == 0 or attached))
        if skill == 1:
            return bool(attached)
        return False  # hold persists until the task ends.
