"""A shared scorer for all skill-and-carrier pairs."""

import numpy as np
import torch
from torch import nn

from .planner import ACTIONS, FEED, SORTED, SKILLS


JOBS = np.array([job for _, job in ACTIONS])
SKILL_FEATURES = np.repeat(np.eye(len(SKILLS), dtype=np.float32), len(set(JOBS)), axis=0)
FEATURES = len(SKILLS) + 6 + 2 + 4 + 4 + 4 + 2


def encode(goal, state):
    location = np.eye(6, dtype=np.float32)[list(state.locations)]
    observed = np.column_stack((location, state.reads, np.asarray(state.cycles) / 2))
    desired = np.asarray([[target.cycles / 2, target.inspect, target.destination == FEED,
                           target.destination == SORTED] for target in goal.targets], dtype=np.float32)
    roles = np.zeros((len(state.locations), 4), dtype=np.float32)
    if goal.read_before_index:
        first, second = goal.read_before_index
        roles[first] = [1, 0, 0, state.cycles[second] / 2]
        roles[second] = [0, 1, state.reads[first], 0]
    shared = np.r_[np.eye(4)[state.enclosure], location[:, 1:3].any(axis=0)]
    return np.concatenate((SKILL_FEATURES, observed[JOBS], desired[JOBS], roles[JOBS],
                           np.broadcast_to(shared, (len(ACTIONS), len(shared)))), axis=1, dtype=np.float32)


def network(width):
    return nn.Sequential(nn.Linear(FEATURES, width), nn.SiLU(), nn.Linear(width, width),
                         nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))


class Selector:
    def __init__(self, model):
        self.model = model.eval()

    @torch.inference_mode()
    def scores(self, goal, state):
        return self.model(torch.from_numpy(encode(goal, state))).squeeze(-1).numpy()

    def __call__(self, goal, state):
        return int(np.argmax(self.scores(goal, state)))
