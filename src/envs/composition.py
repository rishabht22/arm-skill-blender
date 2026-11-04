"""One shared train/test environment for privileged-state skill selection.

Actions select frozen HULC options. An option runs until CALVIN's native task
oracle reports success or the time limit is reached. The graph and state are
visible to the head. This is a single-arm, graph-conditioned baseline.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


def encode_observation(state, skills, task, completed, decisions, frames, decision_limit, option_steps):
    index = {skill: i for i, skill in enumerate(skills)}
    n = len(skills)
    goal, done, graph = np.zeros(n), np.zeros(n), np.zeros((n, n))
    for skill in task.sequence:
        goal[index[skill]] = 1
        for requirement in task.prerequisites(skill):
            graph[index[requirement], index[skill]] = 1
    for skill in completed:
        done[index[skill]] = 1
    state = np.asarray(state, dtype=np.float32).copy()
    state[3:6] /= np.pi
    state[7:14] /= np.pi
    obs = np.concatenate([state, goal, done, graph.ravel(),
                          [decisions / decision_limit, frames / (option_steps * decision_limit)]])
    if not np.isfinite(obs).all():
        raise ValueError("Non-finite head observation")
    return np.clip(obs, -10., 10.).astype(np.float32)


class CompositionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, backend, active_skills, tasks, *, option_steps=360, decision_limit=12):
        super().__init__()
        if option_steps < 1 or decision_limit < max(len(t.sequence) for t in tasks):
            raise ValueError("Invalid option or decision budget")
        self.backend, self.skills, self.tasks = backend, tuple(active_skills), tuple(tasks)
        self.option_steps, self.decision_limit = option_steps, decision_limit
        self.index = {skill: i for i, skill in enumerate(self.skills)}
        n = len(self.skills)
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(-10., 10., shape=(39 + 2 * n + n * n + 2,), dtype=np.float32)
        self.completed, self.starts = set(), {}
        self.task = None
        self.ended = True

    def _observation(self):
        return encode_observation(self.backend.state(), self.skills, self.task, self.completed,
                                  self.decisions, self.frames, self.decision_limit, self.option_steps)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        name = (options or {}).get("task")
        candidates = [task for task in self.tasks if name is None or task.name == name]
        if not candidates:
            raise ValueError(f"Unknown task: {name}")
        self.task = candidates[int(self.np_random.integers(len(candidates)))]
        self.episode_seed = int(seed if seed is not None else self.np_random.integers(2**31))
        self.backend.reset(self.task.initial_state, self.episode_seed)
        self.completed, self.starts = set(), {}
        self.decisions, self.frames, self.ended = 0, 0, False
        self.trace = []
        return self._observation(), self._info(False, "reset")

    def _info(self, success, reason, **kwargs):
        return dict(is_success=bool(success), task=self.task.name, completed=len(self.completed),
                    required=len(self.task.sequence), decisions=self.decisions, frames=self.frames,
                    episode_seed=self.episode_seed, reason=reason, **kwargs)

    def step(self, action):
        if self.ended:
            raise RuntimeError("reset() is required before stepping")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid skill action: {action}")
        self.decisions += 1
        state_before = self.backend.state().tolist()
        skill = self.skills[int(action)]
        ready = (skill in self.task.sequence and skill not in self.completed
                 and self.task.prerequisites(skill) <= self.completed)
        terminated, success = False, False
        reward, reason, steps = -.25, "invalid_selection", 0
        if ready:
            achieved, steps, start = self.backend.execute(skill, self.option_steps)
            self.frames += steps
            if achieved:
                self.completed.add(skill)
                self.starts[skill] = start
                reward, reason = 1. - .0001 * steps, "skill_success"
                if len(self.completed) == len(self.task.sequence):
                    success = all(self.backend.achieved(s, self.starts[s]) for s in self.task.final_checks)
                    terminated = True
                    reward += 5. if success else -1.
                    reason = "success" if success else "final_condition_lost"
            else:
                reward, terminated, reason = -1., True, "frozen_skill_failed"
        truncated = not terminated and self.decisions >= self.decision_limit
        self.ended = terminated or truncated
        info = self._info(
            success, "decision_limit" if truncated else reason, skill=skill, option_steps=steps,
            selection_valid=ready,
        )
        self.trace.append(dict(skill=skill, selection_valid=ready, reward=float(reward),
                               frames=steps, reason=info["reason"], state_before=state_before,
                               state_after=self.backend.state().tolist()))
        if self.ended:
            info["decision_trace"] = list(self.trace)
        return self._observation(), float(reward), terminated, truncated, info

    def teacher_action(self):
        # Explicit scripted baseline and warm-start teacher, never the learned actor.
        for skill in self.task.sequence:
            if skill not in self.completed:
                return self.index[skill]
        raise RuntimeError("Task already complete")

    def close(self):
        self.backend.close()
