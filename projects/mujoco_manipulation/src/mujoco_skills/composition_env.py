"""The identical Gymnasium task used for coordinator training and evaluation."""

import gymnasium as gym
import numpy as np

from .cooperative import ACTIONS, STAGES, CooperativeSkills


class CompositionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, frames_per_action=10, max_decisions=150, on_frame=None):
        self.scene = CooperativeSkills()
        self.frames_per_action = int(frames_per_action)
        self.max_decisions = int(max_decisions)
        if self.frames_per_action < 1 or self.max_decisions < 1:
            raise ValueError("Action duration and episode budget must be positive")
        self.on_frame = on_frame
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(-10., 10., (57,), dtype=np.float32)
        self.decisions = 0
        self.milestones = set()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        episode_seed = int(self.np_random.integers(0, 2**31 - 1)) if seed is None else seed
        delay = int(self.np_random.integers(0, 81))
        self.scene.reset(episode_seed, holder_delay=delay)
        self.decisions, self.milestones = 0, set()
        return self.observation(), self.info()

    def observation(self):
        s = self.scene
        # Ground-truth simulator state. This is a state-based coordination study.
        positions = np.r_[s.pot, s.block, s.skills[0].eef, s.skills[1].eef]
        velocities = np.r_[s.env.sim.data.get_body_xvelp(s.env.pot.root_body),
                           s.env.sim.data.get_body_xvelp(s.env.block.root_body)]
        rotations = s.env.sim.data.body_xmat[s.env.pot_body_id].copy()
        stages = np.concatenate([np.eye(len(STAGES))[STAGES.index(k.stage)] for k in s.skills])
        flags = [s.holding, s.inside, s.handoff_verified, s.pending_holder,
                 s.skills[0].grasped, s.skills[1].grasped]
        result = np.r_[positions, velocities, rotations, stages, flags].astype(np.float32)
        if result.shape != self.observation_space.shape or not np.isfinite(result).all():
            raise RuntimeError("Invalid coordinator observation")
        return np.clip(result, -10., 10.)

    def info(self):
        return {"is_success": self.scene.success, "failure": self.scene.failure,
                "physics_steps": self.scene.steps, "handoff_verified": self.scene.handoff_verified}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid coordinator action {action}")
        s = self.scene
        valid = s.command(int(action))
        self.decisions += 1
        reward = -.01 - (.05 if not valid else 0.)
        for _ in range(self.frames_per_action):
            s.step()
            if self.on_frame:
                self.on_frame(s)
            checks = {"held": s.holding,
                      "picked": s.skills[1].stage in ("carry", "lower", "open", "retreat", "done"),
                      "transferred": s.handoff_verified, "finished": s.success}
            for name, condition in checks.items():
                if condition and name not in self.milestones:
                    reward += {"held": 1., "picked": 1., "transferred": 3., "finished": 10.}[name]
                    self.milestones.add(name)
            if s.failure or s.premature_release or s.success:
                break
        failed = bool(s.failure or s.premature_release)
        if failed:
            reward -= 5.
        terminated = failed or s.success
        truncated = self.decisions >= self.max_decisions and not terminated
        return self.observation(), reward, terminated, truncated, self.info()

    def close(self):
        self.scene.close()
