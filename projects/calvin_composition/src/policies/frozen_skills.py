"""The only interface from the composition head to the pretrained controller."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.envs.calvin_reset import initial_observations
from src.policies.calvin import load_hulc


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenSkills:
    def __init__(self, asset_root, catalog_path, *, device="auto", renderer="auto", show_gui=False):
        self.model, self.env, self.oracle, self.checkpoint = load_hulc(
            asset_root, device=device, renderer=renderer, show_gui=show_gui,
        )
        catalog = json.loads(Path(catalog_path).read_text())
        self.instructions = {s["id"]: s["instruction"] for s in catalog["skills"]}
        if any(p.requires_grad for p in self.model.parameters()) or self.model.training:
            self.close()
            raise RuntimeError("Base policy must be frozen and in evaluation mode")
        missing = set(self.instructions.values()) - set(self.model.lang_embeddings)
        if missing:
            self.close()
            raise ValueError(f"Instructions missing from released embeddings: {missing}")
        self.original_digest = parameter_digest(self.model)
        self.on_frame = None

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("The base policy was unfrozen")
        if parameter_digest(self.model) != self.original_digest:
            raise RuntimeError("Frozen policy parameters or buffers changed")

    def reset(self, state, seed):
        robot, scene = initial_observations(state, seed=seed)
        self.env.reset(robot_obs=robot, scene_obs=scene)
        self.model.reset()
        return self.state()

    def state(self):
        raw = self.env.unwrapped.get_state_obs()
        vector = np.concatenate([raw["robot_obs"], raw["scene_obs"]]).astype(np.float32)
        if vector.shape != (39,) or not np.isfinite(vector).all():
            raise ValueError("Unexpected or non-finite CALVIN physical state")
        return vector

    def info(self):
        return deepcopy(self.env.get_info())

    def achieved(self, skill, start):
        return bool(self.oracle.get_task_info_for_set(start, self.env.get_info(), {skill}))

    def execute(self, skill, max_steps):
        start = self.info()
        self.model.reset()  # Reset the latent plan, never the physical scene.
        obs = self.env.get_obs()
        success = False
        with torch.inference_mode():
            for steps in range(1, max_steps + 1):
                action = self.model.step(obs, self.instructions[skill])
                if not torch.isfinite(action).all():
                    raise ValueError("Frozen policy produced a non-finite action")
                obs, _, done, _ = self.env.step(action)
                for value in obs.values():
                    tensors = value.values() if isinstance(value, dict) else [value]
                    if any(not torch.isfinite(t).all() for t in tensors):
                        raise ValueError("Non-finite low-level observation")
                if self.on_frame:
                    self.on_frame(self.env, skill, steps)
                success = self.achieved(skill, start)
                if success or done:
                    break
        return success, steps, start

    def close(self):
        self.env.close()
