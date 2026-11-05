"""Frozen released, point-cloud-conditioned joint-action assembly policy."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from .assets import VENDOR, bootstrap

# Public catalogs are exclusive, even though both roles reuse a pretrained backbone.
SKILLS = {
    0: ('R4', 'R12', 'P4', 'P12', 'BNC', 'USB'),
    1: ('Gs', 'Gm', 'Gl', 'N8', 'N12', 'RJ45'),
}
TASKS = SKILLS[0] + SKILLS[1]
CHECKPOINT = VENDOR / 'checkpoints/e8192_it12500.pt'


class Experts:
    def __init__(self):
        bootstrap()
        from assembly_policy.evaluate import TorchPolicy, env_config
        torch.set_num_threads(1)
        self.policy = TorchPolicy(CHECKPOINT, list(TASKS), device='cpu', deterministic=True)
        self.policy.model.eval().requires_grad_(False)
        self.config = env_config(self.policy.args, nthread=1)
        self.checkpoint_sha256 = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
        self.initial = self.digest()

    def digest(self):
        h = hashlib.sha256()
        for name, tensor in self.policy.model.state_dict().items():
            h.update(name.encode()); h.update(tensor.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def assert_frozen(self):
        if self.policy.model.training or any(p.requires_grad or p.grad is not None for p in self.policy.model.parameters()):
            raise RuntimeError('Assembly expert is not frozen')
        if self.digest() != self.initial:
            raise RuntimeError('Assembly expert parameters changed')
        if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != self.checkpoint_sha256:
            raise RuntimeError('Released expert checkpoint changed')

    def __call__(self, env):
        return self.policy(env)
