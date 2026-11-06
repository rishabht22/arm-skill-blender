"""Contact and sampled swept-volume metrics for batched control steps."""

from __future__ import annotations

from typing import Any

import torch


class ContactMonitor:
    def __init__(self, env: Any, threshold_n: float) -> None:
        self.scene = env.scene
        self.threshold_n = threshold_n
        left, right = [agent.robot.get_links() for agent in env.agent.agents]
        self.arm_pairs = [(a, b) for a in left for b in right]
        self.table_pairs = [
            [(link, env.table) for link in links] for links in (left, right)
        ]
        self.object_pairs = [
            [[(link, obj) for link in links] for obj in env.objects]
            for links in (left, right)
        ]
        # Columns are arm-arm, left/right table, left/right non-target object.
        self.peak_n = torch.zeros((env.num_envs, 5), device=env.device)
        self.frames = torch.zeros_like(self.peak_n, dtype=torch.int32)

    def reset(self, lanes: torch.Tensor) -> None:
        self.peak_n[lanes] = 0
        self.frames[lanes] = 0

    def _peak(self, pairs: list[tuple[Any, Any]]) -> torch.Tensor:
        peak = torch.zeros_like(self.peak_n[:, 0])
        for first, second in pairs:
            force = self.scene.get_pairwise_contact_forces(first, second)
            peak = torch.maximum(peak, torch.linalg.vector_norm(force, dim=-1))
        return peak

    def sample(self, active: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        peaks = [self._peak(self.arm_pairs)]
        peaks.extend(self._peak(pairs) for pairs in self.table_pairs)
        for arm, object_pairs in enumerate(self.object_pairs):
            non_target = [
                self._peak(pairs).masked_fill(
                    ~active[:, idx] | (targets[:, arm] == idx), 0
                )
                for idx, pairs in enumerate(object_pairs)
            ]
            peaks.append(torch.stack(non_target, dim=-1).amax(dim=-1))
        forces = torch.stack(peaks, dim=-1)
        self.frames.add_(forces > self.threshold_n)
        self.peak_n.copy_(torch.maximum(self.peak_n, forces))
        return forces
