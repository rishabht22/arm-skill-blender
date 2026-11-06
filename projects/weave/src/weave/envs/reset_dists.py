"""Per-lane randomization with fixed scene topology."""

from __future__ import annotations

from typing import Any

import torch


def sample_scene(
    rng: Any,
    config: dict[str, Any],
    support_heights: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return object positions, yaw, and active masks in reset-lane order."""
    count = len(support_heights)
    lane_count = rng.batch_size
    if len(config["spawn_slots_xy_m"]) != count:
        raise ValueError("Configure one tabletop spawn slot per instantiated object")

    slot_order = torch.as_tensor(
        rng.uniform(size=count), device=device, dtype=torch.float32
    ).argsort(dim=1)
    slots = torch.as_tensor(
        config["spawn_slots_xy_m"], dtype=torch.float32, device=device
    )
    jitter = config["spawn_jitter_m"]
    xy = slots[slot_order] + torch.as_tensor(
        rng.uniform(-jitter, jitter, (count, 2)), dtype=torch.float32, device=device
    )
    z = (
        torch.as_tensor(support_heights, dtype=torch.float32, device=device)
        + config["spawn_clearance_m"]
    )
    positions = torch.cat((xy, z.expand(lane_count, -1).unsqueeze(-1)), dim=-1)
    active_count = torch.as_tensor(
        rng.randint(config["min_active_objects"], count + 1), device=device
    )
    active = slot_order < active_count[:, None]
    parked = (
        torch.as_tensor(config["parking_origin_m"], dtype=torch.float32, device=device)
        .expand(count, -1)
        .clone()
    )
    parked[:, 1] += torch.arange(count, device=device) * config["parking_spacing_m"]
    positions = torch.where(active[..., None], positions, parked)
    yaw = torch.as_tensor(
        rng.uniform(-torch.pi, torch.pi, count), device=device, dtype=torch.float32
    )
    return positions, torch.where(active, yaw, 0.0), active
