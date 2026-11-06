"""Phase 1 cell on the native RoboFactory two-Panda environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import robofactory
import sapien
import torch
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
from mani_skill.utils.structs.pose import Pose
from robofactory.tasks.two_robots_stack_cube import TwoRobotsStackCubeEnv
from transforms3d.euler import euler2quat

from .contact_monitor import ContactMonitor
from .objects import load_hinged_pot, load_training_objects
from .reset_dists import sample_scene


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)), dim=-1)


class TwoArmCellEnv(TwoRobotsStackCubeEnv):
    """Fixed topology, independent lane resets, and physical joint-delta control."""

    def __init__(
        self, cell_config: dict[str, Any], num_envs: int, backend: str = "gpu"
    ) -> None:
        self.cell_config = cell_config
        super().__init__(
            num_envs=num_envs,
            sim_backend=backend,
            obs_mode="state_dict",
            control_mode="pd_joint_delta_pos",
            reward_mode="none",
            enhanced_determinism=True,
        )

    def _load_scene(self, options: dict[str, Any]) -> None:
        # Reuse the native scene builder and table. Replace its task-specific primitives.
        self.cfg["scene"]["primitives"] = []
        super()._load_scene(options)
        cfg = self.cell_config
        self.objects, self.support_heights, half_extents = load_training_objects(
            self.scene, cfg
        )
        self.half_extents = torch.tensor(half_extents, device=self.device)
        self.pot = load_hinged_pot(self.scene, Path(robofactory.ASSET_DIR), cfg["pot"])
        self.markers = {
            name: actors.build_box(
                self.scene,
                [marker["radius_m"]] * 2 + [cfg["marker_half_height_m"]],
                [0.22, 0.55, 0.65, 1.0],
                name=name,
                body_type="kinematic",
                add_collision=False,
            )
            for name, marker in cfg["fixtures"].items()
        }
        self.active_objects = torch.zeros(
            (self.num_envs, len(self.objects)), dtype=torch.bool, device=self.device
        )
        self.target_ids = torch.full(
            (self.num_envs, 2), -1, dtype=torch.long, device=self.device
        )
        self.previous_action = torch.zeros((self.num_envs, 2, 8), device=self.device)
        self.contact_monitor = ContactMonitor(self, cfg["contact_threshold_n"])
        self.footprint_links = [
            [agent.tcp, *(self._link(agent, name) for name in cfg["footprint_links"])]
            for agent in self.agent.agents
        ]
        self.footprint_local_bounds = [
            [self._link_bounds(link) for link in links]
            for links in self.footprint_links
        ]
        self.footprint_active = torch.zeros_like(self.target_ids, dtype=torch.bool)
        self.footprint_bounds = torch.zeros(
            (self.num_envs, 2, 2, 3), device=self.device
        )

    @staticmethod
    def _link(agent: Any, name: str) -> Any:
        return next(link for link in agent.robot.get_links() if link.name == name)

    def _initialize_episode(
        self, env_idx: torch.Tensor, options: dict[str, Any]
    ) -> None:
        rng = self._batched_episode_rng[env_idx]
        cfg = self.cell_config
        positions, yaw, active = sample_scene(
            rng, cfg, self.support_heights, self.device
        )
        for idx, actor in enumerate(self.objects):
            actor.set_pose(
                Pose.create_from_pq(positions[:, idx], _yaw_quaternion(yaw[:, idx]))
            )
        self.active_objects[env_idx] = active
        self.target_ids[env_idx] = -1
        self.previous_action[env_idx] = 0
        self.contact_monitor.reset(env_idx)
        self.footprint_active[env_idx] = False
        self.footprint_bounds[env_idx] = 0

        for name, actor in self.markers.items():
            marker = cfg["fixtures"][name]
            actor.set_pose(
                Pose.create_from_pq(
                    self._fixture_position(
                        rng,
                        marker["xy_m"],
                        marker["jitter_m"],
                        cfg["marker_half_height_m"],
                    )
                )
            )
        pot = cfg["pot"]
        self.pot.set_pose(
            Pose.create_from_pq(
                self._fixture_position(
                    rng, pot["xy_m"], pot["xy_jitter_m"], pot["root_z_m"]
                )
            )
        )
        lid = self._tensor(rng.uniform(0, pot["reset_max_angle_rad"], (1,)))
        self.pot.set_qpos(lid)

        noise = cfg["joint_noise_rad"]
        for agent, agent_cfg in zip(self.agent.agents, self.cfg["agents"]):
            pos = agent_cfg["pos"]
            agent.robot.set_pose(
                sapien.Pose(pos["ppos"]["p"], euler2quat(*pos["ppos"]["q"]))
            )
            qpos = self._tensor(pos["qpos"]).expand(len(env_idx), -1).clone()
            qpos[:, :7] += self._tensor(rng.uniform(-noise, noise, (7,)))
            agent.reset(qpos)

    def _tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _fixture_position(
        self, rng: Any, xy: list[float], jitter: float, z: float
    ) -> torch.Tensor:
        xy_positions = self._tensor(xy) + self._tensor(
            rng.uniform(-jitter, jitter, (2,))
        )
        return torch.cat(
            (xy_positions, xy_positions.new_full((rng.batch_size, 1), z)), dim=1
        )

    def evaluate(self) -> dict[str, torch.Tensor]:
        # The skill or task executor supplies the physical goal predicate.
        return {}

    def _get_obs_extra(self, info: dict[str, Any]) -> dict[str, Any]:
        object_poses = [actor.pose for actor in self.objects]
        fixture_poses = [
            self.pot.pose,
            *(actor.pose for actor in self.markers.values()),
        ]
        arm_state = {}
        for index, (name, agent) in enumerate(self.agent.agents_dict.items()):
            base_from_world = agent.robot.pose.inv()
            partner = self.agent.agents[1 - index]
            arm_state[name] = {
                "tcp_pose": (base_from_world * agent.tcp.pose).raw_pose,
                "partner_tcp_pose": (base_from_world * partner.tcp.pose).raw_pose,
                "partner_links": torch.stack(
                    [
                        (base_from_world * link.pose).p
                        for link in self.footprint_links[1 - index][1:]
                    ],
                    dim=1,
                ),
                "object_poses": torch.stack(
                    [(base_from_world * pose).raw_pose for pose in object_poses], dim=1
                ),
                "fixture_poses": torch.stack(
                    [(base_from_world * pose).raw_pose for pose in fixture_poses], dim=1
                ),
                "previous_action": self.previous_action[:, index].clone(),
            }
        return {
            "arms": arm_state,
            "objects_world": torch.stack(
                [pose.raw_pose for pose in object_poses], dim=1
            ),
            "object_half_extents": self.half_extents.expand(self.num_envs, -1, -1),
            "active_objects": self.active_objects.clone(),
            "fixtures_world": torch.stack(
                [pose.raw_pose for pose in fixture_poses], dim=1
            ),
            "lid_angle": self.pot.get_qpos().clone(),
        }

    def begin_footprint(self, arm: int, lanes: torch.Tensor) -> None:
        self.footprint_bounds[lanes, arm] = self._current_bounds(arm)[lanes]
        self.footprint_active[lanes, arm] = True

    def end_footprint(self, arm: int, lanes: torch.Tensor) -> torch.Tensor:
        if not bool(self.footprint_active[lanes, arm].all()):
            raise ValueError("Start a footprint before ending it")
        self.footprint_active[lanes, arm] = False
        return self.footprint_bounds[lanes, arm].clone()

    def _link_bounds(self, link: Any) -> torch.Tensor:
        mesh = get_component_mesh(link._objs[0], to_world_frame=False)
        return (
            self._tensor(mesh.bounds)
            if mesh is not None
            else self._tensor([[0] * 3] * 2)
        )

    def _current_bounds(self, arm: int) -> torch.Tensor:
        world_bounds = []
        for link, local in zip(
            self.footprint_links[arm], self.footprint_local_bounds[arm]
        ):
            pose = link.pose
            rotation = pose.to_transformation_matrix()[:, :3, :3]
            center = pose.p + (rotation @ local.mean(dim=0))
            half_extent = rotation.abs() @ ((local[1] - local[0]) / 2)
            world_bounds.append(
                torch.stack((center - half_extent, center + half_extent), dim=1)
            )
        bounds = torch.stack(world_bounds, dim=1)
        return torch.stack(
            (bounds[:, :, 0].amin(dim=1), bounds[:, :, 1].amax(dim=1)), dim=1
        )

    def _step_action(self, action: dict[str, torch.Tensor]) -> Any:
        for index, name in enumerate(self.agent.agents_dict):
            self.previous_action[:, index] = action[name]
        return super()._step_action(action)

    def step(
        self,
        action: dict[str, torch.Tensor],
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        obs, reward, terminated, truncated, info = super().step(action)
        for arm in range(2):
            current = self._current_bounds(arm)
            bounds = self.footprint_bounds[:, arm]
            swept = torch.stack(
                (
                    torch.minimum(bounds[:, 0], current[:, 0]),
                    torch.maximum(bounds[:, 1], current[:, 1]),
                ),
                dim=1,
            )
            bounds.copy_(
                torch.where(self.footprint_active[:, arm, None, None], swept, bounds)
            )
        info["contact_force_n"] = self.contact_monitor.sample(
            self.active_objects, self.target_ids
        )
        info["contact_peak_n"] = self.contact_monitor.peak_n.clone()
        info["contact_frames"] = self.contact_monitor.frames.clone()
        return obs, reward, terminated, truncated, info
