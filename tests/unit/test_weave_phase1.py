from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import torch

from weave.envs.contact_monitor import ContactMonitor
from weave.envs.objects import hinged_urdf
from weave.envs.reset_dists import sample_scene


class LaneRng:
    def __init__(self, seeds: list[int]) -> None:
        self.rngs = [np.random.RandomState(seed) for seed in seeds]
        self.batch_size = len(seeds)

    def uniform(self, *args, **kwargs) -> np.ndarray:
        return np.array([rng.uniform(*args, **kwargs) for rng in self.rngs])

    def randint(self, *args, **kwargs) -> np.ndarray:
        return np.array([rng.randint(*args, **kwargs) for rng in self.rngs])


def test_reset_layout_is_seeded_per_lane_and_parks_inactive_objects() -> None:
    config = {
        "spawn_slots_xy_m": [[0, 0], [1, 0], [2, 0]],
        "spawn_jitter_m": 0.01,
        "spawn_clearance_m": 0.002,
        "parking_origin_m": [9, 0, 0],
        "parking_spacing_m": 0.2,
        "min_active_objects": 1,
    }
    device = torch.device("cpu")
    positions, yaw, active = sample_scene(
        LaneRng([11, 19, 23]), config, [0.1] * 3, device
    )
    for batch_value, single_value in zip(
        (positions, yaw, active), sample_scene(LaneRng([19]), config, [0.1] * 3, device)
    ):
        torch.testing.assert_close(batch_value[1], single_value[0])
    assert not torch.equal(positions[0], positions[1])
    assert (~active).any()
    assert (positions[~active][:, 0] == 9).all()
    assert (yaw[~active] == 0).all()


def test_hinge_keeps_closed_shape_at_original_transform(tmp_path: Path) -> None:
    source = tmp_path / "native.urdf"
    source.write_text(
        '<robot><link name="link_0"><visual><origin xyz="1 2 3"/>'
        '<geometry><mesh filename="lid.obj"/></geometry></visual></link>'
        '<joint name="joint_0" type="prismatic"><origin xyz="0 0 0"/>'
        '<axis xyz="0 1 0"/><child link="link_0"/><limit lower="0" upper="999"/>'
        "</joint></robot>"
    )
    output = tmp_path / "hinge.urdf"
    hinged_urdf(source, output, [0, 1, 2], 1.25)
    root = ElementTree.parse(output).getroot()
    joint = root.find("joint")
    assert joint.attrib["type"] == "revolute"
    assert float(joint.find("limit").attrib["upper"]) == 1.25
    assert joint.find("axis").attrib["xyz"] == "1 0 0"
    assert root.find("link/visual/origin").attrib["xyz"] == "1.0 1.0 1.0"
    assert root.find(".//mesh").attrib["filename"] == str(tmp_path / "lid.obj")


def test_contact_monitor_counts_frames_and_excludes_target_and_inactive_objects() -> (
    None
):
    agents = [
        SimpleNamespace(robot=SimpleNamespace(get_links=lambda n=n: [n]))
        for n in ("L", "R")
    ]
    forces = {
        ("L", "R"): [3.0, 0.0],
        ("R", "table"): [2.0, 2.0],
        ("L", "obj0"): [7.0, 7.0],
        ("L", "obj1"): [4.0, 4.0],
    }

    def pair_force(first: str, second: str) -> torch.Tensor:
        x = torch.tensor(forces.get((first, second), [0.0, 0.0]))
        return torch.stack((x, x * 0, x * 0), dim=-1)

    monitor = ContactMonitor(
        SimpleNamespace(
            scene=SimpleNamespace(get_pairwise_contact_forces=pair_force),
            agent=SimpleNamespace(agents=agents),
            table="table",
            objects=["obj0", "obj1"],
            num_envs=2,
            device=torch.device("cpu"),
        ),
        1.0,
    )
    active = torch.tensor([[True, True], [True, False]])
    targets = torch.tensor([[0, -1], [0, -1]])
    expected = torch.tensor([[3, 0, 2, 4, 0], [0, 0, 2, 0, 0]], dtype=torch.float32)
    torch.testing.assert_close(monitor.sample(active, targets), expected)
    monitor.sample(active, targets)
    torch.testing.assert_close(monitor.frames, (expected > 1).to(torch.int32) * 2)
    monitor.reset(torch.tensor([0]))
    assert not monitor.peak_n[0].any()
    assert monitor.peak_n[1, 2] == 2
