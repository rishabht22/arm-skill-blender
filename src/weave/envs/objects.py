"""Physical training objects and the hinged variant of RoboFactory's pot."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene
    from mani_skill.utils.structs import Actor, Articulation


def hinged_urdf(source: Path, output: Path, pivot: list[float], travel: float) -> None:
    """Move the lid frame to its hinge and retain its original closed geometry."""
    tree = ElementTree.parse(source)
    root = tree.getroot()
    joint = root.find("./joint[@name='joint_0']")
    if joint is None or joint.attrib["type"] != "prismatic":
        raise ValueError("Unexpected RoboFactory lid joint")
    lid = root.find("./link[@name='link_0']")
    if lid is None or joint.find("child").attrib["link"] != lid.attrib["name"]:
        raise ValueError("Unexpected RoboFactory lid link")
    joint.set("type", "revolute")
    joint.find("origin").set("xyz", " ".join(map(str, pivot)))
    joint.find("axis").set("xyz", "1 0 0")
    joint.find("limit").attrib.update(
        lower="0", upper=str(travel), effort="0", velocity="10"
    )
    for shape in [*lid.findall("visual"), *lid.findall("collision")]:
        origin = shape.find("origin")
        offset = [float(x) for x in origin.get("xyz", "0 0 0").split()]
        origin.set(
            "xyz", " ".join(str(value - anchor) for value, anchor in zip(offset, pivot))
        )
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.attrib["filename"]).resolve()))
    tree.write(output, encoding="utf-8", xml_declaration=True)


def load_hinged_pot(
    scene: ManiSkillScene,
    asset_dir: Path,
    config: dict[str, Any],
) -> Articulation:
    from robofactory.utils.building.nonconvex_urdf_loader import NonconvexURDFLoader

    source = asset_dir / "objects/pot_annotated/mobility.urdf"
    loader = NonconvexURDFLoader()
    loader.set_scene(scene)
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = False
    loader.scale = config["scale"]
    loader.set_density(config["density_kg_m3"])
    with tempfile.TemporaryDirectory(prefix="weave-pot-") as directory:
        adapted = Path(directory) / source.name
        hinged_urdf(source, adapted, config["pivot_asset_xyz"], config["travel_rad"])
        return loader.load(
            str(adapted), package_dir=str(source.parent), name="weave_pot"
        )


def load_training_objects(
    scene: ManiSkillScene,
    config: dict[str, Any],
) -> tuple[list[Actor], list[float], list[list[float]]]:
    from mani_skill import ASSET_DIR
    from mani_skill.utils.building import actors

    entities, support_heights, half_extents = [], [], []
    for index, (edge, color) in enumerate(
        zip(config["cube_edges_m"], config["cube_colors"])
    ):
        half = edge / 2
        entities.append(actors.build_cube(scene, half, color, name=f"cube_{index}"))
        support_heights.append(half)
        half_extents.append([half] * 3)

    metadata = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
    if not metadata.is_file():
        raise FileNotFoundError(
            "YCB assets missing. Run scripts/run_phase1.sh to install them."
        )
    model_db = json.loads(metadata.read_text())
    for model_id in config["ycb_ids"]:
        model = model_db[model_id]
        scale = model.get("scales", [1.0])[0]
        low, high = model["bbox"]["min"], model["bbox"]["max"]
        builder = actors.get_actor_builder(scene, id=f"ycb:{model_id}")
        entities.append(builder.build(name=model_id))
        support_heights.append(-low[2] * scale)
        half_extents.append([(b - a) * scale / 2 for a, b in zip(low, high)])
    return entities, support_heights, half_extents
