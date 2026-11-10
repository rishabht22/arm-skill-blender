"""Three MuJoCo workcells with one Panda and fixed fixture geometry."""

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask

from ..individual_skill_generation.skills import make_env


SCENES = ("tabletop", "fixtures", "assembly")
TABLE_Z = 0.8


class PandaWorkshop(ManipulationEnv):
    scene_files = {name: Path(__file__).with_name("scenes") / f"{name}.xml" for name in SCENES}

    def __init__(self, scene, **kwargs):
        if scene not in self.scene_files:
            raise ValueError(f"Unknown workshop scene: {scene}")
        self.scene = scene
        super().__init__(**kwargs)

    def _load_model(self):
        super()._load_model()
        arena = TableArena(table_offset=(0, 0, TABLE_Z))
        robot = self.robots[0].robot_model
        robot.set_base_xpos(robot.base_xpos_offset["table"](arena.table_full_size[0]))
        scene_path = self.scene_files[self.scene]
        scene = ElementTree.parse(scene_path).getroot()
        worldbody = scene.find("worldbody")
        equality = scene.find("equality")
        if equality is not None:
            arena.root.append(equality)
        bodies = worldbody.findall("body")
        self.entities = tuple(body.attrib["name"] for body in bodies)
        self.free_objects = tuple(body.attrib["name"] for body in bodies if body.find("freejoint") is not None)
        self.fixture_joints = tuple(joint.attrib["name"] for joint in worldbody.iter("joint"))
        # robosuite calculates inertia from group 0 and renders group 1.
        for body in bodies:
            for geom in body.findall("geom"):
                collision = ElementTree.SubElement(body, "geom", {**geom.attrib, "group": "0"})
                if "name" in geom.attrib:
                    geom.set("name", collision.attrib["name"] + "_visual")
                geom.attrib.update(contype="0", conaffinity="0")
        arena.worldbody.extend(worldbody)
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot])

    def _reset_internal(self):
        super()._reset_internal()
        for name in self.free_objects:
            pose = self.sim.data.get_joint_qpos(name).copy()
            # Keep insertion and channel start poses within their staging geometry.
            jitter = 0.002 if name in ("peg", "key", "puck") else 0.007
            pose[:2] += self.rng.uniform(-jitter, jitter, size=2)
            self.sim.data.set_joint_qpos(name, pose)

    def reward(self, action=None):
        return 0.0


def make_workshop(scene, seed):
    return make_env(seed, environment="PandaWorkshop", scene=scene,
                    initialization_noise={"magnitude": .004, "type": "gaussian"})
