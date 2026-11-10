"""Industrial-style connector service cell on the existing Panda controller."""

from pathlib import Path

import numpy as np

from baseline.individual_skill_generation.skills import make_env
from baseline.workshop.environment import PandaWorkshop


SCENE = "service"
KEEPER_OPEN = -.99
SENSORS = ("sensor", "spare")
STEM_LENGTH = .044
TONGUE_DEPTH = .038
CONTACT_PART = {"sensor": "sensor_stem", "spare": "spare_stem", "cartridge": "cartridge_tongue"}


class PandaServiceBench(PandaWorkshop):
    scene_files = {SCENE: Path(__file__).with_name("scenes") / "service.xml"}

    def _reset_internal(self):
        super()._reset_internal()
        for name in (*SENSORS, "cartridge"):
            pose = self.sim.data.get_joint_qpos(name).copy()
            yaw = self.rng.uniform(-.38, .38)
            pose[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.sim.data.set_joint_qpos(name, pose)


def make_bench(seed):
    return make_env(seed, environment="PandaServiceBench", scene=SCENE,
                    initialization_noise={"magnitude": .004, "type": "gaussian"})
