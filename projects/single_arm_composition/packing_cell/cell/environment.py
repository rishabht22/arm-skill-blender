"""A Panda workcell with a mechanically blocked sliding cargo drawer."""

from pathlib import Path

from baseline.individual_skill_generation.skills import make_env
from baseline.workshop.environment import PandaWorkshop


SCENE = "packing"
DRAWER_OPEN = .173
BOLT_UP = .055
PARTS = {"red": .018, "blue": .018, "green": .016}


class PandaPackingCell(PandaWorkshop):
    scene_files = {SCENE: Path(__file__).with_name("scenes") / "packing.xml"}


def make_cell(seed):
    return make_env(seed, environment="PandaPackingCell", scene=SCENE,
                    initialization_noise={"magnitude": .004, "type": "gaussian"})
