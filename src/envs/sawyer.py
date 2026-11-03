"""Keep robosuite 1.5.1's Sawyer meshes compatible with MuJoCo 3.3."""

from robosuite.models.robots import Sawyer as RobosuiteSawyer


class Sawyer(RobosuiteSawyer):
    """Sawyer with shell inertia for its two flat visual head meshes.

    robosuite registers subclasses by name, so this replaces its Sawyer
    model in the registry without changing the robot name in experiments.
    """

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        # These visual meshes have no volume. The bodies already provide
        # explicit inertias; this only fixes MuJoCo's mesh compilation.
        # Upstream fix: https://github.com/ARISE-Initiative/robosuite/pull/662
        for name in ("head_1", "head_4"):
            mesh = self.asset.find(f"mesh[@name='{self.naming_prefix}{name}']")
            if mesh is not None:
                mesh.set("inertia", "shell")
