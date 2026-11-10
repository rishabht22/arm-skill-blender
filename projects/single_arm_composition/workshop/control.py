"""Execute a single Cartesian path or gripper command and record its outcome."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from individual_skill_generation.skills import TOP_DOWN, PandaArm, Rollout


@dataclass
class Motion:
    positions: np.ndarray
    rotations: Rotation
    grip: float | None = None
    held: str | None = None
    effect: Callable[[], bool] | None = None
    check_pose: bool = True
    guard: Callable[[], bool] | None = None


class WorkshopArm(PandaArm):
    """Reuse the Panda controller, with named entities and fixture state."""

    def __init__(self, env, on_frame=None):
        super().__init__(env)
        self.on_frame = on_frame
        model = env.sim.model
        self.geoms = {name: model.geom_name2id(name) for name in env.entities}
        self.bodies = {name: model.body_name2id(name) for name in env.entities}
        if self.on_frame is not None:
            self.on_frame(env.sim.get_state().flatten())

    def location(self, name):
        geom = self.geoms[name] if name in self.geoms else self.env.sim.model.geom_name2id(name)
        return self.env.sim.data.geom_xpos[geom].copy()

    def orientation(self, name):
        quat = self.env.sim.data.body_xquat[self.bodies[name]]
        return Rotation.from_quat(quat, scalar_first=True)

    def joint(self, name):
        return float(self.env.sim.data.get_joint_qpos(name))

    def holding(self, name):
        return bool(self.env._check_grasp(self.gripper, name))

    def contact(self, first, second):
        return bool(self.env.check_contact(first, second))

    @property
    def aperture(self):
        return abs(np.diff(self.env.sim.data.qpos[self.fingers])[0])

    def observe(self):
        return {
            **self.robot_observation(),
            "entity_position": np.array([self.location(name) for name in self.env.entities]),
            "entity_quaternion_xyzw": np.array([self.orientation(name).as_quat() for name in self.env.entities]),
            "grasped": np.array([self.holding(name) for name in self.env.entities]),
            "fixture_position": np.array([self.joint(name) for name in self.env.fixture_joints]),
        }

    def path(self, positions, rotations=None, **checks):
        positions = np.asarray(positions)
        if rotations is None:
            rotations = self.rotation
        if rotations.single:
            rotations = Rotation.from_quat(np.repeat(rotations.as_quat()[None], len(positions), axis=0))
        return Motion(positions, rotations, **checks)

    def line(self, position, rotation=None, *, steps=None, **checks):
        position = np.asarray(position)
        distance = np.linalg.norm(position - self.position)
        if steps is None:
            steps = max(10, int(np.ceil(distance / .012)))
        positions = np.linspace(self.position, position, steps + 1)[1:]
        rotations = None
        if rotation is not None:
            interpolate = Slerp([0, 1], Rotation.concatenate([self.rotation, rotation]))
            rotations = interpolate(np.linspace(0, 1, steps + 1)[1:])
        return self.path(positions, rotations, **checks)

    def stage(self, position, rotation=TOP_DOWN):
        """Free-space positioning for isolated test setup."""
        return self.run(self.line(position, rotation))

    def run(self, motion):
        observations = [self.observe()]
        states = [self.env.sim.get_state().flatten()]
        actions = []
        held = motion.held
        if held:
            if not self.holding(held):
                return Rollout(False, np.empty((0, self.env.action_dim)),
                               {key: np.asarray([value]) for key, value in observations[0].items()},
                               np.asarray(states))
            offset = self.rotation.inv().apply(self.location(held) - self.position)
        if motion.grip is not None:
            self.grip_command = motion.grip

        def advance(action):
            self.env.step(action)
            actions.append(action)
            states.append(self.env.sim.get_state().flatten())
            if self.on_frame is not None:
                self.on_frame(states[-1])
            observations.append(self.observe())
            if motion.guard is not None and not motion.guard():
                return False
            return held is None or np.linalg.norm(
                self.location(held) - self.position - self.rotation.apply(offset)
            ) < .025

        stable = 0
        for position, rotation in zip(motion.positions, motion.rotations, strict=True):
            if not advance(self.action_for(position, rotation)):
                break
        else:
            action = actions[-1]
            for _ in range(60):
                if not advance(action):
                    break
                reached = np.linalg.norm(self.position - position) < .009
                aligned = (rotation * self.rotation.inv()).magnitude() < .12
                passed = (not motion.check_pose or (reached and aligned))
                passed &= held is None or self.holding(held)
                passed &= motion.effect is None or bool(motion.effect())
                stable = stable + 1 if passed else 0
                if stable == 5:
                    break

        return Rollout(
            stable == 5, np.asarray(actions),
            {key: np.asarray([obs[key] for obs in observations]) for key in observations[0]},
            np.asarray(states),
        )
