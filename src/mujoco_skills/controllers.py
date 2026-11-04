"""Scripted closed-loop skills. These are not learned neural policies.

Only normal OSC actions move the robot and contact forces move the cubes.
No object teleportation, welded grasp, or physics-state reset occurs in a chain.
"""

import numpy as np
from contextlib import nullcontext


class TabletopSkills:
    def __init__(self, env, on_step=None, step_context=nullcontext):
        self.env, self.on_step = env, on_step
        self.controller = env.robots[0].part_controllers["right"]
        self.gripper = -1.
        self.events = []
        self.step_context = step_context

    @property
    def eef(self):
        return self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id["right"]].copy()

    def cube(self, name):
        return self.env.sim.data.body_xpos[getattr(self.env, name + "_body_id")].copy()

    def step(self, target):
        delta = np.clip(10. * (np.asarray(target) - self.eef), -.65, .65)
        if self.controller.input_ref_frame == "base":
            delta = self.controller.origin_ori.T @ delta
        action = np.zeros(self.env.action_dim)
        action[:3], action[-1] = delta, self.gripper
        with self.step_context():
            observation, _, done, _ = self.env.step(action)
        if self.on_step:
            self.on_step(self.env, self.events[-1] if self.events else "start")
        if not np.isfinite(self.eef).all():
            raise RuntimeError("Non-finite end-effector position")
        if done:
            raise RuntimeError("Scene horizon exhausted before composition finished")
        return observation

    def move(self, target, *, tolerance=.005, budget=180):
        target = np.asarray(target).copy()
        stable = 0
        for _ in range(budget):
            self.step(target)
            stable = stable + 1 if np.linalg.norm(target - self.eef) < tolerance else 0
            if stable >= 8:
                return
        raise RuntimeError(f"Movement failed: error={np.linalg.norm(target - self.eef):.3f} m")

    def hold(self, target, steps):
        for _ in range(steps):
            self.step(target)

    def reach(self, name="cubeA"):
        self.events.append("reach")
        self.gripper = -1.
        target = self.cube(name) + [0., 0., .13]
        self.move(target)

    def grasp(self, name="cubeA"):
        self.events.append("grasp")
        self.move(self.cube(name) + [0., 0., .005])
        self.gripper = 1.
        self.hold(self.eef, 30)
        obj = getattr(self.env, name)
        if not self.env._check_grasp(self.env.robots[0].gripper["right"], obj.contact_geoms):
            raise RuntimeError("Grasp failed: both fingers are not contacting the cube")

    def lift(self):
        self.events.append("lift")
        start = self.cube("cubeA")
        self.move(self.eef + [0., 0., .14])
        if self.cube("cubeA")[2] < start[2] + .09:
            raise RuntimeError("Lift failed: the cube did not follow the gripper")

    def carry(self, target_xy):
        self.events.append("carry")
        target = self.eef
        target[:2] = target_xy
        self.move(target)

    def place(self, support="cubeB"):
        self.events.append("place")
        # Account for each object's half-height and the current grasp offset.
        offset = self.eef - self.cube("cubeA")
        target = self.cube(support) + [0., 0., .025 + .02 + .003] + offset
        self.move(target, tolerance=.004)

    def release(self):
        self.events.append("release")
        self.gripper = -1.
        self.hold(self.eef, 30)

    def place_on_table(self, target_xy):
        self.events.append("place_on_table")
        offset = self.eef - self.cube("cubeA")
        target = np.array([*target_xy, .8 + .02 + .003]) + offset
        self.move(target, tolerance=.004)

    def retreat(self):
        self.events.append("retreat")
        self.move(self.eef + [0., 0., .14])
        self.hold(self.eef, 30)

    def stack(self):
        self.reach()
        self.grasp()
        self.lift()
        self.carry(self.cube("cubeB")[:2])
        self.place()
        self.release()
        self.retreat()
        return bool(self.env._check_success())

    def unstack_and_relocate(self, target_xy=(-.12, .16)):
        self.reach()
        self.grasp()
        self.lift()
        self.carry(target_xy)
        self.place_on_table(target_xy)
        self.release()
        self.retreat()
        cube = self.cube("cubeA")
        return bool(np.linalg.norm(cube[:2] - target_xy) < .025 and abs(cube[2] - .82) < .01)
