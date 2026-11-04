"""Independent, fixed feedback skills for a physical two-arm collaboration.

Each skill emits only its own arm's actions. The coordinator controls when
skills start and when the holder puts the container down. No learned weights,
welds, object repositioning, or partner-controller calls are hidden here.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .cooperative_scene import make_cooperative_scene, reset_scene

STAGES = ("idle", "approach", "descend", "close", "lift", "holding", "carry",
          "lower", "open", "retreat", "done", "failed")
ACTIONS = ("wait", "start_holder", "start_placer", "release_holder")


class ArmSkill:
    def __init__(self, env, index):
        self.env, self.index = env, index
        self.stage, self.age, self.near = "idle", 0, 0
        self.target = self.eef
        self.rotation = self.orientation
        self.grip = -1.
        self.failure = None
        self.events = []
        self.grasp_offset = np.zeros(3)
        self.lower_target = None

    @property
    def eef(self):
        return self.env.sim.data.site_xpos[self.env.robots[self.index].eef_site_id["right"]].copy()

    @property
    def orientation(self):
        return self.env.sim.data.site_xmat[self.env.robots[self.index].eef_site_id["right"]].reshape(3, 3).copy()

    @property
    def object_position(self):
        body = self.env.pot_body_id if self.index == 0 else self.env.block_body_id
        return self.env.sim.data.body_xpos[body].copy()

    @property
    def grasped(self):
        geoms = self.env.pot.handle0_geoms if self.index == 0 else self.env.block.contact_geoms
        return bool(self.env._check_grasp(self.env.robots[self.index].gripper["right"], geoms))

    def enter(self, stage, target=None):
        self.stage, self.age, self.near = stage, 0, 0
        if target is not None:
            self.target = np.asarray(target).copy()
        self.events.append(stage)

    def start(self):
        if self.stage != "idle":
            return False
        if self.index == 0:
            # The handle bar runs along world X; close the finger pads along Y.
            self.rotation = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        point = (self.env.sim.data.site_xpos[self.env.handle0_site_id].copy()
                 if self.index == 0 else self.object_position)
        self.enter("approach", point + [0., 0., .13])
        return True

    def release(self):
        if self.index != 0 or self.stage != "holding":
            return False
        self.enter("lower", self.lower_target)
        return True

    def fail(self, message):
        self.failure = message
        self.enter("failed", self.eef)

    def action(self):
        result = np.zeros(7)
        result[:3] = np.clip(10. * (self.target - self.eef), -.65, .65)
        error = Rotation.from_matrix(self.rotation @ self.orientation.T).as_rotvec()
        result[3:6] = np.clip(2. * error, -.5, .5)
        result[6] = self.grip
        return result

    def advance(self):
        self.age += 1
        error = np.linalg.norm(self.target - self.eef)
        rotation_error = Rotation.from_matrix(self.rotation @ self.orientation.T).magnitude()
        self.near = self.near + 1 if error < .005 and rotation_error < .06 else 0
        reached = self.near >= 8
        if self.stage not in ("idle", "holding", "done", "failed") and self.age > 240:
            self.fail(f"{self.stage} timed out: position error {error:.3f} m")
            return
        if self.stage == "approach" and reached:
            point = (self.env.sim.data.site_xpos[self.env.handle0_site_id].copy()
                     if self.index == 0 else self.object_position + [0., 0., .005])
            self.enter("descend", point)
        elif self.stage == "descend" and reached:
            self.grip = 1.
            self.enter("close", self.eef)
        elif self.stage == "close" and self.age >= 30:
            if not self.grasped:
                self.fail("both fingers did not grasp the object")
                return
            self.grasp_offset = self.eef - self.object_position
            self.lower_target = self.eef.copy()
            self.enter("lift", self.eef + [0., 0., .075 if self.index == 0 else .17])
        elif self.stage == "lift" and reached:
            if not self.grasped:
                self.fail("object slipped during lift")
            elif self.index == 0:
                self.enter("holding", self.eef)
            else:
                target = self.eef
                target[:2] = self.env.sim.data.body_xpos[self.env.pot_body_id][:2]
                # Clear the rim while carrying across, even if the holder is late.
                target[2] = max(target[2], 1.03)
                self.enter("carry", target)
        elif self.stage == "carry" and reached:
            pot = self.env.sim.data.body_xpos[self.env.pot_body_id].copy()
            # Release above the rim so the open fingers never enter the box.
            self.enter("lower", pot + [0., 0., .085] + self.grasp_offset)
        elif self.stage == "lower" and reached:
            self.grip = -1.
            self.enter("open", self.eef)
        elif self.stage == "open" and self.age >= 30:
            self.enter("retreat", self.eef + [0., 0., .14])
        elif self.stage == "retreat" and reached:
            self.enter("done", self.eef)


class CooperativeSkills:
    def __init__(self):
        self.env = make_cooperative_scene()

    def reset(self, seed=0, holder_delay=0):
        reset_scene(self.env, seed)
        self.skills = [ArmSkill(self.env, index) for index in range(2)]
        self.steps, self.delay = 0, int(holder_delay)
        self.pending_holder = False
        self.handoff_verified = False
        self.premature_release = False
        self.stable_finish = 0
        self.initial_pot_z = float(self.pot[2])
        self.closed_placer = False
        self.events = []
        return self

    @property
    def pot(self):
        return self.env.sim.data.body_xpos[self.env.pot_body_id].copy()

    @property
    def block(self):
        return self.env.sim.data.body_xpos[self.env.block_body_id].copy()

    @property
    def upright(self):
        matrix = self.env.sim.data.body_xmat[self.env.pot_body_id].reshape(3, 3)
        return bool(matrix[2, 2] > np.cos(np.deg2rad(12.)))

    @property
    def inside(self):
        matrix = self.env.sim.data.body_xmat[self.env.pot_body_id].reshape(3, 3)
        relative = matrix.T @ (self.block - self.pot)
        return bool(abs(relative[0]) < .043 and abs(relative[1]) < .043
                    and -.026 < relative[2] < .03)

    @property
    def holding(self):
        holder = self.skills[0]
        return bool(holder.stage == "holding" and holder.grasped and self.upright
                    and self.pot[2] > self.initial_pot_z + .04)

    @property
    def success(self):
        return self.stable_finish >= 20

    @property
    def failure(self):
        return next((skill.failure for skill in self.skills if skill.failure), None)

    def command(self, action):
        if action == 1 and self.skills[0].stage == "idle" and not self.pending_holder:
            self.pending_holder = True
            return True
        if action == 2:
            return self.skills[1].start()
        if action == 3:
            if self.skills[0].stage == "holding":
                if not self.handoff_verified or self.skills[1].stage != "done":
                    self.premature_release = True
                return self.skills[0].release()
        return action == 0

    def baseline_action(self):
        if self.skills[0].stage == "idle" and not self.pending_holder:
            return 1
        if self.holding and self.skills[1].stage == "idle":
            return 2
        if self.skills[1].stage == "done" and self.skills[0].stage == "holding":
            return 3
        return 0

    def step(self):
        if self.pending_holder and self.steps >= self.delay:
            self.skills[0].start()
            self.pending_holder = False
        old_stages = tuple(skill.stage for skill in self.skills)
        self.env.step(np.concatenate([skill.action() for skill in self.skills]))
        self.steps += 1
        for skill in self.skills:
            skill.advance()
        new_stages = tuple(skill.stage for skill in self.skills)
        if new_stages != old_stages:
            self.events.append({"step": self.steps, "holder": new_stages[0], "placer": new_stages[1]})
        if self.skills[1].stage == "close":
            self.closed_placer = True
        if (self.closed_placer and self.skills[1].stage in ("open", "retreat", "done")
                and self.inside and self.holding):
            self.handoff_verified = True
        final = (self.handoff_verified and not self.premature_release and self.inside
                 and self.upright and all(s.stage == "done" for s in self.skills)
                 and abs(self.pot[2] - .84) < .012
                 and self.env.check_contact(self.env.block, self.env.pot))
        self.stable_finish = self.stable_finish + 1 if final else 0
        if not np.isfinite(self.env.sim.data.qpos).all():
            raise RuntimeError("Non-finite physics state")

    def report(self):
        return dict(success=self.success, steps=self.steps, handoff_verified=self.handoff_verified,
                    premature_release=self.premature_release, inside=self.inside,
                    holder=self.skills[0].stage, placer=self.skills[1].stage,
                    failure=self.failure, pot=self.pot.tolist(), block=self.block.tolist(),
                    events=self.events)

    def close(self):
        self.env.close()
