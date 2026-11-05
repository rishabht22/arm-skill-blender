"""MuJoCo development sandbox, not a port of the RoboFactory benchmark.

    Official Panda kinematics/inertias and collision geometry; simple barrier;
    proximity-gated idealized point attachments instead of contact-based grasps.
"""
import copy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .common import ROOT, decision_features
from .skills import FrozenJointSkills


def _vector(value):
    return " ".join(str(float(v)) for v in value)


def _robot_scene():
    source = ROOT / "assets/panda/panda.xml"
    original = ET.parse(source).getroot()
    root = ET.Element("mujoco", model="two_panda_skill_sandbox")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(source.parent / "assets"), autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", iterations="80")
    ET.SubElement(root, "visual").append(ET.Element("global", offwidth="960", offheight="720"))
    root.append(copy.deepcopy(original.find("default")))
    root.append(copy.deepcopy(original.find("asset")))
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(world, "geom", name="table", type="plane", size="2 2 .1", rgba=".16 .19 .23 1",
                  contype="3", conaffinity="3")
    actuator = ET.SubElement(root, "actuator")
    contact = ET.SubElement(root, "contact")
    for a, y in enumerate((-.75, .75)):
        body = copy.deepcopy(original.find("worldbody/body"))
        body.set("pos", f"0 {y} 0")
        yaw = np.pi / 2 if a == 0 else -np.pi / 2
        body.set("quat", _vector([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]))
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name", f"a{a}_" + element.get("name"))
            if element.tag == "body":
                element.set("gravcomp", "1")
        for geom in body.iter("geom"):
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.set("group", "1")
            geom.set("rgba", ".85 .88 .92 1" if a == 0 else ".28 .57 .78 1")
        hand = next(b for b in body.iter("body") if b.get("name") == f"a{a}_hand")
        ET.SubElement(hand, "site", name=f"a{a}_tcp", pos="0 0 .1034", size=".008", rgba="1 .6 .1 1")
        world.append(body)
        for joint in body.iter("joint"):
            name = joint.get("name")
            is_finger = "finger" in name
            ET.SubElement(actuator, "position", name=name + "_motor", joint=name,
                          kp="200" if is_finger else "900", kv="8" if is_finger else "70",
                          forcerange="-100 100")
        ET.SubElement(contact, "exclude", body1=f"a{a}_link0", body2=f"a{a}_link1")
    return root


class LiftSandbox:
    control_dt = .05  # RoboFactory records at 20 Hz.

    def __init__(self, library, seed=0, randomization=0.0):
        if randomization < 0:
            raise ValueError("randomization must be nonnegative")
        self.executor = FrozenJointSkills(library)
        root = _robot_scene()
        # Locate the barrier's handles using the nominal calibrated approach poses.
        nominal = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        d = mujoco.MjData(nominal)
        d.qpos[:] = np.concatenate([self.executor.command(a, 0) for a in range(2)])
        mujoco.mj_forward(nominal, d)
        endpoints = np.stack([d.site(f"a{a}_tcp").xpos.copy() for a in range(2)])
        self.start_center = endpoints.mean(0)
        self.local_handles = endpoints - self.start_center
        world = root.find("worldbody")
        for a, p in enumerate(endpoints):
            # Small static supports keep the proxy object at its calibrated height.
            h = max(.005, p[2] - .015)
            ET.SubElement(world, "geom", name=f"support{a}", type="box",
                          pos=_vector([p[0], p[1], h / 2]), size=_vector([.025, .025, h / 2]),
                          rgba=".3 .32 .35 1", contype="2", conaffinity="2")
        barrier = ET.SubElement(world, "body", name="barrier", pos=_vector(self.start_center))
        ET.SubElement(barrier, "freejoint", name="barrier_free")
        ET.SubElement(barrier, "geom", type="capsule", name="barrier_bar",
                      fromto=_vector(self.local_handles.reshape(-1)), size=".015", mass=".1",
                      rgba=".94 .58 .16 1", contype="2", conaffinity="2")
        eq = ET.SubElement(root, "equality")
        for a in range(2):
            ET.SubElement(eq, "connect", name=f"grasp{a}", body1=f"a{a}_hand", body2="barrier",
                          anchor="0 0 0", active="false", solref=".01 1")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        for a in range(2):
            e = self.model.equality(f"grasp{a}").id
            self.model.eq_data[e, :3] = [0, 0, .1034]
            self.model.eq_data[e, 3:6] = self.local_handles[a]
        self.offset = np.r_[np.random.default_rng(seed).uniform(-randomization, randomization, 2), 0.]
        self.data.qpos[:18] = np.concatenate([arm["initial"] for arm in library["arms"]])
        self.data.qpos[18:21] += self.offset
        self.data.ctrl[:] = self.data.qpos[:18]
        mujoco.mj_forward(self.model, self.data)
        self.initial_height = self.data.body("barrier").xpos[2]
        self.attachments = np.zeros(2, dtype=bool)
        self.collision_frames = 0

    def completed(self, active):
        """Low-level skills own termination; this is not a learned scheduler."""
        return np.asarray([self.executor.is_complete(a, int(skill), self.data.qpos[a * 9:(a + 1) * 9],
                                                    bool(self.attachments[a]))
                           for a, skill in enumerate(active)], dtype=bool)

    def decision_observation(self, active, terminated):
        return decision_features(self.data.qpos[:18].reshape(2, 9),
                                 self.data.qvel[:18].reshape(2, 9), active, terminated)

    def handles(self):
        body = self.data.body("barrier")
        return self.local_handles @ body.xmat.reshape(3, 3).T + body.xpos

    def step(self, skills, guarded=True):
        skills = np.asarray(skills, dtype=int).copy()
        blocked = 0
        # A simple temporal constraint; the head still decides when to switch.
        if guarded and not self.attachments.all():
            needs_shared_grasp = np.isin(skills, [2, 3])
            blocked = int(needs_shared_grasp.sum())
            skills[needs_shared_grasp] = 1
        for a, s in enumerate(skills):
            if s == -1:  # Parked arm keeps its existing motor targets.
                continue
            self.data.ctrl[a * 9:(a + 1) * 9] = self.executor.command(a, int(s))
        for _ in range(round(self.control_dt / self.model.opt.timestep)):
            handles = self.handles()
            for a in range(2):
                proximity = np.linalg.norm(self.data.site(f"a{a}_tcp").xpos - handles[a])
                fingers = self.data.qpos[a * 9 + 7:a * 9 + 9].mean()
                if skills[a] == 1 and fingers < .008 and proximity < .05:
                    self.attachments[a] = True
                if skills[a] == 0:
                    self.attachments[a] = False
                self.data.eq_active[self.model.equality(f"grasp{a}").id] = self.attachments[a]
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("Nonfinite MuJoCo state")
        # Count real arm-arm contacts, not barrier attachment events.
        collided = False
        for contact in self.data.contact:
            b1 = self.model.body(self.model.geom_bodyid[contact.geom1]).name
            b2 = self.model.body(self.model.geom_bodyid[contact.geom2]).name
            collided |= (b1.startswith("a0_") and b2.startswith("a1_")) or (b2.startswith("a0_") and b1.startswith("a1_"))
        self.collision_frames += int(collided)
        return skills, blocked

    def status(self):
        height = float(self.data.body("barrier").xpos[2] - self.initial_height)
        level_error = float(abs(np.diff(self.handles()[:, 2])[0]))
        return {"height_gain_m": height, "handle_height_difference_m": level_error,
                "attachments": self.attachments.tolist(), "arm_collision_frames": self.collision_frames,
                "sandbox_goal_reached": bool(height > .15 and level_error < .04 and self.attachments.all())}
