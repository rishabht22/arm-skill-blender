"""Closed-loop primitive controllers and physical observations for Aster."""
from dataclasses import dataclass
import math

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .scene import CARRIERS, HOME, SCAN, STEM, WORK, build, layout


PRIMITIVES = ("read_load", "read", "read_return", "load", "close", "latch", "index",
              "unlatch", "open", "unload", "release", "reject", "reseat")
PHASES = (("read_load", "read", "read_return"),
          ("load", "close", "latch", "index", "unlatch", "open", "unload"))
CLEAR = .319
OPEN = .132
LOCK = .026
ANGLE = np.pi / 2
DOWN = Rotation.from_euler("x", np.pi)


@dataclass(frozen=True)
class Outcome:
    ok: bool
    reason: str = ""


def check(condition, reason):
    return Outcome(bool(condition), "" if condition else reason)


class Workcell:
    def __init__(self, layout_id):
        self.configuration = layout(layout_id)
        self.model = mj.MjModel.from_xml_string(build(self.configuration))
        self.data = mj.MjData(self.model)
        self.ik = mj.MjData(self.model)
        self.tcp = self.model.site("tcp").id
        self.limits = self.model.jnt_range[:7].copy()
        self.ik_cache = {}
        self._jp = np.empty((3, self.model.nv))
        self._jr = np.empty_like(self._jp)
        self._body = {self.model.body(i).name: i for i in range(self.model.nbody)}
        self._geom = {self.model.geom(i).name: i for i in range(self.model.ngeom)}
        self._q = {self.model.joint(i).name: int(self.model.jnt_qposadr[i])
                   for i in range(self.model.njnt)}
        self._eq = {self.model.equality(i).name: i for i in range(self.model.neq)}
        self._static_slots = {key: (np.array([x, y, z + STEM]), yaw)
                              for key, (x, y, z, yaw) in self.configuration.slots.items()}
        self._locations = None
        self.reset(0, tuple(range(CARRIERS)), 0)

    def reset(self, seed, bindings, grades, recovery=(), destinations=tuple(range(CARRIERS))):
        self.bindings = bindings
        self.destinations = destinations
        self.homes = tuple(f"feed{slot}" for slot in range(CARRIERS))
        self.grades = tuple((grades >> i) & 1 for i in range(CARRIERS))
        self.reads = [-1] * CARRIERS
        self.cycles = [0] * CARRIERS
        self.index_count = 0
        self._rotor_command = 0.
        self._held = None
        self.trace = None
        self._pressed = False
        self.button_peak = 0.
        mj.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        d = self.data
        d.qpos[:7] = HOME + rng.uniform(-.015, .015, 7)
        d.qpos[7:9] = .031
        d.ctrl[:9] = d.qpos[:9]
        d.qpos[self._q["lid"]] = OPEN
        for job, carrier in enumerate(bindings):
            p = self._q[f"c{carrier}"]
            start = f"recovery{job}" if job in recovery else self.homes[job]
            x, y, z, direction = self.configuration.slots[start]
            d.qpos[p:p + 3] = (x, y, z + STEM + .0005)
            d.qpos[p:p + 2] += rng.uniform(-.00065, .00065, 2)
            yaw = direction + rng.uniform(-.035, .035)
            d.qpos[p + 3:p + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mj.mj_forward(self.model, d)
        self.advance(25)

    def position(self, body):
        return self.data.xpos[self._body[body]].copy()

    def rotation(self, body):
        return Rotation.from_matrix(self.data.xmat[self._body[body]].reshape(3, 3), assume_valid=True)

    def j(self, name):
        return float(self.data.qpos[self._q[name]])

    def geom(self, name):
        return self.data.geom_xpos[self._geom[name]].copy()

    def hand(self):
        return self.data.site_xpos[self.tcp].copy()

    def advance(self, ticks, *, panel=False):
        self._locations = None
        if self.trace is not None:
            self.trace.append((ticks, self.data.ctrl.copy(), self.data.eq_active.copy(),
                               self.model.eq_data.copy(), panel))
        if not panel:
            mj.mj_step(self.model, self.data, nstep=ticks)
            return
        for _ in range(ticks):
            mj.mj_step(self.model, self.data)
            depth = self.j("button")
            self.button_peak = max(depth, self.button_peak)
            pressed = depth > .007
            if pressed and not self._pressed and self.j("lid") < .007 and self.j("bolt") > .022:
                self._rotor_command = ANGLE if self._rotor_command < 1 else 0.
                self.data.ctrl[9] = self._rotor_command
                self.index_count += 1
            self._pressed = pressed

    def ik_target(self, point, yaw):
        # Round only the IK cache key/target, below the keyed seating tolerance.
        key = (*np.rint(np.asarray(point) * 5000).astype(int), round(yaw, 3))
        if key in self.ik_cache:
            return self.ik_cache[key]
        target = np.asarray(key[:3]) / 5000
        # Use the quantized yaw in the solve as well as in the key, so an
        # earlier rollout cannot change the result of a later one.
        rotation = (Rotation.from_euler("z", key[3]) * DOWN).as_matrix()
        q = self.ik.qpos
        q[:7] = HOME
        for _ in range(36):
            mj.mj_kinematics(self.model, self.ik)
            mj.mj_comPos(self.model, self.ik)
            current = self.ik.site_xmat[self.tcp].reshape(3, 3)
            error = target - self.ik.site_xpos[self.tcp]
            orient = Rotation.from_matrix(rotation @ current.T, assume_valid=True).as_rotvec()
            if np.linalg.norm(error) < .00012 and np.linalg.norm(orient) < .001:
                break
            mj.mj_jacSite(self.model, self.ik, self._jp, self._jr, self.tcp)
            jac = np.concatenate((self._jp[:, :7], .36 * self._jr[:, :7]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + .00008 * np.eye(6),
                                            np.r_[error, .36 * orient])
            q[:7] = np.clip(q[:7] + np.clip(delta, -.23, .23),
                            self.limits[:, 0] + .02, self.limits[:, 1] - .02)
        result = q[:7].copy()
        if len(self.ik_cache) >= 24_000:
            self.ik_cache.clear()
        self.ik_cache[key] = result
        return result

    def pose(self, target, yaw=0., *, speed=.48, settle=16, panel=False, linear=False, stop=None):
        start = self.data.ctrl[:7].copy()
        distance = np.linalg.norm(target - self.hand())
        ticks = max(12, int(distance / (speed * self.model.opt.timestep)))
        segments = max(2, math.ceil(ticks / 6))
        if linear:
            origin = self.hand()
            joints = [self.ik_target(origin + (target - origin) * (i / segments), yaw)
                      for i in range(1, segments + 1)]
        else:
            end = self.ik_target(target, yaw)
            joints = [start + (end - start) * (i / segments) for i in range(1, segments + 1)]
        for command in joints:
            self.data.ctrl[:7] = command
            self.advance(math.ceil(ticks / segments), panel=panel)
            if stop is not None and stop():
                self.data.ctrl[:7] = self.data.qpos[:7]
                return False
        self.advance(settle, panel=panel)
        return True

    def jaws(self, width, ticks=16):
        self.data.ctrl[7:9] = width
        self.advance(ticks)

    def touches(self, first, second):
        a, b = self._geom[first], self._geom[second]
        g = self.data.contact.geom[:self.data.ncon]
        return bool(np.any(((g[:, 0] == a) & (g[:, 1] == b)) |
                           ((g[:, 1] == a) & (g[:, 0] == b))))

    def grip(self, body, handle, yaw):
        point = self.geom(handle)
        self.jaws(.031, 8)
        self.pose(np.array([*point[:2], CLEAR]), yaw)
        self.pose(point + [0, 0, .003], yaw, speed=.34, settle=29, linear=True)
        self.jaws(.0045, 34)
        geom_id = self._geom[handle]
        pairs = self.data.contact.geom[:self.data.ncon]
        other = np.r_[pairs[pairs[:, 0] == geom_id, 1], pairs[pairs[:, 1] == geom_id, 0]]
        fingers = self.model.geom_bodyid[other]
        if not all(self._body[name] in fingers for name in ("left_finger", "right_finger")):
            self.jaws(.031)
            self.pose(np.r_[self.hand()[:2], CLEAR], yaw, linear=True)
            return False
        # Capture the measured relative pose after opposed jaw contact. This
        # models a compliant retention aid and does not align or move the part.
        hand = self.rotation("hand")
        offset = hand.inv().apply(self.position(body) - self.position("hand"))
        quat = (hand.inv() * self.rotation(body)).as_quat()
        eq = self._eq[f"grip_{body}"]
        self.model.eq_data[eq, :3] = 0
        self.model.eq_data[eq, 3:6] = offset
        self.model.eq_data[eq, 6:10] = quat[[3, 0, 1, 2]]
        self.data.eq_active[eq] = True
        self._held = body
        return True

    def let_go(self, yaw):
        self.data.eq_active[self._eq[f"grip_{self._held}"]] = False
        self._held = None
        self.jaws(.031, 14)
        self.pose(np.r_[self.hand()[:2], CLEAR], yaw, linear=True)
        self.advance(8)

    def carrier(self, job):
        return f"c{self.bindings[job]}"

    def slot(self, tag):
        pos, yaw = self._static_slots[tag]
        if tag == WORK:
            return pos + [0, 0, .006], self.j("rotor")
        return pos, yaw

    def seated(self, job, tag):
        carrier = self.carrier(job)
        target, yaw = self.slot(tag)
        desired = Rotation.from_euler("z", yaw)
        return (np.linalg.norm(self.position(carrier) - target) < .0038
                and (desired.inv() * self.rotation(carrier)).magnitude() < .14
                and self.touches(f"{carrier}_stem", f"{tag}_floor"))

    def transfer(self, job, source, target, *, malformed=False):
        carrier = self.carrier(job)
        yaw = self.rotation(carrier).as_euler("xyz")[2]
        # The gripping neck is symmetric under a half turn.
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        if not self.grip(carrier, f"{carrier}_grip", yaw):
            return Outcome(False, f"grasp:{source}")
        # First clear the keyed throat in a straight, axial extraction.
        self.pose(np.r_[self.hand()[:2], CLEAR], yaw, linear=True)
        position, target_yaw = self.slot(target)
        if malformed:
            target_yaw += np.pi
        hand_rotation = self.rotation("hand")
        relative = self.rotation(carrier).inv() * hand_rotation
        wanted = Rotation.from_euler("z", target_yaw) * relative
        hand_yaw = (wanted * DOWN.inv()).as_euler("xyz")[2]
        hand_yaw = (hand_yaw + np.pi + .000001) % (2 * np.pi) - np.pi
        # Rotate in the free volume before entering the bore.
        self.pose(np.r_[position[:2], CLEAR], hand_yaw, speed=.62)
        hand_rotation = self.rotation("hand")
        offset = hand_rotation.inv().apply(self.hand() - self.position(carrier))
        end = position + hand_rotation.apply(offset) - [0, 0, .003]
        self.pose(end + [0, 0, .037], hand_yaw, speed=.5)
        self.pose(end, hand_yaw, speed=.23, settle=25, linear=True)
        if not malformed and (target != WORK or self.j("lid") > OPEN - .007) and not self.seated(job, target):
            # One guarded recentering pass with the measured grasp transform.
            self.pose(self.hand() + [0, 0, .034], hand_yaw, linear=True)
            actual = self.rotation("hand")
            rel = self.rotation(carrier).inv() * actual
            corrected = Rotation.from_euler("z", target_yaw) * rel * DOWN.inv()
            hand_yaw = corrected.as_euler("xyz")[2]
            end = position + self.hand() - self.position(carrier) - [0, 0, .002]
            self.pose(end + [0, 0, .033], hand_yaw, speed=.25, settle=30)
            self.pose(end, hand_yaw, speed=.17, settle=28, linear=True)
        self.let_go(hand_yaw)
        return check(self.seated(job, target), f"seat:{target}")

    def move_slide(self, name, target):
        handle = f"{name}_handle"
        yaw = 0. if name == "lid" else np.pi / 2
        axis = np.array((1, 0, 0) if name == "lid" else (0, 1, 0))
        delta = target - self.j(name)
        if abs(delta) < .0025:
            return Outcome(True)
        if not self.grip(name, handle, yaw):
            return Outcome(False, f"grasp:{name}")
        stop = (lambda: self.touches("strike", "bolt_pin")) if name == "lid" else None
        clear = self.pose(self.hand() + delta * axis, yaw, speed=.25, settle=23, linear=True,
                          stop=stop)
        self.let_go(yaw)
        return check(clear and abs(self.j(name) - target) < .007,
                     f"travel:{name}" if clear else "contact:strike_bolt")

    def press(self):
        before = self.index_count
        button = self.geom("button_cap")
        self.jaws(.002)
        self.pose(button + [0, 0, .105])
        self.pose(button + [0, 0, -.0025], speed=.30, linear=True, panel=True)
        self.pose(button + [0, 0, .105], linear=True, panel=True)
        self.jaws(.031)
        self.advance(75)
        return check(self.index_count == before + 1 and
                     abs(self.j("rotor") - self._rotor_command) < .045, "index:interlock_or_stall")

    def locations(self):
        if self._locations is None:
            result = ["displaced"] * CARRIERS
            contact = self.data.contact.geom[:self.data.ncon]
            floor_ids = {self._geom[f"{tag}_floor"]: tag for tag in self._static_slots}
            stem_ids = {self._geom[f"{self.carrier(job)}_stem"]: job for job in range(CARRIERS)}
            for first, second in contact:
                if first in stem_ids:
                    first, second = second, first
                job = stem_ids.get(second)
                tag = floor_ids.get(first)
                if job is not None and tag is not None and self.seated(job, tag):
                    result[job] = tag
            self._locations = tuple(result)
        return self._locations

    def location(self, job):
        return self.locations()[job]

    def available(self, kind, job):
        home = self.homes[job]
        locations = self.locations()
        here = locations[job]
        open_ = self.j("lid") > OPEN - .007
        closed = self.j("lid") < .007
        locked = self.j("bolt") > .022
        unlocked = self.j("bolt") < .006
        if kind in ("load", "read_load"):
            dest = WORK if kind == "load" else SCAN
            return here == home and dest not in locations and (kind != "load" or open_)
        if kind in ("close", "latch", "index", "unlatch", "open", "unload"):
            if here != WORK:
                return False
            return {
                "close": not closed and unlocked, "latch": closed and not locked,
                "index": closed and locked, "unlatch": closed and not unlocked,
                "open": not open_ and unlocked, "unload": open_,
            }[kind]
        if kind in ("read", "read_return"):
            return here == SCAN
        if kind in ("release", "reject"):
            return here == home and self.reads[job] == (kind == "reject")
        if kind == "reseat":
            return here == f"recovery{job}"
        raise ValueError(kind)

    def execute(self, kind, job):
        if not self.available(kind, job):
            return Outcome(False, "precondition")
        home = self.homes[job]
        if kind == "load":
            return self.transfer(job, home, WORK)
        if kind == "unload":
            return self.transfer(job, WORK, home)
        if kind == "read_load":
            return self.transfer(job, home, SCAN)
        if kind == "read_return":
            return self.transfer(job, SCAN, home)
        if kind == "read":
            beam = self.data.site("read_beam").xpos.copy() - [0, .025, 0]
            hit = np.full(1, -1, dtype=np.int32)
            group = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
            distance = mj.mj_ray(self.model, self.data, beam, np.array([0., 1., 0.]),
                                 group, True, -1, hit)
            if hit[0] != self._geom[f"{self.carrier(job)}_stem"] or not 0 < distance < .05:
                return Outcome(False, "reader:beam")
            self.reads[job] = self.grades[job]
            self.advance(8)
            return Outcome(True)
        if kind == "reseat":
            return self.transfer(job, f"recovery{job}", home)
        if kind in ("release", "reject"):
            return self.transfer(job, home, f"{kind}{self.destinations[job]}")
        if kind == "close":
            return self.move_slide("lid", 0.)
        if kind == "open":
            return self.move_slide("lid", OPEN)
        if kind == "latch":
            return self.move_slide("bolt", LOCK)
        if kind == "unlatch":
            return self.move_slide("bolt", 0.)
        outcome = self.press()
        if outcome.ok and self.seated(job, WORK):
            self.cycles[job] += 1
            return outcome
        return Outcome(False, outcome.reason or "index:carrier_displaced")
