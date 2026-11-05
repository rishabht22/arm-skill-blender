"""Both native assembly stations in one real-contact MuJoCo world.

Each station supplies one workpiece at job dispatch; only that workpiece may
be replaced at a job boundary. Every other robot/part state is carried over.
There are no welded grasps or phase controllers inside the neural expert.
"""
from collections import OrderedDict
import xml.etree.ElementTree as ET

import mujoco
from mujoco import rollout
import numpy as np

from .assets import bootstrap
from .experts import TASKS, Experts

OFFSETS = (np.array([0., -.65, .8]), np.array([0., .65, .8]))
STATE = mujoco.mjtState.mjSTATE_FULLPHYSICS


class JointWorld:
    def __init__(self, experts=None, *, visual=False):
        bootstrap()
        from assembly_policy.sim.env import BatchedAssemblyEnv
        self.experts = Experts() if experts is None else experts
        self.shadow = BatchedAssemblyEnv(list(TASKS), np.zeros(2, dtype=np.int64), self.experts.config)
        self.visual = visual
        self.cache = OrderedDict()
        self.model = self.data = None
        self.tasks = None
        self.steps = 0

    def _model(self, tasks):
        from assembly_policy.sim.scene import build_xml
        key = tuple(tasks)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        parent = mujoco.MjSpec.from_string('''<mujoco model="precision_assembly_cell">
          <compiler angle="radian" autolimits="true"/>
          <option timestep="0.0005" integrator="implicitfast" cone="elliptic" iterations="50" ls_iterations="20" tolerance="1e-8"/>
          <visual><headlight ambient=".4 .4 .4" diffuse=".5 .5 .5"/></visual>
          <worldbody>
            <geom name="worktable" type="plane" pos="0 0 .8" size="1.5 1.5 .1" contype="1" conaffinity="14" friction=".75 .005 .0001" rgba=".25 .28 .32 1"/>
            <light pos="1 0 3" dir="-.2 0 -1"/>
          </worldbody></mujoco>''')
        children = []
        for arm, task in enumerate(tasks):
            xml = ET.fromstring(build_xml(task, self.shadow.pcfg, visual=self.visual))
            world = xml.find('worldbody')
            table = world.find("geom[@name='table']")
            world.remove(table)  # one table plane, not two duplicate contact surfaces
            child = mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode'))
            frame = parent.worldbody.add_frame(name=f'station_{arm}', pos=OFFSETS[arm])
            parent.attach(child, prefix=f'a{arm}_', frame=frame)
            children.append(child)
        model = parent.compile()
        self.cache[key] = model
        if len(self.cache) > 4:
            self.cache.popitem(last=False)
        return model

    def _indices(self, arm):
        model = self.model
        def joint(name):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'a{arm}_{name}')
        joints = [joint(f'joint{i}') for i in range(1, 8)] + [joint(f'finger_joint{i}') for i in (1, 2)]
        qp = np.array([model.jnt_qposadr[j] for j in joints])
        qv = np.array([model.jnt_dofadr[j] for j in joints])
        part = joint('part_free')
        pq, pv = int(model.jnt_qposadr[part]), int(model.jnt_dofadr[part])
        acts = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'a{arm}_actuator{i}') for i in range(1, 9)])
        return qp, qv, pq, pv, acts

    def rebuild(self, tasks, states):
        self.tasks = tuple(tasks)
        self.model = self._model(tasks)
        self.data = mujoco.MjData(self.model)
        self.indices = [self._indices(arm) for arm in range(2)]
        for arm in range(2):
            self._write_state(arm, states[arm])
        mujoco.mj_forward(self.model, self.data)
        self.shadow.set_states(np.arange(2), np.asarray(states), [TASKS.index(t) for t in tasks])
        self.steps = 0

    def _write_state(self, arm, state):
        e = self.shadow
        qp, qv, pq, pv, acts = self.indices[arm]
        self.data.qpos[qp[:7]] = state[e.sq]
        self.data.qpos[qp[7:]] = state[e.sf]
        self.data.qvel[qv[:7]] = state[e.sqd]
        self.data.qvel[qv[7:]] = state[1 + e.models[0].model.nq + e.idx.finger_dof]
        self.data.qpos[pq:pq+7] = state[e.sp]
        self.data.qpos[pq:pq+3] += OFFSETS[arm]
        self.data.qvel[pv:pv+6] = state[e.spd]
        self.data.ctrl[acts[:7]] = state[e.sq]
        self.data.ctrl[acts[7]] = .04 if np.mean(state[e.sf]) > .036 else e.ecfg.grip_close

    def _read_state(self, arm):
        e = self.shadow
        qp, qv, pq, pv, _ = self.indices[arm]
        row = e.state[arm].copy()
        row[0] = self.data.time
        row[e.sq], row[e.sf] = self.data.qpos[qp[:7]], self.data.qpos[qp[7:]]
        row[e.sqd] = self.data.qvel[qv[:7]]
        row[1 + e.models[0].model.nq + e.idx.finger_dof] = self.data.qvel[qv[7:]]
        row[e.sp] = self.data.qpos[pq:pq+7]
        row[e.sp[:3]] -= OFFSETS[arm]
        row[e.spd] = self.data.qvel[pv:pv+6]
        return row

    def synchronize(self):
        self.shadow.state = np.stack([self._read_state(i) for i in range(2)])

    def step(self, actions):
        e = self.shadow
        actions = np.asarray(actions, dtype=float)
        scale = np.asarray(e.ecfg.action_scale_joints) if e.ecfg.action_scale_joints else e.ecfg.action_scale
        targets = np.clip(e.state[:, e.sq] + scale * actions[:, :7], e.jlo, e.jhi)
        e.grip_cmd = np.where(actions[:, 7] > e.ecfg.grip_deadband, .04,
                             np.where(actions[:, 7] < -e.ecfg.grip_deadband, e.ecfg.grip_close, e.grip_cmd))
        ctrl = np.zeros(self.model.nu)
        for arm, (_, _, _, _, acts) in enumerate(self.indices):
            ctrl[acts[:7]], ctrl[acts[7]] = targets[arm], e.grip_cmd[arm]
        n = int(round(e.ecfg.control_dt / self.model.opt.timestep))
        initial = np.empty(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, initial, STATE)
        trajectory = np.empty((1, n, len(initial)))
        controls = np.broadcast_to(ctrl, (1, n, self.model.nu)).copy()
        before = self.data.time
        rollout.rollout(self.model, self.data, initial[None], controls, state=trajectory)
        mujoco.mj_setState(self.model, self.data, trajectory[0, -1], STATE)
        mujoco.mj_forward(self.model, self.data)
        unstable = not np.isfinite(trajectory).all() or abs(self.data.time - before - e.ecfg.control_dt) > 1e-6
        self.synchronize()
        e.prev_action = actions.copy()
        prop = e._proprio(np.arange(2))
        e.hist = np.concatenate([e.hist[:, 1:], prop[:, None]], axis=1)
        e.t += 1
        self.steps += 1
        violations = []
        for arm, (_, qv, _, _, _) in enumerate(self.indices):
            vel = trajectory[0][:, 1 + self.model.nq + qv[:7]]
            violations.append(bool((np.abs(vel) > e.vlim).any()) or unstable)
        distance = e.assembly_distance()
        return dict(success=(distance < e.ecfg.success_threshold) & ~np.asarray(violations),
                    safety=np.asarray(violations), distance=distance, unstable=unstable)

    def close(self):
        self.shadow.close()
