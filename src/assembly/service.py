"""Workpiece exchange, explicit entry/exit motion, and frozen neural assembly."""
from dataclasses import dataclass
import math

import numpy as np

from .experts import SKILLS, TASKS
from .world import JointWorld
from .assets import VENDOR


@dataclass
class Job:
    index: int
    arm: int
    skill: str
    state: np.ndarray
    predecessors: tuple = ()
    release: float = 0.


def loose_state(skill, seed):
    from assembly_policy.sim.resets import ResetGenerator
    from assembly_policy.sim.scene import PhysicsConfig
    from assembly_policy.sim.env import EnvConfig
    generator = ResetGenerator(skill, PhysicsConfig(fine_families=EnvConfig.fine_families), seed=seed)
    return generator.generate('loose', 1)[0][0]


class ProductionCell:
    def __init__(self, experts=None, *, visual=False):
        self.world = JointWorld(experts, visual=visual)
        self.phases = ['idle', 'idle']
        self.jobs = [None, None]
        self.goal = [None, None]
        self.start_q = [None, None]
        self.phase_age = np.zeros(2, dtype=int)
        self.phase_length = np.ones(2, dtype=int)
        self.elapsed = np.zeros(2)
        self.failed = None
        self.completed = []
        self.time = 0.
        self.busy_time = np.zeros(2)
        self.energy = np.zeros(2)

    def reset(self, jobs):
        from assembly_policy.sim.robot import HOME_Q
        self.all_jobs = jobs
        first = [next(j for j in jobs if j.arm == a) for a in (0, 1)]
        states = np.stack([j.state.copy() for j in first])
        e = self.world.shadow
        states[:, e.sq] = HOME_Q
        states[:, e.sf] = .04
        states[:, 1 + e.models[0].model.nq:] = 0.
        self.world.rebuild([j.skill for j in first], states)
        self.phases, self.jobs = ['idle', 'idle'], [None, None]
        self.phase_age[:] = 0
        self.elapsed[:] = 0
        self.busy_time[:] = 0
        self.energy[:] = 0
        self.failed, self.completed, self.time = None, [], 0.

    def _motion(self, arm, target_position, target_quat, phase):
        from assembly_policy.sim.robot import solve_ik
        w, e = self.world, self.world.shadow
        tm = e.models[TASKS.index(w.tasks[arm])]
        q = e.state[arm, e.sq].copy()
        goal, ok = solve_ik(tm.model, tm.datas[0], tm.idx, target_position, target_quat,
                            q_init=q, restarts=5)
        if not ok:
            self.failed = f'IK entry/exit failed: arm {arm} {phase}'
            return
        seconds = max(.8, 2 * float(np.max(np.abs(goal - q) / e.vlim)))
        self.goal[arm], self.start_q[arm] = goal, q
        self.phase_length[arm] = max(1, int(math.ceil(seconds / e.ecfg.control_dt)))
        self.phase_age[arm] = 0
        self.phases[arm] = phase

    def start(self, job):
        from assembly_policy.sim.robot import mat_to_quat
        if job.skill not in SKILLS[job.arm]:
            raise ValueError('Skill is unavailable for this arm')
        if self.phases[job.arm] != 'idle':
            raise ValueError('Station is already busy')
        w, e, arm = self.world, self.world.shadow, job.arm
        # New incoming workpiece only; robot joints and the other station persist.
        states = e.state.copy()
        states[arm, e.sp] = job.state[e.sp]
        states[arm, e.spd] = job.state[e.spd]
        tasks = list(w.tasks); tasks[arm] = job.skill
        history, previous, times, grip = e.hist.copy(), e.prev_action.copy(), e.t.copy(), e.grip_cmd.copy()
        clock, steps = w.data.time, w.steps
        w.rebuild(tasks, states)
        w.data.time, w.steps = clock, steps
        e.state[:, 0] = clock
        e.hist[:], e.prev_action[:], e.t[:], e.grip_cmd[:] = history, previous, times, grip
        e.t[arm] = 0
        e.prev_action[arm] = 0
        e.hist[arm] = e._proprio([arm])[0]
        self.jobs[arm], self.elapsed[arm] = job, 0.
        # Visible, timed open-gripper approach. The actual grasp/reorientation/
        # insertion/threading is entirely the released neural policy afterward.
        down = np.array([0., 1., 0., 0.])
        point = e.state[arm, e.sp[:3]] + [0., 0., .13]
        self._motion(arm, point, down, 'approach')

    def step(self):
        from assembly_policy.sim.robot import mat_to_quat
        w, e = self.world, self.world.shadow
        actions = w.experts(e)
        for arm in range(2):
            phase = self.phases[arm]
            if phase == 'idle':
                actions[arm] = 0.
            elif phase in ('approach', 'retract'):
                fraction = min(1., (self.phase_age[arm] + 1) / self.phase_length[arm])
                fraction = fraction * fraction * (3 - 2 * fraction)
                target = self.start_q[arm] + fraction * (self.goal[arm] - self.start_q[arm])
                scale = np.asarray(e.ecfg.action_scale_joints)
                actions[arm, :7] = np.clip((target - e.state[arm, e.sq]) / scale, -4., 4.)
                actions[arm, 7] = 1.
        info = w.step(actions)
        dt = e.ecfg.control_dt
        self.time += dt
        for arm in range(2):
            phase = self.phases[arm]
            if phase == 'idle':
                continue
            self.busy_time[arm] += dt
            self.elapsed[arm] += dt
            self.phase_age[arm] += 1
            # Track actual actuator effort; rendering never contributes to time.
            acts = w.indices[arm][-1]
            self.energy[arm] += float(np.square(w.data.actuator_force[acts]).sum()) * dt
            if info['safety'][arm]:
                self.failed = f'safety violation: arm {arm}, {self.jobs[arm].skill}, {phase}'
            if phase == 'approach' and self.phase_age[arm] >= self.phase_length[arm] + 5:
                self.phases[arm] = 'expert'
                self.phase_age[arm] = 0
            elif phase == 'expert':
                if info['success'][arm]:
                    fk = e._fk([arm])
                    self._motion(arm, fk['tcp_p'][0] + [0., 0., .12], mat_to_quat(fk['tcp_R'][0]), 'retract')
                elif self.phase_age[arm] * dt >= e.ecfg.horizon_s:
                    self.failed = f'assembly timeout: arm {arm}, {self.jobs[arm].skill}'
            elif phase == 'retract' and self.phase_age[arm] >= self.phase_length[arm] + 10:
                # A transient near-goal while still gripping is not enough.
                if info['distance'][arm] > .002:
                    self.failed = f'assembly moved after release: {self.jobs[arm].skill}, error={info["distance"][arm]:.4f}m'
                else:
                    self.completed.append(dict(job=self.jobs[arm].index, arm=arm, skill=self.jobs[arm].skill,
                                               seconds=float(self.elapsed[arm]), finish=self.time,
                                               error_m=float(info['distance'][arm])))
                    self.phases[arm] = 'idle'
                    self.jobs[arm] = None
        return info

    def close(self):
        self.world.close()
