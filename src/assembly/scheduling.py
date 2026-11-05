"""Event-driven two-station kit production: measured planning model and physics.

A kit has one assembly operation at each station, with work-order precedence.
The precedence is a production constraint, not an asserted physical handover.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from .experts import SKILLS, TASKS
from .service import Job, ProductionCell, loose_state
from .assets import ROOT


@dataclass(frozen=True)
class Operation:
    index: int
    arm: int
    skill: str
    predecessor: int
    release: float


def batch(seed, kits=4):
    rng = np.random.default_rng(seed)
    result = []
    for k in range(kits):
        release = 0. if k == 0 else float(rng.integers(0, 8))
        # Re-entrant kit route: insert a keyed/round part, seat a gear, mate a
        # connector, then finish the threaded/connector component. Different
        # kits compete for the two stations at different stages.
        choices = (SKILLS[0][:4], SKILLS[1][:3], SKILLS[0][4:], SKILLS[1][3:])
        for stage, skills in enumerate(choices):
            index = 4*k + stage
            result.append(Operation(index, stage % 2, str(rng.choice(skills)),
                                    -1 if stage == 0 else index-1, release))
    return result


def critical_time(state, index):
    following=[j.index for j in state.operations if j.predecessor==index and state.status[j.index]!=2]
    return state.means[index]+max((critical_time(state,i) for i in following),default=0.)


def materialize(operations, seed):
    key = hashlib.sha256(json.dumps([j.__dict__ for j in operations],sort_keys=True).encode()).hexdigest()[:12]
    cache = ROOT / 'data/assembly/batches' / f'{seed}_{key}.npz'
    if cache.is_file():
        with np.load(cache) as data:
            states = data['states']
    else:
        states = np.stack([loose_state(j.skill, seed*997 + j.index*71) for j in operations])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, states=states)
    return [Job(j.index, j.arm, j.skill, states[j.index], () if j.predecessor < 0 else (j.predecessor,), j.release)
            for j in operations]


class SchedulingState:
    def __init__(self, operations, profiles):
        self.operations, self.profiles = operations, profiles
        self.means = np.array([profiles['skills'][j.skill]['mean_seconds'] for j in operations])
        self.stds = np.array([profiles['skills'][j.skill]['std_seconds'] for j in operations])
        self.time = 0.
        self.status = np.zeros(len(operations), dtype=np.int8) # waiting/running/complete
        self.active = np.full(2, -1, dtype=int)
        self.started = np.zeros(len(operations))
        self.busy_time = np.zeros(2)
        self.failure = None
        self.events = []
        self.decisions = 0

    @property
    def done(self):
        return bool((self.status == 2).all() or self.failure)

    @property
    def success(self):
        return bool((self.status == 2).all() and not self.failure)

    def ready(self):
        return [j.index for j in self.operations if self.status[j.index] == 0 and self.active[j.arm] < 0
                and j.release <= self.time + 1e-8
                and (j.predecessor < 0 or self.status[j.predecessor] == 2)]

    def mask(self):
        mask = np.zeros(len(self.operations)+1, dtype=bool)
        mask[self.ready()] = True
        # Wait can be useful while a high-priority kit is about to become ready.
        mask[-1] = bool((self.active >= 0).any() or any(j.release > self.time + 1e-8 and self.status[j.index] == 0 for j in self.operations))
        return mask

    def features(self):
        values = []
        for j in self.operations:
            remaining_chain = self.means[j.index]
            if j.arm == 0:
                remaining_chain += self.means[j.index+1]
            values += list(np.eye(len(TASKS))[TASKS.index(j.skill)])
            values += [float(j.arm), *np.eye(3)[self.status[j.index]],
                       self.means[j.index]/20, self.stds[j.index]/10,
                       max(0.,j.release-self.time)/10,
                       1. if j.predecessor < 0 else float(self.status[j.predecessor] == 2),
                       remaining_chain/40, (self.time-self.started[j.index])/20 if self.status[j.index] == 1 else 0.]
        values += [self.time/180, *[float(a >= 0) for a in self.active], *list(self.status == 2)]
        return np.asarray(values,dtype=np.float32)

    def report(self):
        return dict(success=self.success, makespan_s=self.time if self.success else None,
                    elapsed_s=self.time, failure=self.failure,
                    utilization=(self.busy_time/max(self.time,1e-8)).tolist(), events=self.events,
                    completed=int((self.status == 2).sum()), decisions=self.decisions)


class PlanningModel(SchedulingState):
    """Cheap duration surrogate; all reported robot speedups require physics tests."""
    def __init__(self, operations, profiles, seed=0, stochastic=True):
        super().__init__(operations,profiles)
        rng=np.random.default_rng(seed+100000)
        self.durations = np.maximum(.5, rng.normal(self.means,self.stds)) if stochastic else self.means.copy()
        self.ends = np.full(2,np.inf)
        self._settle()

    def forecast(self):
        other = copy.deepcopy(self)
        other.durations = self.means.copy()
        # Do not let a planner or head inspect hidden sampled completion times.
        for arm,index in enumerate(other.active):
            if index >= 0:
                other.ends[arm] = max(other.time+.1,other.started[index]+other.means[index])
        return other

    def _advance(self):
        future=[float(t) for t in self.ends if np.isfinite(t) and t>self.time+1e-8]
        future += [j.release for j in self.operations if self.status[j.index] == 0 and j.release>self.time+1e-8]
        if not future:
            if not self.success:self.failure='scheduler deadlock'
            return
        target=min(future)
        for arm in (0,1):
            if self.active[arm]>=0:self.busy_time[arm]+=target-self.time
        self.time=target
        for arm in (0,1):
            if self.active[arm]>=0 and self.ends[arm]<=self.time+1e-8:
                self.status[self.active[arm]]=2;self.active[arm]=-1;self.ends[arm]=np.inf

    def _settle(self):
        while not self.done and not self.ready():self._advance()

    def step(self,action):
        if self.done:raise RuntimeError('Batch is finished')
        if not 0 <= action <= len(self.operations) or not self.mask()[action]:raise ValueError('Illegal dispatch')
        self.decisions+=1
        if action==len(self.operations):self._advance()
        else:
            j=self.operations[action];self.status[action]=1;self.active[j.arm]=action
            self.started[action]=self.time;self.ends[j.arm]=self.time+self.durations[action]
            self.events.append(dict(time=self.time,job=action,arm=j.arm,skill=j.skill))
        self._settle()
        if self.time>180 or self.decisions>4*len(self.operations):self.failure='scheduler budget exhausted'


class PhysicsBatch(SchedulingState):
    def __init__(self, operations, profiles, seed, *, cell=None, on_frame=None):
        super().__init__(operations,profiles)
        self.cell=ProductionCell() if cell is None else cell
        self.owned=cell is None
        self.jobs=materialize(operations,seed)
        self.cell.reset(self.jobs)
        self.on_frame=on_frame
        self.consumed=0
        self._settle()

    def _advance(self,until_choice=False):
        previous_completed=self.consumed
        releases=[j.release for j in self.operations if self.status[j.index]==0 and j.release>self.time+1e-8]
        next_release=min(releases,default=np.inf)
        while not self.done:
            self.cell.step()
            self.time=self.cell.time
            self.busy_time=self.cell.busy_time.copy()
            self.failure=self.cell.failed
            for event in self.cell.completed[self.consumed:]:
                index=event['job'];self.status[index]=2;self.active[event['arm']]=-1
            self.consumed=len(self.cell.completed)
            if self.on_frame:self.on_frame(self.cell.world)
            if self.time>180:self.failure='production deadline exceeded'
            if until_choice and self.ready():break
            if not until_choice and (self.consumed>previous_completed or self.time>=next_release-1e-8):break

    def _settle(self):
        if not self.done and not self.ready():self._advance(until_choice=True)

    def step(self,action):
        if self.done:raise RuntimeError('Batch is finished')
        if not 0<=action<=len(self.operations) or not self.mask()[action]:raise ValueError('Illegal dispatch')
        self.decisions+=1
        if action==len(self.operations):
            self._advance()
        else:
            j=self.operations[action]
            self.cell.start(self.jobs[action])
            self.status[action]=1;self.active[j.arm]=action;self.started[action]=self.time
            self.events.append(dict(time=self.time,job=action,arm=j.arm,skill=j.skill))
            if self.cell.failed:self.failure=self.cell.failed
        self._settle()
        if self.decisions>4*len(self.operations):self.failure='scheduler budget exhausted'

    def planning_state(self):
        proxy=PlanningModel(self.operations,self.profiles,stochastic=False)
        proxy.time=self.time;proxy.status=self.status.copy();proxy.active=self.active.copy()
        proxy.started=self.started.copy();proxy.busy_time=self.busy_time.copy();proxy.events=list(self.events)
        for arm,index in enumerate(proxy.active):
            if index>=0:proxy.ends[arm]=max(proxy.time+.1,proxy.started[index]+proxy.means[index])
        return proxy

    def close(self):
        self.cell.world.experts.assert_frozen()
        if self.owned:self.cell.close()
