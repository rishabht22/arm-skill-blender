"""Scheduling baselines and bounded lookahead using expected skill times only."""
import copy
import numpy as np
from .scheduling import critical_time


def clone(state):
    other=copy.copy(state)
    for field in ('status','active','started','busy_time','durations','ends'):
        setattr(other,field,getattr(state,field).copy())
    other.events=list(state.events)
    return other


def lower_bound(state):
    remaining=np.zeros(2)
    for arm,index in enumerate(state.active):
        if index>=0:remaining[arm]+=max(0.,state.ends[arm]-state.time)
    for job in state.operations:
        if state.status[job.index]==0:remaining[job.arm]+=state.means[job.index]
    return state.time+max(remaining)


def lookahead(state,width=12):
    initial=state.planning_state() if hasattr(state,'planning_state') else state.forecast()
    frontier=[(initial,None)]
    # Seed feasible complete plans, so beam pruning cannot discard every
    # reasonable solution on larger re-entrant work orders.
    finished=[]
    for mode in ('fifo','shortest','critical_path','johnson'):
        baseline=clone(initial);first=None
        while not baseline.done:
            action=choose(baseline,mode)
            if first is None:first=action
            baseline.step(action)
        if baseline.success:finished.append((baseline.time,first))
    for _ in range(4*len(initial.operations)):
        candidates=[]
        for s,first in frontier:
            if s.success:
                finished.append((s.time,first));continue
            for action in np.flatnonzero(s.mask()):
                child=clone(s);child.step(int(action))
                choice=int(action) if first is None else first
                if child.success:finished.append((child.time,choice))
                elif not child.failure:candidates.append((child,choice))
        if not candidates:break
        candidates.sort(key=lambda item:(lower_bound(item[0]),item[0].time))
        frontier=candidates[:width]
    if finished:return min(finished,key=lambda item:item[0])[1]
    raise RuntimeError('Lookahead could not find a feasible schedule')


def choose(state,mode='fifo'):
    ready=state.ready()
    if mode=='lookahead':return lookahead(state)
    if mode=='sequential':
        if (state.active>=0).any():return len(state.operations)
        return ready[0] if ready else len(state.operations)
    if not ready:return len(state.operations)
    if mode=='fifo':return min(ready)
    if mode=='shortest':return min(ready,key=lambda i:state.means[i])
    if mode=='critical_path':
        return max(ready,key=lambda i:critical_time(state,i))
    if mode=='johnson':
        kits=len(state.operations)//4
        first=[sum(state.means[4*k+s] for s in (0,2)) for k in range(kits)]
        second=[sum(state.means[4*k+s] for s in (1,3)) for k in range(kits)]
        early=[k for k in range(kits) if first[k]<=second[k]]
        late=[k for k in range(kits) if k not in early]
        order=sorted(early,key=lambda k:first[k])+sorted(late,key=lambda k:-second[k])
        rank={kit:i for i,kit in enumerate(order)}
        return min(ready,key=lambda i:rank[i//4])
    raise ValueError('Unknown scheduler')
