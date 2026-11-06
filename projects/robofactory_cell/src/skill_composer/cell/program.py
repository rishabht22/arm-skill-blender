"""Goal programs and observations shared by offline recordings and online execution."""
from dataclasses import dataclass, asdict
import numpy as np

OPS = ('pick', 'place', 'open_lid')
MAX_NODES = 20
FEATURES = 54

@dataclass
class Node:
    op: str
    arm: int
    obj: str = 'part0'
    target: str = 'goal0'
    deps: tuple = ()
    stack: bool = False
    container: bool = False

    def to_dict(self):
        return asdict(self)


def source_program(task):
    if task == 'PickMeat':
        return [Node('pick', 0)]
    if task == 'StackCube':
        return [Node('pick', 0), Node('place', 0, deps=(0,), stack=True)]
    if task == 'PassShoe':
        return [Node('pick', 0), Node('place', 0, target='dock0', deps=(0,)),
                Node('pick', 1, target='dock0', deps=(1,)), Node('place', 1, deps=(2,))]
    if task == 'PlaceFood':
        return [Node('pick', 0), Node('open_lid', 1, obj='lid'),
                Node('place', 0, deps=(0, 1), container=True)]
    if task == 'TwoRobotsStackCube':
        return [Node('pick', 0), Node('pick', 1, obj='part1'),
                Node('place', 0, deps=(0,)),
                Node('place', 1, obj='part1', target='goal1', deps=(1, 2), stack=True)]
    raise ValueError(task)


def composed_program(name='relay_stack', parts=3):
    """Evaluation-only compositions. Never used to create training labels."""
    if name == 'parallel_stack':
        return source_program('TwoRobotsStackCube')
    if name == 'handoff':
        return source_program('PassShoe')
    if name == 'container':
        return source_program('PlaceFood')
    if name not in ('relay_stack', 'relay_container', 'relay_tower'):
        raise ValueError(name)
    if not 2 <= parts <= 4:
        raise ValueError('Use two to four parts')
    nodes=[]
    staging=[]
    left_previous=None
    right_previous=None
    # Each source part must cross the workspace before the receiver can use it.
    for p in range(parts):
        pick=len(nodes)
        nodes.append(Node('pick', 0, obj=f'part{p}', deps=(() if left_previous is None else (left_previous,))))
        stage=len(nodes)
        nodes.append(Node('place', 0, obj=f'part{p}', target=f'dock{p}', deps=(pick,)))
        staging.append(stage)
        left_previous=stage
    lid=None
    if name == 'relay_container':
        lid=len(nodes)
        nodes.append(Node('open_lid', 0, obj='lid', deps=tuple(staging)))
    for p in range(parts):
        pick=len(nodes)
        dependencies=[staging[p]]
        if right_previous is not None: dependencies.append(right_previous)
        nodes.append(Node('pick', 1, obj=f'part{p}', target=f'dock{p}', deps=tuple(dependencies)))
        deps=[pick]
        if lid is not None: deps.append(lid)
        nodes.append(Node('place', 1, obj=f'part{p}', target=f'goal{p}', deps=tuple(deps),
                          stack=p>0, container=lid is not None))
        right_previous=len(nodes)-1
    return nodes


def features(nodes, status, proprio=None):
    """No task ID, node index, future expert actions, or demonstration clock.

    Status: 0=pending, 1=running, 2=done. Dependency progress is supplied as a
    goal-program observation, not an action mask. The policy learns when to wait.
    """
    status=np.asarray(status)
    x=np.zeros((2, MAX_NODES, FEATURES), dtype=np.float32)
    mask=np.zeros((2, MAX_NODES+1), dtype=bool)
    mask[:, -1]=True  # WAIT is always available.
    busy=[any(n.arm==a and status[i]==1 for i,n in enumerate(nodes)) for a in range(2)]
    for i,node in enumerate(nodes):
        dep_done=sum(status[d]==2 for d in node.deps)
        for a in range(2):
            row=x[a,i]
            row[OPS.index(node.op)]=1
            row[3+int(status[i])]=1
            row[6]=node.arm==a
            row[7]=dep_done/max(1,len(node.deps)) if node.deps else 1
            row[8]=len(node.deps)-dep_done
            row[9]=all(status[d]==2 for d in node.deps)
            row[10]=busy[a]
            row[11]=busy[1-a]
            row[12]=sum(status==2)/len(nodes)
            row[13]=node.stack
            row[14]=node.container
            row[15]=node.target.startswith('dock')
            row[16]=sum(i in other.deps for other in nodes)/4
            if proprio is not None:
                row[17:35]=proprio[a]
                row[35:53]=proprio[1-a]
            row[53]=1  # Real node; padded slots remain zero.
            mask[a,i]=node.arm==a and status[i]==0 and not busy[a]
    return x,mask


def robot_state(qpos,qvel):
    scale=np.array([3,2,3,3,3,4,3,.04,.04],dtype=np.float32)
    return np.concatenate([np.asarray(qpos)/scale,np.clip(np.asarray(qvel)/2,-2,2)],axis=-1).astype(np.float32)
