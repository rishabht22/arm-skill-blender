"""Publishers' trajectories only; semantic boundaries are weak annotations."""
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np

from ..common import ROOT, sha256, write_json
from ..data import split_episodes
from .program import Node, MAX_NODES, features, source_program, robot_state

RAW=ROOT/'data/robofactory'
DATA=ROOT/'data/processed/multitask.npz'
RUN=ROOT/'outputs/cell'
REVISION='63f8a85'
TASKS={
 'PickMeat':'7ede3079f213921c31df299ee1f76fd9db1c2220fcdbba3a49942ff1a318e2a5',
 'StackCube':'57d6b7beac139fbc6a4349274de7c5ab778ef04aca529c5d69a5e9692cf9228b',
 'PassShoe':'13d8264fa8809216a0db271e8e5933239910d43d062c3412d6d8d6a8c705fb72',
 'PlaceFood':'1606c85f5fc91bc35a58674dfb5e3a91bfbaf1be7028ef5f61a72bc121e79de2',
 'TwoRobotsStackCube':'76641c6a8bb944fb5eacfec51bd5174f34553d11851aae89383044f323130510',
}


def download(raw=RAW):
    raw=Path(raw);raw.mkdir(parents=True,exist_ok=True)
    def fetch(task):
        records=[]
        for ext in ('json','h5'):
            p=raw/f'{task}.{ext}'
            if not p.exists():
                print('Downloading',p.name,flush=True)
                partial=p.with_suffix(p.suffix+'.part')
                urllib.request.urlretrieve(f'https://huggingface.co/datasets/FACEONG/RoboFactory_Dataset/resolve/{REVISION}/{task}/{p.name}',partial)
                if ext=='h5' and sha256(partial)!=TASKS[task]:raise ValueError('Upstream checksum mismatch')
                partial.replace(p)
            digest=sha256(p)
            if ext=='h5' and digest!=TASKS[task]:raise ValueError('Local checksum mismatch: '+str(p))
            records.append({'file':p.name,'bytes':p.stat().st_size,'sha256':digest})
        print('Verified published',task,flush=True)
        return {'task':task,'files':records}
    with ThreadPoolExecutor(max_workers=3) as pool: records=list(pool.map(fetch,TASKS))
    write_json(raw/'manifest.json',{'source':'FACEONG/RoboFactory_Dataset','revision':REVISION,'tasks':records})
    return records


def actions(group):
    if isinstance(group['actions'],h5py.Dataset):return [group['actions'][:]]
    return [group['actions'][a][:] for a in sorted(group['actions'])]


def _runs(mask):
    d=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(d==1),np.flatnonzero(d==-1)))


def intervals(task, arrays):
    result=[]
    for a,commands in enumerate(arrays):
        gripper=commands[:,-1]
        delta=np.diff(gripper,prepend=gripper[0])
        closing=_runs(delta<-.025)
        opening=_runs(delta>.025)
        if not closing:raise ValueError('No gripper closing interval')
        c0,c1=closing[0]
        speed=np.r_[0,np.linalg.norm(np.diff(commands[:,:7],axis=0),axis=1)]
        moving=np.flatnonzero((speed>.003)&(np.arange(len(speed))<c0))
        start=int(moving[0]) if len(moving) else 0
        opening=[r for r in opening if r[0]>c1]
        release=opening[0] if opening else None
        limit=int(release[0]) if release else len(commands)-1
        # First substantial post-grasp motion is the source expert's lift.
        lift_runs=[r for r in _runs(speed>.003) if r[0]>=c1 and r[0]<limit and r[1]-r[0]>=3]
        end=int(lift_runs[0][1]+1) if lift_runs else limit
        if task=='PickMeat':end=len(commands)
        if task=='PlaceFood' and a==1:
            result.append((1,start,min(end,len(commands))))
            continue
        node_indices={
          'PickMeat':{0:(0,None)},'StackCube':{0:(0,1)},
          'PassShoe':{0:(0,1),1:(2,3)},
          'PlaceFood':{0:(0,2)},'TwoRobotsStackCube':{0:(0,2),1:(1,3)},
        }
        pick,place=node_indices[task][a]
        result.append((pick,start,min(end,len(commands))))
        if place is not None:
            if release is None:raise ValueError('Missing release interval')
            place_start=min(end,len(commands)-1)
            if task=='TwoRobotsStackCube' and a==1:
                carry=[r for r in _runs(speed>.003) if r[0]>=end and r[1]-r[0]>=4]
                if carry:place_start=int(carry[0][0])
            result.append((place,place_start,int(release[1])))
    if task=='TwoRobotsStackCube':
        receiver=next(s for n,s,e in result if n==3)
        result=[(n,s,max(e,receiver)) if n==2 else (n,s,e) for n,s,e in result]
    if task=='PlaceFood':
        food_ready=next(e for n,s,e in result if n==0)
        result=[(n,s,min(e,food_ready)) if n==1 else (n,s,e) for n,s,e in result]
    # Handoff place includes the source arm's retreat, up to receiver approach.
    if task=='PassShoe':
        receive=next(s for n,s,e in result if n==2)
        result=[(n,s,max(e,receive)) if n==1 else (n,s,e) for n,s,e in result]
    return result


def prepare(raw=RAW,output=DATA):
    raw,output=Path(raw),Path(output)
    xx,mm,yy,ids,tt,task_ids=[],[],[],[],[],[]
    audit=[];splits={k:[] for k in ['train','validation','test']};sources=[];durations={k:[] for k in ['pick','place','open_lid']}
    count=frames=0
    for task_index,task in enumerate(TASKS):
        metadata=json.loads((raw/f'{task}.json').read_text())
        episode_ids=[e['episode_id'] for e in metadata['episodes'] if e.get('success') is True]
        partitions=split_episodes(episode_ids,seed=7)
        local_partition={i:k for k,values in partitions.items() for i in values}
        digest=sha256(raw/f'{task}.h5')
        if digest!=TASKS[task]:raise ValueError('Incorrect published data file '+task)
        sources.append({'task':task,'sha256':digest,'episodes':len(episode_ids)})
        with h5py.File(raw/f'{task}.h5') as dataset:
            for eid in episode_ids:
                key=task_index*10000+eid
                group=dataset[f'traj_{eid}']
                aa=actions(group)
                qs=[];vs=[]
                for a in range(len(aa)):
                    agent=group['obs/agent'] if len(aa)==1 else group[f'obs/agent/panda-{a}']
                    qs.append(agent['qpos'][:]);vs.append(agent['qvel'][:])
                nodes=source_program(task)
                spans=intervals(task,aa)
                # Use source start/end observations; no procedural scheduler labels.
                events=sorted({t for _,s,e in spans for t in (s,e)})
                episode_rows=[]
                for time in events:
                    status=np.array([0]*len(nodes))
                    target=np.array([MAX_NODES,MAX_NODES])
                    for index,start,end in spans:
                        status[index]=2 if time>=end else (1 if time>start else 0)
                        if time==start:target[nodes[index].arm]=index
                    state=np.zeros((2,18),dtype=np.float32)
                    for a in range(len(aa)):
                        t=min(time,len(qs[a])-1)
                        state[a]=robot_state(qs[a][t],vs[a][t])
                    x,mask=features(nodes,status,state)
                    # Ignore recorded start-up/settling gaps, but preserve waits
                    # at real operation completions when the partner is still busy.
                    if (target==MAX_NODES).all() and all(status==2):continue
                    if any(not mask[a,target[a]] for a in range(2)):
                        raise ValueError(f'Overlapping source operations {task}/{eid}/{time}')
                    xx.append(x);mm.append(mask);yy.append(target);ids.append(key);tt.append(time);task_ids.append(task_index)
                    episode_rows.append(time)
                splits[local_partition[eid]].append(key)
                count+=1;frames+=len(aa[0])
                audit.append({'task':task,'episode':eid,'key':key,'intervals':spans})
                if local_partition[eid]=='train':
                    for n,s,e in spans:durations[nodes[n].op].append((e-s)*.05)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,features=np.asarray(xx),mask=np.asarray(mm),labels=np.asarray(yy),episode=np.asarray(ids),frame=np.asarray(tt),task=np.asarray(task_ids))
    report={'source':'published RoboFactory recordings only','revision':REVISION,'raw_files':sources,
            'episodes':count,'source_frames':frames,'decision_events':len(xx),'splits':splits,'tasks':list(TASKS),
            'annotation':'Task-specific weak semantic labels inferred from published expert joint/gripper trajectories. Goal graphs are hand annotated. No generated expert trajectories.',
            'augmentation':'Candidate/node permutation and arm exchange in the abstract skill-graph representation only; do not count as new source episodes.',
            'feature_schema':'Goal graph, executor status, both arms measured qpos/qvel; no task ID or future action inputs',
            'unique_observations':len(np.unique(np.asarray(xx).reshape(len(xx),-1),axis=0)),
            'skill_duration_medians_s':{k:float(np.median(v)) for k,v in durations.items()},'audit':audit}
    write_json(output.with_suffix('.json'),report)
    print(f'Prepared {count} published episodes, {frames} recorded frames, {len(xx)} decisions',flush=True)
    return report
