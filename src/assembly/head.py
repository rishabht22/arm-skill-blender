"""Variable-batch dispatch head, trained by lookahead planning and DAgger.

Training uses a measured duration surrogate. Claims of robot efficiency come
only from separate paired closed-loop physics evaluation, never this surrogate.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .assets import ROOT
from .experts import TASKS
from .profiles import load_profiles, signature as expert_signature
from .scheduling import PlanningModel,batch,critical_time
from .planner import lookahead,choose

FEATURES=29


def inputs(state):
    total=max(1,len(state.operations))
    workloads=[sum(state.means[j.index] for j in state.operations if j.arm==a and state.status[j.index]!=2)/100 for a in (0,1)]
    global_state=[state.time/180,*[float(a>=0) for a in state.active],float((state.status==2).mean()),*workloads,
                  float(sum(j.release>state.time for j in state.operations))/total]
    result=[]
    for j in state.operations:
        critical=critical_time(state,j.index)
        local=[*np.eye(len(TASKS))[TASKS.index(j.skill)],float(j.arm),*np.eye(3)[state.status[j.index]],
               state.means[j.index]/20,state.stds[j.index]/10,max(0.,j.release-state.time)/10,
               1. if j.predecessor<0 else float(state.status[j.predecessor]==2),critical/40,
               (state.time-state.started[j.index])/20 if state.status[j.index]==1 else 0.]
        result.append(local+global_state)
    x=np.asarray(result,dtype=np.float32)
    if x.shape[1]!=FEATURES:raise RuntimeError(f'Unexpected dispatch feature shape {x.shape}')
    return x


class DispatchHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(FEATURES,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.score=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,1))
        self.wait=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))

    def forward(self,x,present):
        h=self.embed(x)
        context=(h*present[:,:,None]).sum(1)/present.sum(1,keepdim=True).clamp_min(1)
        scores=self.score(torch.cat([h,context[:,None].expand(-1,x.shape[1],-1)],dim=-1)).squeeze(-1)
        return torch.cat([scores,self.wait(context)],dim=-1)

    @torch.no_grad()
    def act(self,state):
        x=torch.as_tensor(inputs(state))[None]
        scores=self(x,torch.ones(x.shape[:2],dtype=torch.bool))[0]
        mask=torch.as_tensor(state.mask())
        return int(scores.masked_fill(~mask,-1e9).argmax())


def contract(profiles):
    folder=Path(__file__).parent
    operative={name:{key:value[key] for key in ('mean_seconds','std_seconds','success_rate','trials')}
               for name,value in profiles['skills'].items()}
    return dict(experts=expert_signature(),profiles_sha256=hashlib.sha256(json.dumps(operative,sort_keys=True).encode()).hexdigest(),
                features=FEATURES,method='lookahead_distillation_dagger',
                sources={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('scheduling.py','planner.py','head.py')})


def train(*,problems=180,epochs=40,rounds=2):
    if min(problems,epochs,rounds)<1:raise ValueError('Training budgets must be positive')
    torch.set_num_threads(1);torch.manual_seed(0)
    profiles=load_profiles();initial=contract(profiles)
    output=ROOT/'outputs/assembly'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True)
    (output/'profiles.json').write_text(json.dumps(profiles,indent=2)+'\n')
    model=DispatchHead()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    records=[];start=time.monotonic()
    for round_index in range(rounds):
        for seed in range(problems):
            count=2+(seed%5)
            state=PlanningModel(batch(seed,count),profiles,seed=seed+round_index*problems)
            while not state.done:
                teacher=lookahead(state,width=8)
                records.append((inputs(state),state.mask(),teacher))
                action=teacher if round_index==0 else model.act(state)
                state.step(action)
            if seed%30==0:print('planner data',round_index,seed,'examples',len(records),flush=True)
        rng=np.random.default_rng(round_index)
        for epoch in range(epochs):
            order=rng.permutation(len(records))
            for first in range(0,len(order),64):
                selected=[records[i] for i in order[first:first+64]]
                length=max(len(x[0]) for x in selected)
                x=np.zeros((len(selected),length,FEATURES),dtype=np.float32)
                present=np.zeros((len(selected),length),dtype=bool)
                masks=np.zeros((len(selected),length+1),dtype=bool);labels=[]
                for row,(features,mask,label) in enumerate(selected):
                    n=len(features);x[row,:n]=features;present[row,:n]=True;masks[row,:n]=mask[:-1];masks[row,-1]=mask[-1]
                    labels.append(length if label==n else label)
                logits=model(torch.as_tensor(x),torch.as_tensor(present)).masked_fill(~torch.as_tensor(masks),-1e9)
                loss=nn.functional.cross_entropy(logits,torch.as_tensor(labels))
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        validation=proxy_benchmark(model,profiles,range(10000,10040),kits=4)
        (output/f'validation_{round_index}.json').write_text(json.dumps(validation,indent=2)+'\n')
        score=np.mean([r['elapsed_s'] if r['success'] else 180. for r in validation if r['scheduler']=='head'])
        if round_index==0 or score<best:
            best=score;torch.save(model.state_dict(),output/'best_head.pt')
        print('DAgger round',round_index,'validation mean seconds',score,flush=True)
    metadata=dict(contract=initial,problems=problems,epochs=epochs,rounds=rounds,examples=len(records),
                  trainable_parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-start,
                  training_backend='measured duration surrogate; no robot expert gradients')
    if initial!=contract(profiles):raise RuntimeError('Assembly contract changed during head training')
    (output/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output.parent/'latest.json').write_text(json.dumps(dict(run=output.name),indent=2)+'\n')
    print('Saved head',output,flush=True)
    return output


def load(path=None):
    if path is None:
        latest=ROOT/'outputs/assembly/latest.json'
        if not latest.is_file():raise FileNotFoundError('Train an assembly head first')
        path=latest.parent/json.loads(latest.read_text())['run']/'best_head.pt'
    path=Path(path)
    metadata=json.loads((path.parent/'run.json').read_text())
    if metadata['contract']!=contract(load_profiles()):raise RuntimeError('Head/expert/profile contract mismatch')
    model=DispatchHead();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval().requires_grad_(False)
    return model


def proxy_benchmark(model,profiles,seeds,kits=4):
    rows=[]
    for seed in seeds:
        operations=batch(seed,kits)
        for mode in ('sequential','fifo','critical_path','johnson','head'):
            state=PlanningModel(operations,profiles,seed=seed)
            while not state.done:state.step(model.act(state) if mode=='head' else choose(state,mode))
            rows.append(dict(seed=seed,scheduler=mode,backend='duration_surrogate',**state.report()))
    return rows
