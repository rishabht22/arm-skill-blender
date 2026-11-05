"""Permutation-equivariant skill-graph behavior cloning, with no task-ID lookup."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from ..common import sha256,write_json
from .dataset import DATA,RUN
from .program import FEATURES,MAX_NODES

class GraphPolicy(nn.Module):
    def __init__(self,hidden=96):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(FEATURES,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.score=nn.Sequential(nn.Linear(2*hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.wait=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,x,mask):
        z=self.encoder(x)
        real=x[...,-1:]
        context=(z*real).sum(-2)/real.sum(-2).clamp_min(1)
        logits=self.score(torch.cat([z,context.unsqueeze(-2).expand_as(z)],-1)).squeeze(-1)
        logits=torch.cat([logits,self.wait(context)],-1)
        return logits.masked_fill(~mask,-1e9)

    def choose(self,x,m):
        with torch.no_grad():choice=self(torch.from_numpy(x)[None],torch.from_numpy(m)[None]).argmax(-1)[0].numpy()
        return np.where(choice==MAX_NODES,-1,choice)


def load(run=RUN):
    c=torch.load(Path(run)/'policy.pt',map_location='cpu',weights_only=True)
    model=GraphPolicy();model.load_state_dict(c['state_dict']);model.eval()
    return model,c


def _accuracy(policy,x,m,y):
    with torch.no_grad():p=policy(x,m).argmax(-1)
    return float((p==y).all(-1).float().mean()),float((p==y).float().mean())


def train(data=DATA,run=RUN,epochs=300):
    if epochs<=0:raise ValueError('epochs must be positive')
    data,run=Path(data),Path(run);run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(19)
    d=np.load(data,allow_pickle=False);meta=json.loads(data.with_suffix('.json').read_text())
    partition={k:np.isin(d['episode'],v) for k,v in meta['splits'].items()}
    x=torch.from_numpy(d['features']);m=torch.from_numpy(d['mask']);y=torch.from_numpy(d['labels']).long()
    tx,tm,ty=x[partition['train']],m[partition['train']],y[partition['train']]
    vx,vm,vy=x[partition['validation']],m[partition['validation']],y[partition['validation']]
    policy=GraphPolicy();opt=torch.optim.AdamW(policy.parameters(),lr=.001,weight_decay=.0001)
    best=-1.;history=[];generator=torch.Generator().manual_seed(19)
    for epoch in range(1,epochs+1):
        policy.train();total=0.
        for indices in torch.randperm(len(tx),generator=generator).split(128):
            bx,bm,by=tx[indices].clone(),tm[indices].clone(),ty[indices].clone()
            # Shuffle candidates and exchange arms; these are augmentations of
            # the same published recordings, never new expert demonstrations.
            permutation=torch.randperm(MAX_NODES,generator=generator)
            inverse=torch.argsort(permutation)
            bx=bx[:,:,permutation];bm=torch.cat([bm[:,:,:MAX_NODES][:,:,permutation],bm[:,:,-1:]],-1)
            by=torch.where(by==MAX_NODES,by,inverse[by.clamp_max(MAX_NODES-1)])
            if epoch%2:bx=bx.flip(1);bm=bm.flip(1);by=by.flip(1)
            # Published proprioception varies per trajectory. Modality dropout
            # reduces dependence on simulator-specific motor-state correlations.
            if epoch%3:bx[:,:,:,17:53]=0
            opt.zero_grad();logits=policy(bx,bm)
            loss=nn.functional.cross_entropy(logits.reshape(-1,MAX_NODES+1),by.reshape(-1))
            loss.backward();nn.utils.clip_grad_norm_(policy.parameters(),5.);opt.step();total+=loss.item()*len(indices)
        policy.eval();raw_score,arm=_accuracy(policy,vx,vm,vy)
        robust_vx=vx.clone();robust_vx[:,:,:,17:53]=0
        robust_score,_=_accuracy(policy,robust_vx,vm,vy)
        score=min(raw_score,robust_score)
        history.append({'epoch':epoch,'loss':total/len(tx),'validation_joint_accuracy':raw_score,'validation_without_proprioception':robust_score})
        if score>best:
            best=score
            torch.save({'state_dict':policy.state_dict(),'epoch':epoch,'dataset_sha256':sha256(data),
                        'source':'published RoboFactory recordings only','splits':meta['splits']},run/'policy.pt')
        if epoch==1 or epoch%50==0:print(f'epoch {epoch}: loss={total/len(tx):.4f} val joint={score:.3f}',flush=True)
    write_json(run/'training.json',{'method':'behavior cloning; no RL or generated expert labels','source_episodes':meta['episodes'],
                                   'train_episodes':len(meta['splits']['train']),'history':history,'best_validation':best})
    return evaluate(data,run)


def evaluate(data=DATA,run=RUN):
    data,run=Path(data),Path(run);policy,c=load(run)
    if sha256(data)!=c['dataset_sha256']:raise ValueError('Checkpoint/dataset mismatch')
    d=np.load(data,allow_pickle=False);meta=json.loads(data.with_suffix('.json').read_text())
    mask=np.isin(d['episode'],c['splits']['test'])
    x=torch.from_numpy(d['features'][mask]);m=torch.from_numpy(d['mask'][mask]);y=torch.from_numpy(d['labels'][mask]).long()
    joint,arm=_accuracy(policy,x,m,y)
    per_task={}
    for i,name in enumerate(meta['tasks']):
        subset=d['task'][mask]==i
        j,a=_accuracy(policy,x[subset],m[subset],y[subset]);per_task[name]={'joint_accuracy':j,'arm_accuracy':a,'events':int(sum(subset))}
    report={'evaluation':'held-out published trajectories; weak semantic annotations and supplied task graphs',
            'joint_accuracy':joint,'arm_accuracy':arm,'test_events':len(x),'test_episodes':len(c['splits']['test']),
            'train_episodes':len(c['splits']['train']),'source_episodes':meta['episodes'],'selected_epoch':c['epoch'],'per_task':per_task}
    write_json(run/'imitation_evaluation.json',report);print(json.dumps(report,indent=2),flush=True)
    return report
