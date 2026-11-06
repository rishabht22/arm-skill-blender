"""Reproducible galleries assembled from actual evaluation trajectories."""
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageSequence

from ..common import write_json
from .dataset import RUN
from .program import Node
from .experiment import episode


def _clip(path,size):
    with Image.open(path) as im:
        return [f.convert('RGB').resize(size) for f in ImageSequence.Iterator(im)]


def gallery(run=RUN):
    run=Path(run)
    clips=[_clip(run/f'arm{a}_{skill}.gif',(400,333)) for a in range(2) for skill in ['pick','place','stack','open_lid']]
    frames=[]
    for t in range(max(map(len,clips))):
        canvas=Image.new('RGB',(1600,700),'#101820')
        for i,clip in enumerate(clips):canvas.paste(clip[min(t,len(clip)-1)],((i%4)*400,(i//4)*333))
        ImageDraw.Draw(canvas).text((10,680),'Isolated frozen skills: pick | place | stack | open lid. Arm 0 top row; arm 1 bottom row. No learned coordinator controls these clips.',fill='white')
        frames.append(np.asarray(canvas))
    imageio.mimsave(run/'individual_skills.gif',frames,duration=120,loop=0)


def comparison(run=RUN):
    run=Path(run)
    clips=[_clip(run/f'parallel_{name}.gif',(640,532)) for name in ['learned','serial']]
    frames=[]
    for t in range(max(map(len,clips))):
        canvas=Image.new('RGB',(1280,570),'#101820')
        for i,clip in enumerate(clips):canvas.paste(clip[min(t,len(clip)-1)],(i*640,0))
        d=ImageDraw.Draw(canvas)
        d.text((12,538),'Trained composition head: parallel execution',fill='#89d9aa')
        d.text((652,538),'Serial baseline: same task, skills, objects, and seed',fill='#efbc7b')
        d.text((12,557),'Matched simulation timelines; the finished controller stays on its final frame. Playback accelerated equally.',fill='white')
        frames.append(np.asarray(canvas))
    imageio.mimsave(run/'coordination_comparison.gif',frames,duration=90,loop=0)


def demos(run=RUN):
    run=Path(run);records=[]
    for arm in range(2):
        for skill in ['pick','place','stack','open_lid']:
            nodes=[Node('open_lid',arm,obj='lid')] if skill=='open_lid' else [Node('pick',arm)]
            if skill in ('place','stack'):nodes.append(Node('place',arm,deps=(0,),stack=skill=='stack'))
            records.append(episode('individual_'+skill,controller='greedy',run=run,nodes=nodes,gif=run/f'arm{arm}_{skill}.gif'))
    gallery(run)
    for task,parts,name in [('relay_container',3,'relay_container'),('relay_container',4,'relay_container_4parts'),('relay_stack',4,'relay_stack')]:
        records.append(episode(task,parts,run=run,gif=run/(name+'.gif')))
    for controller in ['learned','serial']:
        records.append(episode('relay_stack',3,controller,run=run,gif=run/f'parallel_{controller}.gif'))
    comparison(run)
    write_json(run/'demo_results.json',{'note':'Actual evaluation trajectories for visualization; none enter imitation training','episodes':records})
    print(f'Demo galleries saved in {run}',flush=True)
