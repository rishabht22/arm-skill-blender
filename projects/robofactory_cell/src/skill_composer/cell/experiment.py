"""Closed-loop manipulation, isolated skill inspection, and matched baselines."""
import json,time
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont

from ..common import write_json,sha256
from .dataset import RUN
from .policy import load
from .program import Node,composed_program
from .world import AssemblyCell


def choose(world,policy,controller):
    if controller=='learned':return policy.choose(*world.observation())
    ready=[[i for i,n in enumerate(world.nodes) if n.arm==a and world.can_start(i)] for a in range(2)]
    if controller=='serial':
        if any(a is not None for a in world.active):return [-1,-1]
        candidates=[i for r in ready for i in r]
        if not candidates:return [-1,-1]
        i=min(candidates);return [i if world.nodes[i].arm==a else -1 for a in range(2)]
    if controller=='greedy':return [r[0] if r else -1 for r in ready]
    if controller in ('arm0','arm1'):
        arm=int(controller[-1]);return [ready[a][0] if a==arm and ready[a] else -1 for a in range(2)]
    raise ValueError(controller)


def camera():
    c=mujoco.MjvCamera();c.lookat[:]=[.02,0,.20];c.distance=2.45;c.azimuth=145;c.elevation=-35
    return c


def panel(world,rgb,controller,task):
    w=rgb.shape[1];im=Image.new('RGB',(w,rgb.shape[0]+185),'#101820');im.paste(Image.fromarray(rgb),(0,0))
    d=ImageDraw.Draw(im);font=ImageFont.load_default(size=16);small=ImageFont.load_default(size=13);y=rgb.shape[0]+8
    names={'learned':'trained composition head','serial':'serial coordinator','greedy':'graph greedy baseline'}
    d.text((12,y),f'{task} | {names.get(controller,controller)} | t={world.time:.1f}s | {sum(world.status==2)}/{len(world.nodes)} operations',font=font,fill='white')
    for a in range(2):
        i=world.active[a]
        txt='waiting' if i is None else f'{world.nodes[i].op}: {world.nodes[i].obj} -> {world.nodes[i].target}'
        d.text((12,y+28+22*a),f'Arm {a}: {txt}',font=small,fill='#8ed3ff' if a==1 else '#e6e9ed')
    for i,n in enumerate(world.nodes):
        px=12+(i%10)*max(45,(w-24)//10);py=y+81+(i//10)*26
        color=['#34434e','#edb744','#69d49b'][world.status[i]]
        d.rounded_rectangle((px,py,px+44,py+21),radius=4,fill=color)
        symbol={'pick':'K','place':'P','open_lid':'L'}[n.op]
        d.text((px+3,py+3),f'{i}:{symbol}',font=small,fill='#091016')
    d.text((12,y+138),f'K=pick, P=place, L=lid | rejected dispatches={world.rejected} | arm contact frames={world.arm_contacts}',font=small,fill='#b7c2cd')
    d.text((12,y+158),'Published recordings only for training | supplied task graph | idealized grasp welds',font=small,fill='#b7c2cd')
    return np.asarray(im)


def episode(task='relay_stack',parts=3,controller='learned',seed=0,run=RUN,viewer=False,gif=None,max_time=70.,delay=0.,nodes=None):
    policy,_=load(run)
    world=AssemblyCell(nodes if nodes is not None else composed_program(task,parts),seed=seed,delay=delay)
    renderer=window=None;frames=[];c=camera();last=None
    try:
        if gif:renderer=mujoco.Renderer(world.model,height=480,width=800)
        if viewer:
            from mujoco import viewer as mj_viewer
            window=mj_viewer.launch_passive(world.model,world.data)
            window.cam.lookat[:]=c.lookat;window.cam.distance=c.distance;window.cam.azimuth=c.azimuth;window.cam.elevation=c.elevation
        for step in range(round(max_time/world.dt)):
            start=time.monotonic()
            world.dispatch(choose(world,policy,controller));world.step()
            current=tuple(world.active)
            if viewer and current!=last:
                labels=['WAIT' if i is None else f'{world.nodes[i].op} {world.nodes[i].obj} -> {world.nodes[i].target}' for i in current]
                print(f't={world.time:.2f}s | Arm 0: {labels[0]} | Arm 1: {labels[1]}',flush=True);last=current
            if renderer and step%5==0:
                renderer.update_scene(world.data,camera=c);frames.append(panel(world,renderer.render(),controller,task))
            if window:
                if not window.is_running():break
                with window.lock():
                    window.user_scn.ngeom=0
                    for a,index in enumerate(world.active):
                        geom=window.user_scn.geoms[window.user_scn.ngeom]
                        pos=world.data.site(f'a{a}_tcp').xpos+np.array([0,0,.11])
                        mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_LABEL,np.zeros(3),pos,np.eye(3).reshape(-1),np.array([1.,1.,1.,1.]))
                        geom.label=f'Arm {a}: WAIT' if index is None else f'Arm {a}: {world.nodes[index].op} {world.nodes[index].obj}'
                        window.user_scn.ngeom+=1
                window.sync();time.sleep(max(0.,world.dt-(time.monotonic()-start)))
            if all(world.status==2) or world.failures:break
        # Allow released parts to settle; final positions determine physical success.
        for _ in range(12):world.step()
        if renderer:
            renderer.update_scene(world.data,camera=c)
            frames.extend([panel(world,renderer.render(),controller,task)]*8)
    finally:
        if renderer:renderer.close()
        if window:window.close()
    result=world.result();result.update(task=task,parts=parts,controller=controller,seed=seed)
    result['strict_success']=bool(result['success'] and not result['rejected_dispatches'] and not result['sampled_arm_contact_frames'])
    if gif:
        import imageio.v2 as imageio
        Path(gif).parent.mkdir(parents=True,exist_ok=True);imageio.mimsave(gif,frames,duration=90,loop=0)
    return result


def evaluate(run=RUN,seeds=20,parts=3,seed_start=1000,tasks=None,filename='closed_loop_evaluation.json'):
    if seeds<=0:raise ValueError('seeds must be positive')
    rows=[]
    for task in tasks or ['handoff','parallel_stack','container','relay_stack','relay_container']:
        for seed in range(seed_start,seed_start+seeds):
            for controller in ['learned','serial','greedy']:
                r=episode(task,parts,controller,seed,run);rows.append(r)
        print(task,[(c,round(np.mean([r['success'] for r in rows if r['task']==task and r['controller']==c]),3)) for c in ['learned','serial','greedy']],flush=True)
    summary=[]
    for task in sorted({r['task'] for r in rows}):
        for c in ['learned','serial','greedy']:
            subset=[r for r in rows if r['task']==task and r['controller']==c];successful=[r['time_s'] for r in subset if r['success']]
            summary.append({'task':task,'controller':c,'episodes':len(subset),'success_rate':float(np.mean([r['success'] for r in subset])),
                            'strict_success_rate':float(np.mean([r['strict_success'] for r in subset])),
                            'mean_success_time_s':float(np.mean(successful)) if successful else None,
                            'mean_rejected_dispatches':float(np.mean([r['rejected_dispatches'] for r in subset])),
                            'mean_contact_frames':float(np.mean([r['sampled_arm_contact_frames'] for r in subset]))})
    report={'training_source':'Published RoboFactory only; no generated expert labels or RL','held_out_compositions':['relay_stack','relay_container'],
            'seed_start':seed_start,'seeds_per_task':seeds,'parts':parts,
            'policy_sha256':sha256(Path(run)/'policy.pt'),'executor_sha256':sha256(Path(__file__).with_name('world.py')),
            'generalization_scope':'Unseen supplied task graphs built from known operation types; not unseen natural-language goals or learned perception',
            'grasp_model':'Proximity-gated welds; low-level scripted IK policies are frozen','summary':summary,'episodes':rows}
    write_json(Path(run)/filename,report)
    return report
