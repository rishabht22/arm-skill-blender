"""Articulated two-Panda manipulation cell with reusable closed-loop skills.

Objects simulate under gravity. Grasping uses proximity-gated idealized welds;
this is explicitly not a friction-grasp or RoboFactory benchmark reproduction.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from ..robot import panda_scene as _robot_scene, vector as _vector
from .program import Node, features, robot_state

HOME=np.array([0.,.3927,0.,-2.3,0.,2.6927,.7854,.04,.04])
COLORS=[(.94,.30,.22,1),(.25,.70,.94,1),(.98,.75,.18,1),(.42,.83,.44,1)]

@dataclass
class Segment:
    q: np.ndarray
    duration: float
    hook: str = ''


class AssemblyCell:
    dt=.05
    def __init__(self,nodes,seed=0,delay=0.,speed=1.):
        self.nodes=nodes
        self.status=np.zeros(len(nodes),dtype=int)
        self.held=[None,None]
        self.active=[None,None]
        self.motions=[[],[]]
        self.segment_time=np.zeros(2)
        self.segment_start=[None,None]
        self.time=0.;self.delay=delay;self.speed=speed
        self.rejected=0;self.failures=[];self.trace=[];self.completed=[]
        self.arm_contacts=0;self.lid_open=False
        root=_robot_scene()
        root.set('model','RoboFactory_skill_composition_cell')
        root.find('option').set('iterations','40')
        world=root.find('worldbody')
        world.find("geom[@name='table']").set('rgba','.14 .18 .22 1')
        for a in range(2):
            world.find(f"body[@name='a{a}_link0']").set('pos',f'0 {-.62 if a==0 else .62} 0')
        for actuator in root.find('actuator'):
            if 'finger' not in actuator.get('joint'):
                actuator.set('kp','1400');actuator.set('kv','90')
        self.objects=sorted({n.obj for n in nodes if n.obj!='lid'})
        rng=np.random.default_rng(seed)
        self.initial_positions={}
        eq=ET.SubElement(root,'equality')
        for p,obj in enumerate(self.objects):
            first=next(n for n in nodes if n.obj==obj and n.op=='pick')
            pos=np.array([-.26+.15*p, -.22 if first.arm==0 else .22, .026])
            pos[:2]+=rng.uniform(-.012,.012,2)
            self.initial_positions[obj]=pos
            self._object(world,eq,obj,pos,COLORS[p%4],'.024 .024 .024')
        self.goals={}
        places=[n for n in nodes if n.op=='place' and not n.target.startswith('dock')]
        stack_count=0
        target_y=0. if any(n.arm==0 for n in places) else .15
        center=np.array([.18,target_y])
        for n in places:
            if n.target in self.goals:continue
            if n.stack:
                stack_count+=1
            # All final parts form a physical tower, or enter a raised container.
            z=.026+.049*stack_count
            self.goals[n.target]=np.array([*center,z])
        self.docks={f'dock{p}':np.array([-.28+.16*p,0.,.066]) for p in range(4)}
        used_docks={n.target for n in nodes if n.target.startswith('dock')}
        for name,pos in self.docks.items():
            if name not in used_docks:continue
            ET.SubElement(world,'geom',name=name,type='cylinder',size='.045 .02',pos=_vector([*pos[:2],.02]),rgba='.31 .37 .43 1',contype='2',conaffinity='2')
            ET.SubElement(world,'site',name=name+'_marker',type='cylinder',size='.037 .001',pos=_vector([*pos[:2],.041]),rgba='.25 .80 .80 .65')
        ET.SubElement(world,'geom',type='box',size='.10 .09 .003',pos=_vector([*center,.003]),rgba='.30 .35 .40 1',contype='2',conaffinity='2')
        if len(places)==1 and places[0].stack:
            ET.SubElement(world,'geom',type='box',size='.024 .024 .024',pos=_vector([*center,.026]),rgba='.6 .6 .65 1',contype='2',conaffinity='2')
        if any(n.op=='open_lid' for n in nodes):
            self._object(world,eq,'lid',np.array([*center,.185]),(.60,.67,.77,1),'.12 .10 .012',lid=True)
            for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
                pos=np.array([*center,.075]);pos[axis]+=sign*(.11 if axis==0 else .09)
                size=[.12,.008,.075] if axis==1 else [.008,.10,.075]
                ET.SubElement(world,'geom',type='box',size=_vector(size),pos=_vector(pos),rgba='.35 .40 .47 1',contype='2',conaffinity='2')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model);self.ik=mujoco.MjData(self.model)
        self.data.qpos[:18]=np.tile(HOME,2);self.data.ctrl[:]=self.data.qpos[:18]
        mujoco.mj_forward(self.model,self.data)
        self.rotations=[self.data.site(f'a{a}_tcp').xmat.reshape(3,3).copy() for a in range(2)]
        self.robot_ranges=[]
        for a in range(2):
            joints=[self.model.joint(f'a{a}_joint{i}').id for i in range(1,8)]
            self.robot_ranges.append(self.model.jnt_range[joints].copy())
        # Settle props while holding both robots at their initial pose.
        for _ in range(150):mujoco.mj_step(self.model,self.data)

    def _object(self,world,eq,name,pos,color,size,lid=False):
        body=ET.SubElement(world,'body',name=name,pos=_vector(pos))
        ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'geom',name=name+'_geom',type='box',size=size,mass='.05' if not lid else '.08',rgba=_vector(color),contype='2',conaffinity='2',friction='1 .01 .001')
        for a in range(2):
            ET.SubElement(body,'site',name=f'{name}_grip{a}',pos=f'0 {-.10 if a==0 else .08} .025' if lid else '0 0 0',size='.004',rgba='1 1 1 0')
            ET.SubElement(eq,'weld',name=f'a{a}_{name}_weld',site1=f'a{a}_tcp',site2=f'{name}_grip{a}',active='false',solref='.008 1',torquescale='.05')

    def observation(self):
        state=robot_state(self.data.qpos[:18].reshape(2,9),self.data.qvel[:18].reshape(2,9))
        x,mask=features(self.nodes,self.status,state)
        if self.time<self.delay:mask[1,:-1]=False
        return x,mask

    def solve(self,a,target,seed):
        self.ik.qpos[:]=self.data.qpos
        sid=self.model.site(f'a{a}_tcp').id
        jp=np.zeros((3,self.model.nv));jr=jp.copy()
        last_error=1.
        for initial in (np.asarray(seed[:7]),HOME[:7],np.array([.25,.2,0.,-2.2,0.,2.4,.7854])):
            q=initial.copy()
            for _ in range(220):
                self.ik.qpos[a*9:a*9+7]=q
                mujoco.mj_forward(self.model,self.ik)
                site=self.ik.site(sid)
                delta=np.asarray(target)-site.xpos
                current=site.xmat.reshape(3,3)
                rotation=.5*sum(np.cross(current[:,i],self.rotations[a][:,i]) for i in range(3))
                if np.linalg.norm(delta)<.0015 and np.linalg.norm(rotation)<.04:return q
                mujoco.mj_jacSite(self.model,self.ik,jp,jr,sid)
                jac=np.vstack([jp[:,a*9:a*9+7],.35*jr[:,a*9:a*9+7]])
                error=np.r_[delta,.35*rotation]
                dq=jac.T@np.linalg.solve(jac@jac.T+.0003*np.eye(6),error)
                q+=np.clip(dq,-.12,.12)
                q=np.clip(q,self.robot_ranges[a][:,0]+.005,self.robot_ranges[a][:,1]-.005)
            last_error=float(np.linalg.norm(delta))
        raise RuntimeError(f'IK failed arm {a} target {np.round(target,3)} residual {last_error:.3f}')

    def _trajectory(self,a,node):
        q=self.data.qpos[a*9:(a+1)*9].copy();out=[]
        current=self.data.site(f'a{a}_tcp').xpos.copy()
        def move(pos,fingers=None,hook=''):
            nonlocal q,current
            target=q.copy();target[:7]=self.solve(a,pos,q)
            if fingers is not None:target[7:]=fingers
            duration=max(.22,float(np.max(np.abs(target[:7]-q[:7])))/1.5)
            out.append(Segment(target,duration,hook));q=target;current=np.asarray(pos)
        cruise=.37 if a==0 else .45
        if node.op in ('pick','open_lid'):
            site=self.data.site(f'{node.obj}_grip{a}').xpos.copy()
            move([*current[:2],max(cruise,current[2])],.04)
            move([*site[:2],cruise],.04)
            move(site,.04)
            move(site,0.,'attach')
            lifted=site+np.array([0,0,.17])
            if node.op=='open_lid':lifted=np.array([.18,-.30 if a==0 else .38,.43])
            move(lifted,0.)
            if node.op=='pick':
                move([site[0],-.25 if a==0 else .30,max(.30,lifted[2])],0.)
        else:
            goal=self.docks[node.target] if node.target.startswith('dock') else self.goals[node.target]
            move([*current[:2],max(cruise,current[2])],0.)
            move([*goal[:2],cruise],0.)
            move(goal,0.)
            move(goal,.04,'detach')
            move([*goal[:2],cruise],.04)
            # Clear shared docks before allowing the next receiver to approach.
            move([-.04,-.20 if a==0 else .30,cruise],.04)
        return out

    def can_start(self,index):
        node=self.nodes[index];a=node.arm
        if self.active[a] is not None or self.status[index]!=0:return False
        if a==1 and self.time<self.delay:return False
        if not all(self.status[d]==2 for d in node.deps):return False
        if node.op in ('pick','open_lid'):
            return self.held[a] is None and node.obj not in self.held
        return self.held[a]==node.obj and (not node.container or self.lid_open)

    def dispatch(self,choices):
        for a,index in enumerate(choices):
            index=int(index)
            if index<0:continue
            if self.nodes[index].arm!=a or not self.can_start(index):
                self.rejected+=1;continue
            try:motion=self._trajectory(a,self.nodes[index])
            except RuntimeError as exc:
                self.failures.append(str(exc));continue
            self.active[a]=index;self.status[index]=1;self.motions[a]=motion
            self.segment_time[a]=0.;self.segment_start[a]=self.data.qpos[a*9:(a+1)*9].copy()
            self.trace.append({'time_s':self.time,'arm':a,'node':index,'op':self.nodes[index].op,'object':self.nodes[index].obj,'target':self.nodes[index].target})

    def _hook(self,a,hook,node):
        eq=self.model.equality(f'a{a}_{node.obj}_weld').id
        if hook=='attach':
            site=self.data.site(f'{node.obj}_grip{a}')
            if np.linalg.norm(site.xpos-self.data.site(f'a{a}_tcp').xpos)>.04:
                raise RuntimeError('Grasp missed '+node.obj)
            body=self.data.body(node.obj)
            rotation=body.xmat.reshape(3,3).T@self.data.site(f'a{a}_tcp').xmat.reshape(3,3)
            quaternion=np.zeros(4);mujoco.mju_mat2Quat(quaternion,rotation.reshape(-1))
            self.model.site_quat[self.model.site(f'{node.obj}_grip{a}').id]=quaternion
            self.data.eq_active[eq]=True;self.held[a]=node.obj
        elif hook=='detach':
            self.data.eq_active[eq]=False;self.held[a]=None

    def step(self):
        for a,index in enumerate(self.active):
            if index is None:continue
            segment=self.motions[a][0]
            self.segment_time[a]+=self.dt*self.speed
            t=min(1.,self.segment_time[a]/segment.duration)
            smooth=t*t*(3-2*t)
            self.data.ctrl[a*9:(a+1)*9]=self.segment_start[a]+smooth*(segment.q-self.segment_start[a])
        for _ in range(round(self.dt/self.model.opt.timestep)):mujoco.mj_step(self.model,self.data)
        self.time+=self.dt
        for a,index in enumerate(self.active):
            if index is None:continue
            segment=self.motions[a][0]
            error=np.linalg.norm(self.data.qpos[a*9:a*9+7]-segment.q[:7])
            if self.segment_time[a]>=segment.duration and error<.055:
                try:self._hook(a,segment.hook,self.nodes[index])
                except RuntimeError as exc:
                    self.failures.append(str(exc));self.active[a]=None;continue
                self.motions[a].pop(0);self.segment_time[a]=0.
                self.segment_start[a]=self.data.qpos[a*9:(a+1)*9].copy()
                if not self.motions[a]:
                    self.status[index]=2;self.active[a]=None;self.completed.append(index)
                    if self.nodes[index].op=='open_lid':self.lid_open=True
            elif self.segment_time[a]>segment.duration+4:
                self.failures.append(f'Tracking timeout node {index}');self.active[a]=None
        for contact in self.data.contact:
            names=[self.model.body(self.model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
            if (names[0].startswith('a0_') and names[1].startswith('a1_')) or (names[1].startswith('a0_') and names[0].startswith('a1_')):
                self.arm_contacts+=1;break

    def result(self):
        final_nodes=[i for i,n in enumerate(self.nodes) if n.op=='place' and not n.target.startswith('dock')]
        errors={}
        for i in final_nodes:
            n=self.nodes[i]
            errors[n.obj]=float(np.linalg.norm(self.data.body(n.obj).xpos-self.goals[n.target]))
        return {'success':bool(np.all(self.status==2) and not self.failures and all(e<.025 for e in errors.values())),
                'time_s':round(self.time,3),'completed_nodes':int(sum(self.status==2)),'total_nodes':len(self.nodes),
                'placement_errors_m':errors,'rejected_dispatches':self.rejected,'tracking_failures':self.failures,
                'sampled_arm_contact_frames':self.arm_contacts,'dispatch_trace':self.trace}
