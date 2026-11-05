"""Persistent display model, synchronized only from the real contact simulation.

Contact models change at workpiece exchange; the display contains all part mesh
assets to keep the viewer open. It never supplies states to the physics engine.
"""
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .assets import bootstrap
from .experts import TASKS
from .world import OFFSETS


class Display:
    def __init__(self):
        bootstrap()
        from assembly_policy.viz.demo import build_demo_xml
        parent=mujoco.MjSpec.from_string('<mujoco><compiler angle="radian"/><worldbody/></mujoco>')
        children=[]
        for arm in (0,1):
            root=ET.fromstring(build_demo_xml(list(TASKS)))
            if arm==1:
                table=root.find("worldbody/geom[@name='table']")
                root.find('worldbody').remove(table)
            spec=mujoco.MjSpec.from_string(ET.tostring(root,encoding='unicode'))
            frame=parent.worldbody.add_frame(pos=OFFSETS[arm])
            parent.attach(spec,prefix=f'a{arm}_',frame=frame)
            children.append(spec)
        self.model=parent.compile();self.data=mujoco.MjData(self.model)
        self.q=[];self.mocap=[]
        for arm in (0,1):
            ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,f'a{arm}_{name}')
                 for name in [*[f'joint{i}' for i in range(1,8)],'finger_joint1','finger_joint2']]
            self.q.append(np.array([self.model.jnt_qposadr[i] for i in ids]))
            self.mocap.append({task:int(self.model.body_mocapid[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'a{arm}_part_{task}')]) for task in TASKS})

    def sync(self,world):
        self.data.mocap_pos[:]=[0.,0.,-10.]
        self.data.mocap_quat[:]=[1.,0.,0.,0.]
        for arm in (0,1):
            q,_,pq,_,_=world.indices[arm]
            self.data.qpos[self.q[arm]]=world.data.qpos[q]
            index=self.mocap[arm][world.tasks[arm]]
            self.data.mocap_pos[index]=world.data.qpos[pq:pq+3]
            self.data.mocap_quat[index]=world.data.qpos[pq+3:pq+7]
        mujoco.mj_forward(self.model,self.data)
