"""Original images from the verified public data; these are not simulated replays."""
from pathlib import Path
import h5py,numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .dataset import RAW,RUN,actions,intervals
from .program import source_program


def preview(task='PassShoe',episode=0,run=RUN):
    frames=[];nodes=source_program(task)
    with h5py.File(RAW/f'{task}.h5') as data:
        g=data[f'traj_{episode}'];spans=intervals(task,actions(g))
        camera='head_camera_global'
        if camera not in g['obs/sensor_data']:camera=list(g['obs/sensor_data'])[0]
        rgb=g[f'obs/sensor_data/{camera}/rgb']
        for t in range(0,len(rgb)-1,2):
            panel=Image.new('RGB',(640,550),'#101820');panel.paste(Image.fromarray(rgb[t]).resize((640,480)),(0,0));draw=ImageDraw.Draw(panel)
            draw.text((12,488),f'ORIGINAL published RoboFactory | {task} episode {episode} | frame {t}',fill='white')
            for a in range(2):
                active=[nodes[i].op for i,s,e in spans if nodes[i].arm==a and s<=t<e]
                draw.text((12,510+18*a),f'Arm {a}: {active[0] if active else "waiting"} (weak annotation)',fill='#8ed3ff')
            frames.append(np.asarray(panel))
    out=Path(run)/f'published_{task}.gif';out.parent.mkdir(parents=True,exist_ok=True)
    imageio.mimsave(out,frames,duration=100,loop=0);print(out)
    return out
