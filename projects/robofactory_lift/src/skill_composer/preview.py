"""Show predictions on actual held-out demonstrations, not invented rollouts."""
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .common import RAW, RUN, SKILLS
from .data import annotate


def preview(run=RUN, raw=RAW, episode=None):
    run, raw = Path(run), Path(raw)
    prediction = np.load(run / "test_predictions.npz", allow_pickle=False)
    ids = np.unique(prediction["episode"])
    episode = int(ids[0]) if episode is None else episode
    if episode not in ids:
        raise ValueError(f"Preview requires a test episode; choose one of {ids.tolist()}")
    mask = prediction["episode"] == episode
    predicted = prediction["prediction"][mask]
    event_steps = prediction["step"][mask]
    frames = []
    with h5py.File(raw / "LiftBarrier.h5", "r") as data:
        rgb = data[f"traj_{episode}/obs/sensor_data/head_camera_global/rgb"]
        labels = np.stack([annotate(data[f"traj_{episode}/actions/panda-{a}"][:])[0] for a in range(2)], axis=1)
        for t in range(len(labels)):
            panel = Image.new("RGB", (640, 570), "#151b23")
            panel.paste(Image.fromarray(rgb[t]).resize((640, 480)), (0, 0))
            draw = ImageDraw.Draw(panel)
            draw.text((12, 490), f"Offline event predictions | held-out demo {episode} | frame {t}", fill="white")
            choice = predicted[np.searchsorted(event_steps, t, side="right") - 1]
            for a in range(2):
                correct = choice[a] == labels[t, a]
                draw.text((12, 512 + a * 20), f"Arm {a}: head={SKILLS[choice[a]]:8s}  weak expert label={SKILLS[labels[t,a]]}",
                          fill="#8ee2b0" if correct else "#ffad86")
            frames.append(np.asarray(panel))
    output = run / "held_out_demo.gif"
    imageio.mimsave(output, frames, duration=50, loop=0)
    print(f"Saved {output}")
