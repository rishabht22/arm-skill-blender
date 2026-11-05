"""Matched comparisons of individual skills, learned coordination, and baselines."""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .common import RUN, write_json
from .execution import SCENARIOS, camera, load_trials, render_panel, save_animation


def compare(run=RUN, steps=120, gif=None, seed=0, delay_arm1=0.0, output=None):
    if steps < 10:
        raise ValueError("Comparison needs at least 10 control steps")
    run = Path(run)
    trials, digest = load_trials(run, SCENARIOS, seed, delay_arm1)
    renderers, frames, gallery = [], [], []
    try:
        if gif:
            for trial in trials:
                renderers.append(mujoco.Renderer(trial.env.model, height=270, width=400))
        view = camera()
        gallery_end = None
        for t in range(steps):
            for trial in trials:
                trial.step()
            if gif and t % 2 == 0:
                panels = []
                for trial, renderer in zip(trials, renderers):
                    renderer.update_scene(trial.env.data, camera=view)
                    panels.append(render_panel(trial, renderer.render()))
                canvas = Image.new("RGB", (1200, 900), "#0c1118")
                for i, panel in enumerate(panels):
                    canvas.paste(panel, ((i % 3) * 400, (i // 3) * 425))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 859), f"Arm 1 motor-start delay: {delay_arm1:.1f}s in every panel. Same frozen skills. Goal: both grasps, level lift >0.15m for 0.5s.", fill="white")
                draw.text((12, 879), "MuJoCo proxy with idealized attachments. A passing fixed baseline means learning is not necessary for this task.", fill="#b4beca")
                frames.append(np.asarray(canvas))
                if delay_arm1 == 0 and gallery_end is None:
                    strip = Image.new("RGB", (800, 460), "#0c1118")
                    strip.paste(panels[0], (0, 0))
                    strip.paste(panels[1], (400, 0))
                    ImageDraw.Draw(strip).text((10, 438), "Individual skill inspection | 3x slow playback | each arm has its own scripted skill queue", fill="white")
                    gallery.append(np.asarray(strip))
                    if all(trial.local_completed[a].all() for trial, a in zip(trials[:2], [0, 1])):
                        gallery_end = t + 10
                elif gallery_end is not None and t <= gallery_end:
                    # Keep the final completed-skill panel visible briefly.
                    gallery.append(gallery[-1])
    finally:
        for renderer in renderers:
            renderer.close()
    reports = [trial.report() for trial in trials]
    report = {"environment": "MuJoCo proxy with idealized point attachments; not the RoboFactory benchmark",
              "frozen_skills_sha256": digest, "seed": seed, "steps": steps,
              "success_criterion": "Both attachments, height gain >0.15m, handle height difference <0.04m, no sampled arm contacts, sustained 0.5s",
              "causal_limit": "Single-arm failure shows this defined task requires both grasps. It does not prove a learned head is necessary; compare independent and fixed baselines.",
              "trials": reports}
    output = Path(output) if output else run / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", report)
    for r in reports:
        print(f"{r['scenario']:16s} shared task={'PASS' if r['shared_task_success'] else 'FAIL'}  lift={r['final']['height_gain_m']:.3f} m  guard blocks={r['total_guard_interventions']}")
    if gif:
        save_animation(gif, frames, duration=100)
        if gallery:
            save_animation(output / "individual_skills.gif", gallery, duration=300)
        print(f"Saved {gif}")
    return report

