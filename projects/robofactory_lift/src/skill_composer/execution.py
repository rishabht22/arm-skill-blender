"""Shared event-driven controllers, episode runner, and rendering helpers."""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .common import RUN, SKILLS, sha256, write_json
from .model import load_head
from .sim import LiftSandbox

SCENARIOS = {
    "arm0": ("Arm 0: individual skills", "Arm 1 stays parked"),
    "arm1": ("Arm 1: individual skills", "Arm 0 stays parked"),
    "independent": ("Both arms: independent queues", "Each advances on its own completion"),
    "head": ("Learned head, no guard", "Measures the neural head alone"),
    "head_guarded": ("Learned head + temporal guard", "Lift/hold require both grasps"),
    "fixed_guarded": ("Fixed sequence + temporal guard", "Tests whether learning is necessary"),
}


def skill_name(skill):
    return "parked" if skill == -1 else SKILLS[skill]


class Trial:
    def __init__(self, scenario, library, head, seed=0, delay_arm1=0.0, randomization=0.0):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario}")
        if delay_arm1 < 0:
            raise ValueError("delay_arm1 must be nonnegative")
        self.scenario, self.head = scenario, head
        self.env = LiftSandbox(library, seed=seed, randomization=randomization)
        self.active = np.array([-1, -1])
        self.enabled = np.array([scenario != "arm1", scenario != "arm0"])
        self.guarded = scenario in ("head_guarded", "fixed_guarded")
        self.delay_steps = round(delay_arm1 / self.env.control_dt)
        self.local_completed = np.zeros((2, 4), dtype=bool)
        self.hold_steps = np.zeros(2, dtype=int)
        self.sustained = self.longest_sustained = 0
        self.trace = []

    def step(self):
        env, t = self.env, len(self.trace)
        terminated = env.completed(self.active)
        for a, s in enumerate(self.active):
            if s >= 0 and terminated[a]:
                self.local_completed[a, s] = True
            if s == 3:
                reached = np.linalg.norm(env.data.qpos[a * 9:a * 9 + 7] - env.executor.command(a, 3)[:7]) < .035
                self.hold_steps[a] = self.hold_steps[a] + 1 if reached else 0
                if self.hold_steps[a] >= 10:
                    self.local_completed[a, 3] = True
        # The same optional motor-start delay applies in every scenario.
        available = self.enabled & np.array([True, t >= self.delay_steps])
        events = terminated & available
        proposed = self.active.copy()
        if events.any():
            if self.scenario.startswith("head"):
                with torch.no_grad():
                    proposed = self.head(torch.from_numpy(env.decision_observation(self.active, terminated))[None]).argmax(-1)[0].numpy()
            else:
                proposed = np.minimum(self.active + 1, 3)
        requested = np.where(events, proposed, self.active)
        requested[~available] = -1
        executed, blocked = env.step(requested, guarded=self.guarded)
        self.active = executed.copy()
        status = env.status()
        # The comparison requires a stable shared lift, not a transient crossing.
        valid = status["sandbox_goal_reached"] and status["arm_collision_frames"] == 0
        self.sustained = self.sustained + 1 if valid else 0
        self.longest_sustained = max(self.longest_sustained, self.sustained)
        row = {"step": t, "time_s": (t + 1) * env.control_dt,
               "decision_event": bool(events.any()), "proposed": proposed.tolist(),
               "executed": executed.tolist(), "blocked_requests": blocked,
               "local_completed": self.local_completed.tolist(), **status}
        self.trace.append(row)
        return row

    def report(self):
        return {"scenario": self.scenario, "title": SCENARIOS[self.scenario][0],
                "enabled_arms": self.enabled.tolist(), "temporal_guard": self.guarded,
                "delay_arm1_s": self.delay_steps * self.env.control_dt,
                "shared_task_success": self.longest_sustained >= 10,
                "longest_success_duration_s": self.longest_sustained * self.env.control_dt,
                "local_skills_completed": self.local_completed.tolist(),
                "total_guard_interventions": sum(r["blocked_requests"] for r in self.trace),
                "final": self.env.status(), "trace": self.trace}


def load_trials(run, scenarios, seed=0, delay_arm1=0.0, randomization=0.0):
    run = Path(run)
    head, checkpoint = load_head(run / "head.pt")
    digest = sha256(run / "skills.json")
    if digest != checkpoint["skills_sha256"]:
        raise ValueError("Frozen skill library differs from the trained checkpoint")
    library = json.loads((run / "skills.json").read_text())
    return [Trial(name, library, head, seed, delay_arm1, randomization) for name in scenarios], digest


def camera():
    view = mujoco.MjvCamera()
    view.lookat[:] = [0, 0, .35]
    view.distance, view.azimuth, view.elevation = 2.7, 150, -25
    return view


def render_panel(trial, rgb):
    width, height = rgb.shape[1], rgb.shape[0]
    panel = Image.new("RGB", (width, height + 155), "#151b23")
    panel.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    row, env = trial.trace[-1], trial.env
    title, detail = SCENARIOS[trial.scenario]
    draw.text((10, height + 7), title, font=font, fill="white")
    draw.text((10, height + 29), detail, font=small, fill="#b4beca")
    draw.text((10, height + 49), f"Active: {skill_name(trial.active[0])} / {skill_name(trial.active[1])}   t={row['time_s']:.1f}s", font=small, fill="#92d3ff")
    for a in range(2):
        completed = " ".join(name for name, done in zip(SKILLS, trial.local_completed[a]) if done) or "none yet"
        draw.text((10, height + 69 + 17 * a), f"Arm {a} completed: {completed}", font=small, fill="#b4beca")
    success = trial.longest_sustained >= 10
    draw.text((10, height + 106), f"Shared task: {'PASS' if success else 'not achieved'} | lift {env.status()['height_gain_m']:.3f} m", font=font, fill="#8ee2b0" if success else "#ffc078")
    draw.text((10, height + 130), f"Both grasped: {bool(env.attachments.all())} | guard blocks: {sum(r['blocked_requests'] for r in trial.trace)}", font=small, fill="#b4beca")
    return panel


def save_animation(path, frames, duration):
    import imageio.v2 as imageio
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=duration, loop=0)


def run_episode(run=RUN, scenario="head_guarded", steps=160, viewer=False, gif=None,
                seed=0, randomization=0.0):
    """Run one controller for either a rollout or an individual skill inspection."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    trials, _ = load_trials(run, [scenario], seed=seed, randomization=randomization)
    trial = trials[0]
    renderer = window = None
    frames = []
    view = camera()
    slow_motion = 3 if scenario in ("arm0", "arm1") else 1
    previous = None
    try:
        if gif:
            renderer = mujoco.Renderer(trial.env.model, height=360, width=640)
        if viewer:
            from mujoco import viewer as mj_viewer
            window = mj_viewer.launch_passive(trial.env.model, trial.env.data)
            window.cam.lookat[:] = view.lookat
            window.cam.distance, window.cam.azimuth, window.cam.elevation = view.distance, view.azimuth, view.elevation
        for _ in range(steps):
            started = time.monotonic()
            trial.step()
            active = tuple(trial.active)
            if active != previous:
                print("Active skills: " + " / ".join(skill_name(s) for s in active), flush=True)
                previous = active
            if renderer:
                renderer.update_scene(trial.env.data, camera=view)
                frames.append(np.asarray(render_panel(trial, renderer.render())))
            if window:
                if not window.is_running():
                    break
                window.sync()
                time.sleep(max(0., slow_motion * trial.env.control_dt - (time.monotonic() - started)))
    finally:
        if renderer:
            renderer.close()
        if window:
            window.close()
    if gif:
        save_animation(gif, frames, duration=round(slow_motion * trial.env.control_dt * 1000))
    report = trial.report()
    report["environment"] = "MuJoCo proxy with idealized point attachments; not the RoboFactory benchmark"
    report["seed"] = seed
    report["randomization_m"] = randomization
    filename = f"{scenario}_skills.json" if scenario in ("arm0", "arm1") else "rollout.json"
    write_json(Path(run) / filename, report)
    print(json.dumps({k: report[k] for k in ["scenario", "shared_task_success", "final"]}, indent=2))
    return report
