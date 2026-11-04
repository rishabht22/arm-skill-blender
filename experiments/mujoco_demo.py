"""Run a real pick-and-stack composition without training any policy."""

import argparse
import json
from pathlib import Path
import time
from contextlib import nullcontext

from src.runtime import configure_caches


def main():
    configure_caches()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--task", choices=("stack", "stack-and-relocate"), default="stack-and-relocate")
    args = parser.parse_args()
    from src.mujoco_skills.tabletop import make_tabletop
    from src.mujoco_skills.controllers import TabletopSkills
    import mujoco
    import mujoco.viewer
    from datetime import datetime, timezone
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/mujoco_demo" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    rows = []
    for seed in range(args.seed, args.seed + args.episodes):
        env = make_tabletop(seed, video=args.video)
        writer = None
        renderer = None
        try:
            env.reset()
            context = nullcontext(None) if args.headless else mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(output / f"seed_{seed}.mp4"), fps=20, macro_block_size=1)
                renderer = mujoco.Renderer(env.sim.model._model, height=480, width=640)
                scene_option = mujoco.MjvOption()
                scene_option.geomgroup[0], scene_option.geomgroup[1] = 0, 1
            with context as viewer:
                if viewer:
                    with viewer.lock():
                        viewer.opt.geomgroup[0] = 0
                        viewer.opt.geomgroup[1] = 1
                        viewer.cam.lookat[:] = [0., 0., .85]
                        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 135, -30
                last_skill = None
                def frame(scene, skill):
                    nonlocal last_skill
                    if skill != last_skill:
                        print(f"seed {seed}: {skill}", flush=True)
                        last_skill = skill
                    if viewer:
                        if not viewer.is_running():
                            raise KeyboardInterrupt
                        viewer.sync()
                        time.sleep(scene.control_timestep)
                    if writer:
                        renderer.update_scene(scene.sim.data._data, camera="agentview", scene_option=scene_option)
                        writer.append_data(renderer.render())
                library = TabletopSkills(env, frame, viewer.lock if viewer else nullcontext)
                try:
                    success = library.stack()
                    stacked = success
                    if success and args.task == "stack-and-relocate":
                        success = library.unstack_and_relocate()
                    row = dict(seed=seed, success=success, stacked=stacked, task=args.task, skills=library.events, steps=env.timestep)
                except RuntimeError as error:
                    row = dict(seed=seed, success=False, error=str(error), skills=library.events, steps=env.timestep)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            if writer:
                writer.close()
            if renderer:
                renderer.close()
            env.close()
    (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Success: {sum(r['success'] for r in rows)}/{len(rows)}. Saved: {output}")


if __name__ == "__main__":
    main()
