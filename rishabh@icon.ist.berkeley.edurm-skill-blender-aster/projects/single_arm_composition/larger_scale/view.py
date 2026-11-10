"""Inspect the learned policy and the recorded expert from a matched reset."""
from contextlib import ExitStack, contextmanager
import time
from pathlib import Path

import mujoco as mj

from .protocol import read_manifest
from .rollout import replay, run, save_recording
from .workcell import Workcell


@contextmanager
def frames(args, cell):
    with ExitStack() as stack:
        viewer = renderer = writer = None
        camera = mj.MjvCamera()
        mj.mjv_defaultFreeCamera(cell.model, camera)
        camera.lookat[:] = (.36, 0., .30)
        camera.distance = 1.48
        camera.azimuth = 132
        camera.elevation = -28
        scene_option = mj.MjvOption()
        mj.mjv_defaultOption(scene_option)
        scene_option.geomgroup[3] = 1  # Panda's Menagerie collision meshes.
        if args.show:
            from mujoco import viewer as mj_viewer

            viewer = stack.enter_context(mj_viewer.launch_passive(cell.model, cell.data))
            mj.mjv_defaultFreeCamera(cell.model, viewer.cam)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            viewer.opt.geomgroup[3] = 1
        if args.video:
            import imageio.v2 as imageio

            video_path = Path(args.video)
            video_path.parent.mkdir(parents=True, exist_ok=True)
            renderer = stack.enter_context(mj.Renderer(cell.model, height=720, width=1120))
            writer = stack.enter_context(imageio.get_writer(str(video_path), fps=30))
        previous = -1.
        wall_start = time.monotonic()

        def draw(current):
            nonlocal previous
            now = current.data.time
            if previous >= 0 and now - previous < (args.speed / 30):
                return
            previous = now
            if renderer is not None:
                renderer.update_scene(current.data, camera=camera, scene_option=scene_option)
                writer.append_data(renderer.render())
            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt("viewer closed")
                if current.data is not viewer_data:
                    with viewer.lock():
                        viewer_data.qpos[:] = current.data.qpos
                        viewer_data.qvel[:] = current.data.qvel
                        mj.mj_forward(cell.model, viewer_data)
                viewer.sync()
                delay = now / args.speed - (time.monotonic() - wall_start)
                if delay > 0:
                    time.sleep(delay)

        viewer_data = cell.data
        yield draw if renderer is not None or viewer is not None else None


def view(args):
    if args.speed <= 0:
        raise ValueError("speed must be positive")
    _, _, manifest = read_manifest(args.manifest, "dev")
    locked = None
    if args.split != "dev":
        from .evaluate import read_lock

        locked = read_lock(args.lock, manifest)
    graphs, items, _ = read_manifest(args.manifest, args.split)
    item = next((item for item in items if item.task_id == args.task), None)
    if item is None:
        raise ValueError(f"unknown task ID {args.task} in {args.split}")
    directory = Path(args.recordings)
    archive = directory / f"{item.task_id}-r{args.reset}.npz"
    if args.real_demo:
        if not archive.exists():
            raise FileNotFoundError(f"expert recording missing at {archive}. Run --record-demo first.")
        with frames(args, Workcell(item.layout)) as frame:
            result = replay(archive, item, manifest, frame=frame)
        if (result["reset"], result["method"], result.get("lock")) != (
                args.reset, "expert", locked["sha256"] if locked else None):
            raise ValueError("requested expert replay and archive provenance differ")
        return result
    selector = None
    model = None
    if args.model:
        from .learn import checkpoint_sha, load_selector

        sha = checkpoint_sha(args.model)
        if locked is not None and sha not in (
                row["sha256"] for row in locked["contract"]["models"]):
            raise ValueError("model is absent from the frozen evaluation lock")
        selector, meta = load_selector(args.model, manifest)
        model = {"sha256": sha, "width": meta["width"], "seed": meta["seed"],
                 "episodes": meta["fit"]["episodes"], "parameters": meta["parameters"]}
    cell = Workcell(item.layout)
    with frames(args, cell) as frame:
        if frame is not None:
            advance = cell.advance

            def rendered_advance(ticks, *, panel=False):
                advance(ticks, panel=panel)
                frame(cell)

            cell.advance = rendered_advance
        result, trace = run(graphs[item.graph_id], item, args.reset, manifest,
                            "learned" if args.model else "expert", selector=selector,
                            record=args.record_demo, cell=cell)
    result.update(checkpoint=model, lock=locked["sha256"] if locked else None)
    if args.record_demo:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix(".tmp.npz")
        save_recording(temporary, result, trace)
        replay(temporary, item, manifest)
        temporary.replace(archive)
    return result
