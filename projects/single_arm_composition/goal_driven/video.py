"""Render the robot and the selector's current decision together."""

from contextlib import contextmanager
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@contextmanager
def video_frames(cell, path, speed, goal):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    camera = mj.MjvCamera()
    mj.mjv_defaultFreeCamera(cell.model, camera)
    camera.lookat[:] = (.36, 0., .30)
    camera.distance, camera.azimuth, camera.elevation = 1.48, 132, -28
    options = mj.MjvOption()
    mj.mjv_defaultOption(options)
    options.geomgroup[3] = 1
    font = ImageFont.load_default(size=18)
    title = ImageFont.load_default(size=23)
    previous = -1.
    with mj.Renderer(cell.model, height=640, width=1120) as renderer, imageio.get_writer(target, fps=30) as writer:
        def draw(choice, *, hold=0):
            nonlocal previous
            now = cell.data.time
            if not hold and previous >= 0 and now - previous < speed / 30:
                return
            previous = now
            renderer.update_scene(cell.data, camera=camera, scene_option=options)
            rendered = Image.fromarray(renderer.render())
            image = Image.new("RGB", (rendered.width, rendered.height + 80), "#16242e")
            image.paste(rendered, (0, 80))
            text = ImageDraw.Draw(image)
            text.text((18, 10), "Goal   " + goal, font=title, fill="#eef5fb")
            text.text((18, 46), choice, font=font, fill="#a6dce7")
            pixels = np.asarray(image)
            for _ in range(max(1, hold)):
                writer.append_data(pixels)
        yield draw
