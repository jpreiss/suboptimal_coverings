import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import vispy as vis
import vispy.scene

from videowriter import VideoWriter
from util import with_stem


def plot(datapath, figpath):

    df = pd.read_feather(datapath)
    df["ratio"] = df["cost"] / df["opt_cost"]

    # Reconstruct 3d volume
    grid = len(df["sigma_0"].unique())
    table = df.pivot_table(index=["sigma_0", "sigma_1", "sigma_2"], values="ratio")
    ratio = table.values.reshape((grid, grid, grid))

    # Add a "shell" of infinite cost to get a watertight mesh.
    ratio = np.pad(ratio, ((1, 1),), constant_values=10000)

    # Reflect the data to make the default VisPy light hit the inside.
    ratio = np.flip(ratio, axis=(0, 1))

    # Set up the canvas/scene.
    vis.use("glfw")
    imgsize = (1080, 1080)
    canvas = vis.scene.SceneCanvas(size=imgsize, resizable=False)
    camera = vis.scene.TurntableCamera(azimuth=290, elevation=25)
    camera.set_range((0, grid), (0, grid), (-grid/10.0, grid), margin=0)
    view = canvas.central_widget.add_view()
    view.camera = camera
    view.bgcolor = 1.0 * np.ones(3)

    # Set up the "to-do list" of frames.
    if os.getenv("LQR_FAST") is not None:
        fps = 1
        alphas = [1.0367, 1.0526, 1.0852, 1.1137, 1.2000]
    else:
        fps = 30
        seconds = 4
        n_alphas = seconds * fps
        alpha_min = 1.001
        alpha_max = 1.2
        alphas = np.geomspace(alpha_min, alpha_max, n_alphas)

    # Figure out where we're going to write files. Is a little messy because we
    # want to play nicely with the makefile figure generation, which only gives
    # us one filename based on the name of this code file.
    figpath = Path(figpath)
    subdir = figpath.parent / figpath.stem
    os.makedirs(subdir, exist_ok=True)
    figpath.touch() # So make believes we did something.

    # Open ffmpeg subprocess.
    videopath = subdir / "neighborhoods_3x3.mp4"
    writer = VideoWriter(str(videopath), dt=1.0/fps, shape=imgsize[::-1])

    # Store all the uncompressed frames so we can append them to the video in
    # reverse and get a perfect loop.
    frames = []

    # Add blank frame at start because vispy won't make an empty isosurface.
    buf = canvas.render()[:, :, :3]
    frames.append(buf)
    writer.writeFrame(buf)

    for alpha in alphas:
        mesh = vis.scene.visuals.Isosurface(
            ratio,
            alpha,
            shading="smooth",
            color=0.5*np.ones(3),
        )
        mesh.unfreeze()
        mesh.shininess = 0.0
        view.add(mesh)

        # TODO: figure out why text widget below doesn't work.
        """
        message = f"α = {alpha:.2f}"
        text = vis.scene.visuals.Text(
            "HELLO WORLD",
            pos=(0, 0, 0),
            font_size=100,
        )
        view.add(text)
        """

        # For some reason, offscreen rendering using canvas.render() produces
        # glitches on my machine, but copying the framebuffer with
        # _screenshot() while an actual GL window is open works fine.
        canvas.show()
        canvas.app.process_events()
        buf = vis.gloo.util._screenshot()[:, :, :3]
        frames.append(buf)
        writer.writeFrame(buf)

        mesh.parent = None
        #text.parent = None

    for buf in frames[::-1]:
        writer.writeFrame(buf)

    paths = set()
    for alpha, frame in zip(alphas, frames[1:]):
        imgpath = subdir / f"alpha_{alpha:.4f}.png"
        # Make sure float truncation isn't causing overwritten files.
        assert imgpath not in paths
        paths.add(imgpath)
        Image.fromarray(frame).save(imgpath)

    writer.close()


if __name__ == "__main__":
    _, datapath, figpath = sys.argv
    plot(datapath, figpath)
