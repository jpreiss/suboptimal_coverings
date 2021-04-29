import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp


def plot(datapath, figpath):

    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["text.usetex"] = True  # for \textstyle

    df = pd.read_feather(datapath)

    # Make the names and values pretty.
    names = [f"$\\sigma_{i + 1}$" for i in range(4)]
    for i, name in enumerate(names):
        df[name] = df[f"sigma_{i}"]
    ONE = "$1$"
    TENTH = "$\\textstyle \\frac{1}{10}$"
    df = df.replace({
        1: ONE,
        0.1: TENTH,
    })

    # Set up grid.
    grid = sns.FacetGrid(
        df,
        row=names[0],
        col=names[1],
        row_order=[ONE, TENTH],
        height=0.82,
        aspect=1.05,
        despine=True,
        margin_titles=True,
    )

    # Our per-subplot function.
    def draw_heatmap(*args, **kwargs):
        data = kwargs["data"]
        table = data.pivot(index=args[1], columns=args[0], values=args[2])
        # Fix column order.
        table = table[[TENTH, ONE]]
        sns.heatmap(table, **kwargs)

    # Set up colormap. All this power stuff is to make 1.8-2.0 range look more distinct.
    power = 20.0
    def interp(a, b, mix):
        return (1.0 - mix ** power) * a + (mix ** power) * b
    deep_color = np.array([0.8, 0.8, 0.8])
    mixes = np.linspace(0.3 ** (1.0 / power), 1.0, 100)
    colors = [interp(np.ones(3), deep_color, m) for m in mixes]
    cmap = sns.blend_palette(colors)

    # Use common color scale.
    vmin = df["ratio"].min()
    vmax = df["ratio"].max()

    grid.map_dataframe(
        draw_heatmap,
        names[2],
        names[3],
        "ratio",
        cbar=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        linewidths=0.5,
    )

    plt.savefig(figpath)


if __name__ == "__main__":
    _, datapath, figpath = sys.argv
    plot(datapath, figpath)
