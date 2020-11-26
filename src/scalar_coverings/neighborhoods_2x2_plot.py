import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.ndimage

from util import with_stem


def scatter_lumped(x, y, **kwargs):
    # Messy hack, don't reuse
    mul_x = np.tile([1, 1, np.nan], len(x))
    mul_y = np.tile([1, 1.05, np.nan], len(x))
    x = np.repeat(x, 3) * mul_x
    y = np.repeat(y, 3) * mul_y
    return x, y


def saturate(color, amount=1.0):
    color = np.array(color)
    mean = color.mean()
    diff = color - mean
    new = amount * diff + mean
    return np.clip(new, 0, 1)


def heatmaps(data, x, y, hue, colors, **kwargs):
    lines = []
    for value in data[hue].unique():
        df = data[data[hue] == value]
        lines.append(np.array(scatter_lumped(
            df[x],
            df[y],
        )).T)
    group = matplotlib.collections.LineCollection(
        lines,
        color=colors,
        alpha=0.22,
        linewidths=1.0,
        rasterized=True,
    )
    plt.gca().add_artist(group)


def plot(datapath, figpath):

    matplotlib.use("pgf")
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["lines.solid_capstyle"] = "projecting"

    df = pd.read_feather(datapath)

    # LaTeX names for plotting instead of ASCII.
    # TODO: is there some way to attach a "plotting name" to columns?
    ALPHA = "$\\alpha$"
    A = "$A$"
    SIGMA_1 = "$\\sigma_1$"
    SIGMA_2 = "$\\sigma_2$"
    df = df.rename(columns={
        "alpha": ALPHA,
        "A": A,
        "sigma_1": SIGMA_1,
        "sigma_2": SIGMA_2,
    })
    df = df.replace({
        "eye": "$I$",
        "ones_scl": "$\\textstyle \\frac{1}{n} \\mathbf{1}$",
    })

    # Read the second data file containing the Sigma values for which the
    # covering controllers are optimal.
    sample_path = with_stem(datapath, "neighborhoods_2x2_sample")
    df_sample = pd.read_feather(sample_path)
    names = [f"K_{i}" for i in range(len(df_sample))]
    df_sample["name"] = names
    df_sample = df_sample.rename(columns={
        "sigma_1": SIGMA_1,
        "sigma_2": SIGMA_2,
    })

    # Convert main data from separate columns for each K into "tidy" form.
    df = df.melt(
        id_vars=[SIGMA_1, SIGMA_2, A, "opt"],
        value_vars=names,
        var_name="controller",
        value_name="cost",
    )

    # Copy the tidy data with multiple alpha values to show how neighborhoods
    # grow with alpha. TODO: Can pandas do this in a more database-like way?
    alphas = [1.05, 1.1, 1.35]
    dfs = [df.assign(**{ALPHA: a}) for a in alphas]
    df = pd.concat(dfs)

    # Compute suboptimality ratios and neighborhoods. Erase data points for
    # too-suboptimal points so we can use scatter plotting. Imshow-type plots
    # would be more correct, but getting the right tick labels, log axes,
    # coloring, etc. seemed too difficult.
    df["ratio"] = df["cost"] / df["opt"]
    assert np.all(df["ratio"] >= 1.0)
    df["mask"] = df["ratio"] <= df[ALPHA]
    df = df[df["mask"]]

    df["col"] = [
        f"$A = {{}}${A}\n$\\alpha = {{}}{alpha}$"
        for A, alpha in zip(df[A], df[ALPHA])
    ]

    colors = [saturate(c, 1.4) for c in sns.color_palette("tab10")]
    colors[3], colors[8] = colors[8], colors[3]
    colors = colors[:9]

    grid = sns.FacetGrid(
        data=df,
        col="col",
        col_order=sorted(df["col"].unique()),
        #col=[A, ALPHA],
        height=1.60,
        aspect=0.65,
        #margin_titles=True,
        #gridspec_kws=dict(
            #wspace=0.25,
        #),
    )

    grid.map_dataframe(
        heatmaps,
        x=SIGMA_1,
        y=SIGMA_2,
        hue="controller",
        colors=colors,
    )

    # Overlay markers on the Sigmas for which the controllers are optimal.
    for ax in grid.axes_dict.values():
        sns.scatterplot(
            ax=ax,
            data=df_sample,
            x=SIGMA_1,
            y=SIGMA_2,
            s=10,
            hue="name",
            palette=colors,
            legend=False,
        )

    # Style all facets.
    if False:
        grid.set_titles(template=(
            "{col_var}${{}} = {{}}${col_name}"
            "$,\\ $ "
            "{row_var}${{}} = {{}}${row_name}"
        ))
    else:
        grid.set_titles(template="{col_name}")
    sns.despine(left=True, bottom=True)
    for ax in grid.axes_dict.values():
        if ax.texts:
            ax.texts[0].set_rotation(0)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    sig_min = df_sample[SIGMA_1].min()
    sig_max = df_sample[SIGMA_1].max()
    pad_ratio = 1.8
    lims = [sig_min / pad_ratio, sig_max * pad_ratio]
    grid.set(
        xscale="log",
        yscale="log",
        xlim=lims,
        ylim=lims,
        xticks=lims,
        yticks=lims,
    )

    # DPI for our rasterized relplots must be high to get good quality.
    plt.savefig(figpath, dpi=1000)


if __name__ == "__main__":
    _, datapath, figpath = sys.argv
    plot(datapath, figpath)
