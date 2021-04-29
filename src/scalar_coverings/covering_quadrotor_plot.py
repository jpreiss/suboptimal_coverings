
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import scipy.stats


COVERING = "grid pitch $k$" #empirical $\\coveringnum{\\subopt}{\\Envs}$"
THETA = "$\\theta$"


def plot(datapath, figpath):

    matplotlib.use("pgf")

    df = pd.read_feather(datapath)
    df[THETA] = df["theta"]
    df[COVERING] = df["covering_num"]

    # The logarithmic dependence doesn't take effect for small theta.
    fit_cutoff = 3.0
    df["use_fit"] = df[THETA] >= fit_cutoff

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.65), tight_layout=True)

    sns.scatterplot(
        data=df,
        ax=ax,
        x=THETA,
        y=COVERING,
        s=18,
        markers=True,
        hue="use_fit",
        palette=[
            (0.0, 0.0, 0.0),
            (0.75, 0.75, 0.75),
        ],
        legend=False,
    )

    thetas = np.array(df[THETA])[df["use_fit"]]
    ys = np.array(df[COVERING])[df["use_fit"]]

    thetas_fine = np.geomspace(fit_cutoff, df[THETA].max(), 200)
    names = ("$\\log$", "$\\log^{3/2}$", "$\\log^4$")
    linestyles = ("-", "-", ":")
    funcs = (np.log, lambda x: np.log(x) ** 1.5, lambda x: np.log(x) ** 4)

    # Delete this line to re-enable other fits.
    funcs = funcs[:1]

    # Linear least squares fits.
    for name, func, linestyle in zip(names, funcs, linestyles):
        xs = func(thetas)
        slope, intercept, _, _, _ = sp.stats.linregress(xs, ys)
        predictions = slope * func(thetas_fine) + intercept
        ax.plot(thetas_fine, predictions, label=name, color="black", linestyle=linestyle, linewidth=1)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.00, 1.0, 0.9),
        handlelength=1.35,
        labelspacing=0.2,
        frameon=False,
    )

    sns.despine(ax=ax, left=True, bottom=True)
    ax.set_xscale("log")
    ax.set_yticks([1, 5, 10])
    fig.savefig(figpath)


if __name__ == "__main__":
    _, datapath, figpath = sys.argv
    plot(datapath, figpath)
