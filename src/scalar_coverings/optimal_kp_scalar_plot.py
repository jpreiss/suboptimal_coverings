import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(datapath, figpath):
    df = pd.read_feather(datapath)

    # Matplotlib doesn't make it easy to draw multiple line plots
    # with each plot's color determined by a colormap.
    avals = np.unique(df["a"])
    avals = avals[~np.isnan(avals)]
    norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(avals))
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
    cmap.set_array([])

    # TODO figure out how to remove the padding between colorbar and axes.
    gs_kw = dict(width_ratios=[3, 3, 0.2])

    # The width should really be 5.5, but pgf output somehow still gets padded
    fig, axs = plt.subplots(1, 3, figsize=(5.8, 2.3), gridspec_kw=gs_kw)

    for ax in axs[:-1]:
        ax.set_xlabel("B")
        ax.set_xlim([-1.25, 1.25])

    ax_k, ax_p, ax_c = axs

    for a in avals:
        df2 = df[df["a"] == a]
        ax_k.plot(df2["b"], df2["k"], c=cmap.to_rgba(a))
        ax_p.plot(df2["b"], df2["p"], c=cmap.to_rgba(a))

    ax_k.set_ylabel("optimal K")
    ax_k.set_ylim([-2.5, 2.5])

    ax_p.set_ylabel("optimal P")
    ax_p.set_ylim([1.0, 6.0])

    fig.colorbar(cmap, ticks=(0, 1, 2), label="A", cax=ax_c, pad=0.0)
    fig.tight_layout()

    plt.savefig(figpath, bbox_inches="tight", pad_inches=0.0)


if __name__ == "__main__":
    main(*sys.argv[1:])
