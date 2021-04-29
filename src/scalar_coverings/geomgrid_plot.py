import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main(inpath, outpath):
    plt.rcParams["figure.autolayout"] = False
    plt.rc(
        "figure.constrained_layout",
        use=True,
        h_pad=0.0,
        w_pad=0.0,
    )
    fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.4))

    xs = np.logspace(-1, 0, 9)
    for x in xs:
        ax.axhline(x, color="black", linewidth=0.5)
        ax.axvline(x, color="black", linewidth=0.5)

    plt.axis("square")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(xs[0], xs[-1])
    ticks = [xs[0], 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlabel("$\\sigma_1$")
    ax.set_ylabel("$\\sigma_2$")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.grid(False)

    fig.savefig(outpath)


if __name__ == "__main__":
    main(*sys.argv[1:])
