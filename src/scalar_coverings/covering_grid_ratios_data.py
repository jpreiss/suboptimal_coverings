import itertools as it
import sys

import numpy as np
import pandas as pd

from gcc import *
from quadrotor import *


def main(outpath):
    alpha = 2.0
    A, U, VT = quad3d_decomposition()
    theta = 10.0
    k = 4
    ok, ratios = suboptimal_covering(A, U, VT, theta, alpha=alpha, divisions=k, bail_early=False)
    assert ok
    d = U.shape[1]
    corner_idx = np.stack(list(it.product([0, -1], repeat=d))).T
    corner_ratios = ratios[tuple(corner_idx)]
    sigmas = np.array([1.0 / theta, 1.0])
    corner_sigmas = [sigmas[corner_idx[i]] for i in range(d)]
    assert corner_idx.shape == (d, 2 ** d)
    df = pd.DataFrame({
        **{f"idx_{i}": corner_idx[i] for i in range(d)},
        **{f"sigma_{i}": corner_sigmas[i] for i in range(d)},
        "ratio": corner_ratios
    })
    df.to_feather(outpath)


if __name__ == "__main__":
    main(*sys.argv[1:])
