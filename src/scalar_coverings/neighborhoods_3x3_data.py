import itertools as it
import multiprocessing
import sys

import numpy as np
import pandas as pd

from lqrnum_cts import *
from util import with_stem


def opt_cost(A, B):
    _, P = lqrc(A, B, return_P=True)
    return P.trace()


def main(outpath):
    theta = 100.0
    GRID = 80

    # Synthesize optimal K for low authority in all dimensions.
    A = np.eye(3)
    #A = np.random.normal(size=(3, 3))
    #A = np.ones((3, 3))
    B = (2.0 / theta) * np.eye(3)
    K = lqrc(A, B)

    # Evaluate K vs. optimal on (mostly) higher-authority scenarios.
    values = np.geomspace(1.0 / theta, 1.0, GRID)
    sigmas = np.stack(list(it.product(values, repeat=3)))
    Bs = [np.diag(sigma) for sigma in sigmas]

    pool = multiprocessing.Pool(10)
    costs = pool.starmap(lqrc_cost, [(K, A, B) for B in Bs])
    opt_costs = pool.starmap(opt_cost, [(A, B) for B in Bs])

    df = pd.DataFrame({
        **{f"sigma_{i}": sigmas[:, i] for i in range(3)},
        "cost": costs,
        "opt_cost": opt_costs,
    })
    df.to_feather(outpath)


if __name__ == "__main__":
    main(sys.argv[1])
