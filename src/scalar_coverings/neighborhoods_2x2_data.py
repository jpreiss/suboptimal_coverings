import itertools as it
import multiprocessing
import sys

import numpy as np
import pandas as pd

from lqrnum_cts import *
from util import with_stem


def ones_scl(n):
    return np.ones((n, n)) / float(n)

def worker(A, sigma, sample_Ks, extra):
    B = np.diag(sigma)
    Kopt, Popt = lqrc(A, B, return_P=True)
    Jopt = Popt.trace()
    assert np.isclose(lqrc_cost(Kopt, A, B), Jopt)
    costs = np.array([lqrc_cost(K, A, B) for K in sample_Ks])
    assert not np.any(costs < Jopt)
    return {
        "sigma_1": sigma[0],
        "sigma_2": sigma[1],
        "opt": Jopt,
        **{f"K_{i}": cost for i, cost in enumerate(costs)},
        **extra,
    }

def main(outpath):
    A_funcs = (np.eye, ones_scl)
    theta = 32.0
    records = []

    GRID = 200

    #SAMPLES = 10
    #sample_logs = np.random.uniform(-np.log(theta), 0.0, size=(SAMPLES, 2))
    #sample_sigmas = np.exp(sample_logs)

    SAMPLE_GRID = 3
    sample_vals = np.geomspace(2.0 / theta, 0.5, SAMPLE_GRID)
    sample_sigmas = np.stack(list(it.product(sample_vals, repeat=2)))
    # add jitter
    if False:
        jitter = np.random.uniform(-1, 1, size=sample_sigmas.shape)
        sample_sigmas *= 1.2 ** jitter
    SAMPLES = sample_sigmas.shape[0]


    vals = np.geomspace(1.0 / theta, 1.0, GRID)
    sigmas = np.stack(list(it.product(vals, vals)))

    args = []
    for A_func in A_funcs:
        A = A_func(2)
        sample_Ks = [lqrc(A, np.diag(sigma)) for sigma in sample_sigmas]
        args += [
            (A, sigma, sample_Ks, {"A": A_func.__name__})
            for sigma in sigmas
        ]

    pool = multiprocessing.Pool(10)
    records = pool.starmap(worker, args)

    assert len(records) == GRID * GRID * len(A_funcs)
    df = pd.DataFrame(records)
    df.to_feather(outpath)

    sample_idx = np.stack(it.product(range(SAMPLE_GRID), repeat=2))
    df_sample = pd.DataFrame({
        "sigma_1": sample_sigmas[:, 0],
        "sigma_2": sample_sigmas[:, 1],
        "sigma_1_idx": sample_idx[:, 0],
        "sigma_2_idx": sample_idx[:, 1],
    })
    sample_path = with_stem(outpath, "neighborhoods_2x2_sample")
    df_sample.to_feather(sample_path)


if __name__ == "__main__":
    main(sys.argv[1])
