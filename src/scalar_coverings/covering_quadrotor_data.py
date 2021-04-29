import itertools as it
import multiprocessing
import sys

import numpy as np
import pandas as pd

from gcc import covering_number
from quadrotor import *


def main(outpath):
    alpha = 2.0
    A, U, VT = quad3d_decomposition()
    thetas = np.geomspace(1, 100, 25)

    pool = multiprocessing.Pool(10)
    N_max = 20
    args = [(A, U, VT, theta, alpha, N_max) for theta in thetas]
    covering_nums = pool.starmap(covering_number, args)
    df = pd.DataFrame({
        "theta": thetas,
        "covering_num": covering_nums
    })
    df.to_feather(outpath)


if __name__ == "__main__":
    main(*sys.argv[1:])
