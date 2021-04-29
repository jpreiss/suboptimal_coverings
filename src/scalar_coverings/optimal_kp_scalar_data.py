"""Illustrate that the universal LQR controller is discontinuous."""

import numpy as np
import pandas as pd
import sympy as sym
import matplotlib as mpl
import sys

import matplotlib.pyplot as plt

from lqrsym import *


def main(outpath):

    # The scalar LQR dynamics "matrices".
    a = sym.Symbol("a", real=True)
    b = sym.Symbol("b", real=True)

    p = p_optimal(a, b)
    k = k_optimal(a, b, p)

    # Draw one plot of optimal K vs. B for each value of A.
    num_a = 12
    max_a = 2.0
    avals = np.arange(max_a, 0, -max_a / num_a)
    bvals = np.linspace(-1.25, 1.25, 201)

    records = []

    for i, aval in enumerate(avals):
        k_sub_a = k.subs(a, aval)
        p_sub_a = p.subs(a, aval)

        for bval in bvals:
            pval = float(p_sub_a.subs(b, bval))

            kval = k_sub_a.subs(b, bval)
            assert np.imag(kval) == 0
            kval = np.real(np.complex(kval))

            records.append({
                "a": aval,
                "b": bval,
                "k": kval,
                "p": pval,
            })

    df = pd.DataFrame(records)
    df.to_feather(outpath)


if __name__ == "__main__":
    main(sys.argv[1])
