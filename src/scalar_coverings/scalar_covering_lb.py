"""Symbolic computations suboptimality ratio lower bound.

Computes second derivative for strict convexity and solves for α-level set
boundaries.
"""

import sympy as sym

from lqrsym_cts import *


def main():
    a, b, alpha = sym.symbols("a b alpha", real=True, positive=True)
    k = sym.Symbol("k", real=True, negative=True)

    # Lower bound for J_b(k)
    Jbk_lb = -(k**2)/(2*(a + b*k))

    # Upper bound for J^star_b
    Jopt_ub =  (3*a)/(b**2)

    ratio = Jbk_lb / Jopt_ub
    print("Suboptimality ratio lower bound:")
    sym.pprint(ratio)

    eq0 = ratio - alpha
    solns = sym.solvers.solve(eq0, b)
    solns = [s.simplify() for s in solns]
    print("solutions for ratio = α:")
    for s in solns:
        sym.pprint(s)

    drdb = ratio.diff(b)
    d2rdb2 = drdb.diff(b).simplify()
    num, denom = sym.fraction(d2rdb2)
    num2 = sym.factor(num)
    d2rdb2 = num2 / denom
    print("2nd deriv of ratio wrt b:")
    sym.pprint(d2rdb2)


if __name__ == "__main__":
    main()
