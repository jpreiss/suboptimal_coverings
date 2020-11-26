"""
    [1] Optimal Guaranteed Cost Control and Filtering for Uncertain Linear
        Systems. Ian R. Petersen and Duncan C. McFarlane. IEEE Transactions on
        Automatic Control, Vol. 39, NO. 9, Sept. 1994.
"""


import itertools as it

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are
import scipy.optimize

from util import is_pd, is_psd, minimize_scalar
from lqrnum_cts import *
from quadrotor import *


def _gcc_synthesis(A, B1, B2, tau):
    """Solves a simplified GCC Riccati equation for our application."""

    INF = np.diag(np.repeat(np.inf, A.shape[0]))
    NULLK = np.zeros_like(B2).T
    if tau <= 0:
        return INF, NULLK

    BB = -(B1 @ B1.T) / tau + (B2 @ B2.T) / (tau + 1.0)

    # According to Petersen's GCC formula, we want to solve the ARE
    #
    #    A'P + PA - P BB P + Q = 0,
    #
    # where BB is the exact matrix above. However, SciPy's Riccati solver is
    # designed for standard LQR inputs, so it accepts separate matrices for an
    # ARE of the form
    #
    #    A'P + PA - P B R^{-1} B' P + Q = 0.
    #
    # We can't construct an appropriate B with a Cholesky decomposition because
    # our BB is positive semidefinite, but might not be positive definite.
    # Instead, we work with the eigendecomposition.
    evals, evecs = np.linalg.eigh(BB)

    # Huge eigenvalues will cause numerical problems in the Riccati solver.
    evalmax = np.amax(np.abs(evals))
    assert evalmax < 1e10

    # Allow slightly negative eigenvalues due to finite precision, but treat
    # them as zero when forming B.
    EPSILON = 1e-10
    if not np.all(evals >= -EPSILON):
        return INF, NULLK
    idx, = (evals > EPSILON).nonzero()
    B = evecs[:, idx]
    Rinv = np.diag(evals[idx])
    Rfake = np.diag(1.0 / evals[idx])
    assert np.all(np.isclose(BB.flat, (B @ Rinv @ B.T).flat)), "Low-rank PSD decomposition failed."

    n = A.shape[0]
    P = solve_continuous_are(A, B, np.eye(n), Rfake)
    K = -(1.0 / (1.0 + tau)) * B2.T @ P
    return P, K


def gcc_tauopt(A, B1, B2, fixed=False, bounds=None):
    """Finds an approximately optimal tau for a GCC Riccati equation."""

    if bounds is None:
        bounds = (0.1, 1.0)

    # TODO: Remove `fixed` once we're sure the optimization option is wokring.
    if fixed:
        assert False
        N = 10
        taus = np.geomspace(*bounds, N)
        Ps = [_gcc_synthesis(A, B1, B2, tau)[0] for tau in taus]
        traces = [P.trace() for P in Ps]
        best = np.argmin(traces)
        return Ps[best], taus[best]
    else:
        assert bounds is not None
        def f(x):
            P, _ = _gcc_synthesis(A, B1, B2, x)
            return min(P.trace(), 1e20)
        final_bounds = minimize_scalar(f, *bounds)
        tau = np.mean(final_bounds)
        assert tau != np.inf
        P, K = _gcc_synthesis(A, B1, B2, tau)
        return P, K, tau


def ddf_to_gcc(U, VT, sigma_min, sigma_max, onesided=False):
    """Converts a multi-LQR problem from decomposed form to GCC form."""

    if onesided:
        sigma_mid = sigma_max
        sigma_diff = sigma_max - sigma_min
    else:
        sigma_mid = (sigma_min + sigma_max) / 2.0
        sigma_diff = sigma_max - sigma_mid

    B2 = U @ np.diag(sigma_mid) @ VT
    B1 = U @ np.diag(sigma_diff) @ VT

    n, _ = U.shape
    diff = B2 @ B2.T - B1 @ B1.T + 1e-15 * np.eye(n)
    assert is_psd(diff)

    return B1, B2


def supergeomspace(start, end, n, power):
    assert n > 1
    base_powers = np.zeros(n)
    base_powers[0] = 1
    scale_powers = np.zeros(n)
    for i in range(n - 1):
        base_powers[i + 1] = base_powers[i] * power
        scale_powers[i + 1] = scale_powers[i] * power + 1
    log_base = np.log(start)
    log_scale = (np.log(end) - base_powers[-1] * log_base) / scale_powers[-1]
    log_x = scale_powers * log_scale + base_powers * log_base
    x = np.exp(log_x)
    assert len(x) == n
    assert np.isclose(x[0], start)
    assert np.isclose(x[-1], end)
    return x



def suboptimal_covering(A, U, VT, theta, alpha, divisions, bail_early=True):
    """Tries to construct an α-suboptimality covering for a DDF LQR problem.

    The LQR problem is defined by a fixed A matrix and a set of B matrices of
    the form U @ Σ @ VT, where Σ is a (d x d) diagonal matrix with diagonal
    entries between (1/theta) and 1.

    The covering is defined by a finite set of controllers K such that for any
    B, there is some K whose LQR cost exceeds the optimal controller's cost by
    a multiplicative factor no larger than alpha.

    We conjecture that a covering of size O((log theta)^d) always exists.
    Here, we try to construct the covering with a geometrically spaced grid
    for the diagonal entries of E.

    This routine only attempts to find a covering with a fixed number of
    divisions. An outer loop is needed to search for the minimal covering size.

    Returns True if a covering is found, False otherwise.
    """
    n, d = U.shape
    d2, m = VT.shape
    assert d2 == d
    assert m >= d
    assert A.shape == (n, n)
    Q = np.eye(n)
    R = np.eye(m)

    sigma_values = np.geomspace(1.0 / theta, 1.0, divisions + 1)
    #sigma_values = supergeomspace(1.0 / theta, 1.0, divisions + 1, power=(1+1.0/d))#np.sqrt(2))
    sigma_indices = np.array(list(it.product(range(divisions), repeat=d)))
    assert sigma_indices.shape == (divisions ** d, d)
    

    taus = set()

    ratios = np.zeros([divisions] * d)

    for idx in sigma_indices:
        sigma_min = sigma_values[idx]
        sigma_max = sigma_values[idx + 1]
        B1, B2 = ddf_to_gcc(U, VT, sigma_min, sigma_max)

        # The B with the most control authority will always achieve the lowest
        # optimal cost, so to check our fixed K against the whole hypercube of
        # systems, we only need to compare against the one B.
        B_best = U @ np.diag(sigma_max) @ VT
        P_best = solve_continuous_are(A, B_best, Q, R)
        P, K, tau = gcc_tauopt(A, B1, B2, fixed=False, bounds=(0.1, 1.0))

        B_worst = U @ np.diag(sigma_min) @ VT
        gcc_worstcase = P.trace()
        actual_worstcase = lqrc_cost(K, A, B_worst)
        #if gcc_worstcase != np.inf and gcc_worstcase > 1.01 * actual_worstcase:
            #print(f"GCC says cost is {gcc_worstcase:.1f}, "
                  #f"is actually {actual_worstcase:.1f}")

        ratio = gcc_worstcase / P_best.trace()
        assert ratio > 1, "GCC controller is better than single-system - shouldn't happen!"

        if ratio > alpha and bail_early:
            #print(f"θ = {theta}, divs = {divisions}: failed at")
            #print("Σ_min =")
            #print(sigma_min)
            #print("Σ_max =")
            #print(sigma_max)
            return False, None

        taus.add(tau)
        hypercube_index = tuple([i] for i in idx)
        ratios[hypercube_index] = ratio

    print(f"theta = {theta:.2f}, {divisions} divisions: taus used =")
    with np.printoptions(precision=3):
        print(np.array(sorted(taus)))

    assert not np.any(ratio.flat == 0)

    return True, ratios


def covering_number(A, U, VT, theta, alpha, max_covering=16):
    for n in range(1, max_covering + 1):
        if suboptimal_covering(A, U, VT, theta, alpha, n)[0]:
            return n
    return np.inf


def main():
    #A = np.eye(2)
    #U = np.eye(2)
    #VT = np.eye(2)
    A, U, VT = quad3d_decomposition()
    theta = 16.0
    alpha = 2.0
    #covering_number(A, U, VT, theta, alpha)
    print(supergeomspace(0.1, 1, 10))


if __name__ == "__main__":
    main()
