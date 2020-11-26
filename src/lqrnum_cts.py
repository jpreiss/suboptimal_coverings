"""Numerical (not symbolic) tools for discrete-time LQR problems."""

from collections import namedtuple
from copy import deepcopy
import itertools as it

import numpy as np
import scipy as sp
import scipy.linalg
#from scipy.linalg import solve_discrete_are, solve_continuous_are, so

from util import *
from lqrsym_cts import p_for_k, p_optimal_num

p_optimal = p_optimal_num


#
# LQR Problem class and helpers.
#

def lqrc(A, B, Q=None, R=None, return_P=False):
    """Find the infinite-horizon continuous-time optimal LQR controller."""
    # TODO always return P. Need to fix call sites.
    A, B, Q, R = lqrcheck(A, B, Q, R)
    P = sp.linalg.solve_continuous_are(A, B, Q, R)
    K = - np.linalg.inv(R) @ B.T @ P
    assert eigrealmax(A + B @ K) < 0
    if return_P:
        return K, P
    return K


def lqrd(A, B, Q, R):
    """Find the infinite-horizon discrete-time optimal LQR controller."""
    A, B, Q, R = lqrcheck(A, B, Q, R)
    P = sp.linalg.solve_discrete_are(A, B, Q, R)
    K = -np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    assert eigvalmax(A + B @ K)
    return K


def lqrcheck(A, B, Q=None, R=None):
    """Pre-processes arguments to LQR functions. Ensures matrix types."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    n, m = B.shape
    assert A.shape == (n, n)
    if Q is None:
        Q = np.eye(n)
    if R is None:
        R = np.eye(m)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)
    assert Q.shape == (n, n)
    assert R.shape == (m, m)
    return A, B, Q, R


def lqrc_cost(K, A, B, Q=None, R=None):
    """Computes the continuous-time LQR cost for a given controller."""

    A, B, Q, R = lqrcheck(A, B, Q, R)
    n, _ = B.shape

    eigs = np.linalg.eigvals(A + B @ K)
    if np.any(np.real(eigs) >= 0.0):
        return np.inf

    # Compute the Gramian
    #
    #         /∞
    #     W = |  e^t(A + BK) e^t(A + BK)^T dt
    #         /0
    #
    # by solving a Lyapunov equation.

    F = A + B @ K
    W = sp.linalg.solve_continuous_lyapunov(F, -np.eye(n))
    P = (Q + K.T @ R @ K) @ W

    # P is not the same as the P matrix we get from lqrc,
    # but it has the same trace.
    return P.trace()


def test_lqrc():
    npr = np.random.RandomState()
    for _ in range(100):
        m, n = npr.randint(1, 10, size=2)
        A = npr.normal(size=(n, n))
        B = npr.normal(size=(n, m))
        Q = random_pd(n, npr)
        R = random_pd(m, npr)
        K, P = lqrc(A, B, Q, R, return_P=True)
        cost = lqrc_cost(K, A, B, Q, R)
        assert np.isclose(cost, P.trace())




def k_for_p(a, b, p):
    return -b*p


def suboptimality(a, b, k):
    """Computes (cost of k) / (cost of optimal k) for a, b."""
    p = p_for_k(a, b, k)
    p_opt = p_for_k(a, b, lqrc(a, b))
    return p / p_opt


def b_suboptimality_interval(ratio, a, b, k=None):
    """Computes suboptimality interval w.r.t. b for k.

    The suboptimality interval is the largest open interval of b's such that k
    is no more than `ratio`-suboptimal for all b's in the interval.

    If argument `k` is not supplied, computes the lqr-optimal k for (a, b).
    """
    if k is None:
        k = lqrc(a, b)
    lb = condition_bisection(lambda r: suboptimality(a, b/r, k) < ratio, 1.0, 1e3)
    ub = condition_bisection(lambda r: suboptimality(a, b*r, k) < ratio, 1.0, 1e3)
    return b/lb, b*ub


def a_suboptimality_interval(ratio, a, b, k=None):
    """Computes suboptimality interval w.r.t. a for k.

    The suboptimality interval is the largest open interval of a's such that k
    is no more than `ratio`-suboptimal for all a's in the interval.

    If argument `k` is not supplied, computes the lqr-optimal k for (a, b).
    """
    if k is None:
        k = lqrc(a, b)
    lb = condition_bisection(lambda r: suboptimality(a/r, b, k) < ratio, 1.0, 1e3)
    ub = condition_bisection(lambda r: suboptimality(a*r, b, k) < ratio, 1.0, 1e3)
    return a/lb, a*ub


def covering(value_range, interval_fn):
    """Covers a closed interval using a mapping from points to neighborhoods.

    Uses a greedy algorithm, so output is not necessarily optimal.

    Args:
        value_range (Tuple[float, float]): Closed interval to cover.
        interval_fn (float -> Tuple[float, float]): Computes the covering
            neighborhood for a value in value_range.

    Returns:
        xs (List[float]): Collection such that the union of
            {interval_fn(x) : x in xs} contains value_range.
    """
    minimum, maximum = value_range
    xswitch = minimum
    xs = []
    while xswitch <= maximum:
        def overlaps(xx):
            lb, _ = interval_fn(xx)
            return lb < xswitch
        x = condition_bisection(overlaps, xswitch, maximum)
        _, xswitch = interval_fn(x)
        xs.append(x)
    return xs


def b_covering(a, theta, ratio):
    """Computes a *suboptimality covering* for b in [1/theta, theta].

    A suboptimality covering is a set of k's such that the union of the
    suboptimality intervals for all k's contains the range [1/theta, theta].

    This covering should be close to optimal.
    """
    def interval_fn(b):
        return b_suboptimality_interval(ratio, a, b)
    bs = covering((1.0 / theta, theta), interval_fn)
    ks = [lqrc(a, b) for b in bs]
    return bs, ks


def a_covering(b, theta, ratio):
    """Computes a *suboptimality covering* for a in [1/theta, theta].

    A suboptimality covering is a set of k's such that the union of the
    suboptimality intervals for all k's contains the range [1/theta, theta].

    This covering should be close to optimal.
    """
    def interval_fn(a):
        return a_suboptimality_interval(ratio, a, b)
    aas = covering((1.0 / theta, theta), interval_fn)
    ks = [lqrc(a, b) for a in aas]
    return aas, ks


