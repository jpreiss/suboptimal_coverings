import itertools as it
import pathlib

import numpy as np


#
# Coding utilities.
#

def flatten(list_of_lists):
    return list(it.chain.from_iterable(list_of_lists))


def with_stem(path, stem):
    path = pathlib.Path(path)
    return path.with_name(stem).with_suffix(path.suffix)


def condition_bisection(pred, x0, xmax):
    """Finds argmax of x in [x0, xmax] s.t. pred(x). Assumes monotonicity."""
    # TODO: this is a "geometric" version of the algorithm.
    # Do we also want a linear version? Maybe linear is all we need?
    if not pred(x0):
        raise ValueError("Expects predicate to be true for x0.")
    if x0 < 0:
        raise NotImplementedError("x0 and xmax must be positive.")
    if pred(xmax):
        return xmax
    lb = x0
    ub = xmax
    x = geom_mean((lb, ub))
    # TODO: expose precision parameter that is interpretable.
    while np.log(ub / lb) > 1e-6:
        if pred(x):
            lb = x
        else:
            ub = x
        x = geom_mean((lb, ub))
    return x


def minimize_scalar(f, a, b, tol=1e-5):
    """Golden-section search.

    Given a function f with a single local minimum in the interval [a,b], gss
    returns a subset interval [c,d] that contains the minimum with d-c <= tol.

    From: Wikipedia
    """

    invphi = (np.sqrt(5) - 1) / 2  # 1 / phi
    invphi2 = (3 - np.sqrt(5)) / 2  # 1 / phi^2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # Required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for k in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    if yc < yd:
        return (a, d)
    else:
        return (c, b)


#
# Linear algebra utilities.
#

def is_psd(A):
    """Returns true if A is symmetric positive semidefinite."""
    return np.all(A.flat == A.T.flat) and np.all(np.linalg.eigvalsh(A) >= 0)


def is_pd(A):
    """Returns true if A is symmetric positive definite."""
    return np.all(A.flat == A.T.flat) and np.all(np.linalg.eigvalsh(A) > 0)


def eigvalmax(A):
    """Returns max {|z| : z is eigenvalue of A}. || is complex magnitude."""
    mags = np.abs(np.linalg.eigvals(A))
    return np.amax(mags)


def eigrealmax(A):
    """Returns max {Re[z] : z is eigenvalue of A}."""
    reals = np.real(np.linalg.eigvals(A))
    return np.max(reals)


def opnorm(A):
    """Returns the operator norm: max { |Ax| : |x| = 1 }. || is L2 norm."""
    E = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    return E[0]


def random_pd(dim, npr):
    """Returns a random postiive semidefinite matrix.

    Args:
        dim: The dimensionality of A.
        npr: A numpy.random.RandomState instance.

    Returns:
        A: A PSD matrix such that x^T A x will have magnitude approximately 1
           if x is zero-mean and unit-variance.
    """
    A = npr.normal(size=(dim, dim))
    return A.T @ A / dim


def quadform(x, Q):
    """Computes N quadratic forms.

    Args:
        x: [n, k] array of vectors.
        Q: [k, k] quadratic form matrix.

    Returns:
        x^T Q x : [n] array of quadratic form values.
    """
    return (x @ Q * x).sum(axis=1)


def geom_mean(x):
    """Compute the geometric mean using log-transform to avoid overflow."""
    return np.exp(np.mean(np.log(x)))


def random_K_eigvalmax(npr, A, B, eigmax):
    """Samples an i.i.d. random matrix such that eigvalmax(A - BK) is < eigmax.

    Uses unsophisticated rejection sampling method.

    Args:
        npr: A numpy.random.RandomState instance.
        A: [n, n] matrix with operator norm around 1.
        B: [m, n] matrix with operator norm around 1.
        eigmax: Maximum eigenvalue magnitude of A - BK.

    Returns:
        K: Matrix such that 0.8 * eigmax < eigvalmax(A - BK) < eigmax.
    """
    for i in range(10000):
        n, m = B.shape
        K = 2.0 * eigmax * npr.normal(size=(m, n)) / np.sqrt(n)
        F = A - B @ K
        emax = eigvalmax(F)
        if 0.8 * eigmax < emax < eigmax:
            return K
    raise ValueError("getting desired eigmax", eigmax, "is difficult")


def argmax_inner(x, p):
    """Computes the vector of l_p norm 1 maximizing the inner product with x.

    p may be one of {1, 2, np.inf}.
    """
    n = x.size
    if p == 1:
        i = np.argmax(np.abs(x))
        return np.sign(x[i]) * basis(n, i)
    if p == 2:
        return x / np.linalg.norm(x)
    if p == np.inf:
        return np.sign(x)
    raise ValueError("p must be one of {1, 2, infinity}.")


def basis(n, e):
    """Constructs the e'th basis vector in dimension n."""
    x = np.zeros(n)
    x[e] = 1.0
    return x


def randsphere(npr, n):
    """Samples a random vector on the n-dimensional Euclidean unit sphere."""
    x = npr.normal(size=n)
    x /= np.linalg.norm(x)
    return x
