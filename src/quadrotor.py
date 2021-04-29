"""Constructs decomposed dynamics matrices of linearized quadrotor."""

import numpy as np


def _isdiag(A):
    return np.all(np.diag(np.diag(A)).flat == A.flat)


def _ctrb(A, B):
    """Computes the controllability matrix."""
    n, _ = A.shape
    return np.hstack([
        np.linalg.matrix_power(A, i) @ B
        for i in range(n)
    ])


def check_decomposition(func):
    """Decorator. Verifies returned A, U, VT are decomposed dynamics form."""
    def f(*args, **kwargs):
        A, U, VT = func(*args, **kwargs)
        assert _isdiag(VT @ VT.T)
        C = _ctrb(A, U @ VT)
        rank = np.linalg.matrix_rank(C)
        assert rank == A.shape[0]
        return A, U, VT
    return f


@check_decomposition
def quad2d_decomposition():
    """Constructs 2D multirotor linearized dynamics matrices.

    Linearizing the 2D multirotor makes the horizontal position uncontrollable,
    so we throw that state away. States are

        (altitude, vertical speed, angle, angular speed).

    Inputs are

        (left motor thrust, right motor thrust).

    Returns:
        A, (U, E, VT), where (U, E, VT) is the decomposed-dynamics form of the
        input matrix B.
    """
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ]).astype(np.float32)
    U = np.array([
        [0, 0],
        [1, 0],
        [0, 0],
        [0, 1],
    ]).astype(np.float32)
    VT = np.array([
        [ 1, 1],
        [-1, 1],
    ])
    return A, U, VT


@check_decomposition
def quad3d_decomposition(config="x"):
    """Constructs 3D quadrotor linearized dynamics matrices.

    Returns:
        A, U, VT
    """

    g = 1.0
    Z = np.zeros((3, 3))
    I = np.eye(3)
    G = np.array([
        [0, -g, 0],
        [g,  0, 0],
        [0,  0, 0],
    ])
    A = np.block([
        [Z, I, Z, Z],
        [Z, Z, Z, Z],
        [G, Z, Z, Z],
        [Z, Z, I, Z],
    ])
    if config == "+":
        VT = np.array([
            [ 1,  1,  1,  1],
            [-1,  0,  1,  0],
            [ 0, -1,  0,  1],
            [-1,  1, -1,  1],
        ])
    elif config == "x":
        VT = np.array([
            [ 1,  1,  1,  1],
            [ 1, -1, -1,  1],
            [-1, -1,  1,  1],
            [ 1, -1,  1, -1],
        ])
    else:
        raise ValueError('config must be "x" or "+"')
    U = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    return A, U, VT


if __name__ == "__main__":
    A, U, VT = quad3d_decomposition()
    print("OK")
