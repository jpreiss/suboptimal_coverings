import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import sympy.solvers


def riccati(a, b, q=1, r=1):
    p = sym.Symbol("p", real=True, positive=True)
    return p, 2*a*p - ((p*b)**2)/r + q


def p_optimal(a, b, q=1, r=1):
    p, p_eq0 = riccati(a, b, q, r)
    psoln = sym.solvers.solve(p_eq0, p)
    if len(psoln) > 1:
        return psoln[1]
    return psoln[0]


_a = sym.Symbol("_a", real=True, positive=True)
_b = sym.Symbol("_b", real=True, positive=True)
_q = sym.Symbol("_a", real=True, positive=True)
_r = sym.Symbol("_b", real=True, positive=True)
_k = sym.Symbol("_b", real=True, negative=True)

p_optimal_num = sym.lambdify((_a, _b), p_optimal(_a, _b))

def k_for_p(a, b, p, r=1):
    # Compute the optimal controller from P.
    k = -b*p / r
    return k


def k_optimal(a, b):
    return k_for_p(p_optimal(a, b))


def p_for_k(a, b, k, q=1, r=1):
    try:
        if a + b*k >= 0:
            return np.inf
    except TypeError:
        pass
    return (q + r*k**2) / (-2 * (a + b*k))


def b_for_k(a, k, q=1, r=1):
    b = sym.Symbol("b", real=True)
    cost = p_for_k(a=a, b=b, k=k, q=q, r=r)
    deriv = sym.diff(cost, k)
    soln = sym.solvers.solve(deriv, b)
    assert len(soln) == 1
    return soln[0]


def b_for_k_num(a, k, q=1, r=1):
    if a <= 0:
        raise ValueError("assumes a > 0.")
    if np.abs(k) <= 1.0:
        raise ValueError("optimal k must satisfy |k| > 1 when a > 0.")
    return _b_for_k_num(a, k, q, r)


def print_basic_results():
    
    a, b, q, r = sym.symbols("a b q r", real=True, positive=True)

    p = p_optimal(a, b, q=q)
    print("p_opt =")
    sym.pprint(p)

    k = k_for_p(a, b, p)
    print("k_opt =")
    sym.pprint(k)

    k = sym.Symbol("k", real=True)
    p_k = p_for_k(a, b, k)
    print("p for (a, b, k) =")
    sym.pprint(p_k)

    q = sym.Symbol("q", real=True, positive=True)
    b_k = b_for_k(a, k, q=q)
    print("b for which (a, k) is optimal =")
    sym.pprint(b_k)

    s = sym.Symbol("s", real=True, positive=True)
    b = sym.Symbol("b", real=True)
    cost = p_for_k(a=a, b=b, k=s*k, q=q)
    deriv = sym.diff(cost, k)
    soln = sym.solvers.solve(deriv, b)
    assert len(soln) == 1
    b_sk = soln[0]
    print("b for which (a, sk) is optimal =")
    sym.pprint(b_sk)

    print("ratio of bs")
    sym.pprint((b_sk / b_k).simplify())
    print("difference of bs")
    sym.pprint((b_sk - b_k).simplify())

    #dpdb = sym.diff(p_k, b)
    #print("dp/db =")
    #sym.pprint(dpdb)

    #d2pdb2 = sym.diff(dpdb, b)
    #print("d2p/db2 =")
    #sym.pprint(d2pdb2)



# TODO: use pytest.
def test():
    import numpy as np

    a = sym.Symbol("a", real=True, positive=True)
    b = sym.Symbol("b", real=True, positive=True)
    k = sym.Symbol("k", real=True)

    p_k = p_for_k(a, b, k)
    p_opt = p_optimal(a, b)
    k_opt = k_for_p(a, b, p_opt)
    p_k_opt = p_for_k(a, b, k_opt).simplify()
    assert p_k_opt == p_opt

    p_k_opt = p_k.subs(k, k_opt)
    error = p_k_opt - p_opt

    for aval, bval in 2.0 * np.random.random(size=(10,2)):
        errorval = error.subs(a, aval).subs(b, bval).evalf()
        assert(errorval < 10e-10)

    # Optimal k always has magnitude >= 1 when a > 0!
    for a, k in 2.0 * np.random.random(size=(1000, 2)):
        k = np.random.choice([-1, 1]) * (k + 1.0)
        b = b_for_k(a, k)
        p_k = p_for_k(a, b, k)
        p_opt = p_optimal_num(a, b)
        log_ratio = np.log(p_k / p_opt)
        assert np.abs(log_ratio) < 1e-8



if __name__ == "__main__":
    #state_bound()
    #test()
    #plot_intervals()
    #plot_theta_count()
    print_basic_results()
