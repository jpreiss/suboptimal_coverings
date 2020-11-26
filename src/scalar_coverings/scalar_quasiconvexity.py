"""Symbolic math for showing that suboptimality ratio is quasiconvex in b.

Finds the unique stationary point and shows that the second derivative is
positive there. Prints LaTeX proof to stdout.
"""

import sympy as sym

from lqrsym_cts import *


# Enable for a proof with more steps made explicit.
VERBOSE = False


def main():
    a, b, alpha = sym.symbols("a b alpha", real=True, positive=True)
    k = sym.Symbol("k", real=True, negative=True)
    # For easier access in substitutions, etc.
    sq = sym.sqrt(a**2 + b**2)

    Jbk = p_for_k(a, b, k)
    Jopt = p_optimal(a, b)
    ratio = Jbk / Jopt

    # SymPy likes to expand this; we don't.
    def subs_2abk(expr):
        return expr.subs(
            -2*a - 2*b*k,
            sym.Mul(-2, sym.Add(a, sym.Mul(b, k)), evaluate=False)
        )

    # Applies sign-preserving changes to the first derivative to find
    # stationary points more easily. For some reason, SymPy doesn't want to
    # find and divide these factors if we apply it to the whole expression, so
    # we need to do it term by term.
    def sign_preserve(term):
        term = term / (b * (k**2 + 1))
        term = term * 2 * (a + b*k)**2
        term = term * (a + sq)**2
        term = term * sq
        term = term / a
        return term

    dratio = subs_2abk(ratio.diff(b))
    assert isinstance(dratio, sym.Add)
    eq0 = sym.Add(*(sign_preserve(term) for term in dratio.args))
    # For printing in LaTeX output.
    factor = sym.latex(sign_preserve(1))

    # Express stationary points as lhs = rhs instead of eq0 = 0.
    eq0 = sym.collect(eq0.simplify(), sq)
    eq0 = -eq0
    rhs = subs_2abk(-eq0.coeff(sq) * sq)
    lhs = (eq0 + rhs).simplify()

    # We only consider the domain k > 1, so we can make some simplifications
    # that SymPy won't do on its own.
    def simplify_k1(expr):
        original = expr
        for _ in range(10):
            expr = expr.subs(abs(k**2 - 1), k**2 - 1)
            expr = expr.subs(sym.sqrt(k**4 + 2*k**2 + 1), k**2 + 1)
            expr = expr.simplify()
            if expr == original:
                return expr
            original = expr
        raise ValueError("Stuck in simplify loop.")

    if VERBOSE:
        print(
f"""The second derivative of $r$ is
\\begin{{dmath*}}
    \\frac{{\\partial r}}{{\\partial b}} = {sym.latex(eq0)}.
\\end{{dmath*}}
To solve $\\partial r / \\partial b = 0$ for $b$,
we multiply by the strictly positive factor
\\[
    {factor}
\\]
to get
\\[
    {sym.latex(lhs)} = {sym.latex(rhs)}.
\\]"""
        )
    else:
        print(
f"""To solve $\\partial r / \\partial b = 0$ for $b$,
we multiply $\\partial r/ \\partial b$ (not shown due to length)
by the strictly positive factor
\\[
    {factor}
\\]
and set the result equal to zero to get the equation
\\[
    {sym.latex(lhs)} = {sym.latex(rhs)}.
\\]"""
        )

    # Square both sides.
    lhs = lhs ** 2
    rhs = rhs ** 2
    eq0_sq = lhs.expand() - rhs.expand()
    eq0_sq = (eq0_sq / b**3).simplify()

    if VERBOSE:
        print(
f"""Squaring both sides (which may introduce spurious solutions) yields
\\[
    {sym.latex(lhs)} = {sym.latex(rhs)}.
\\]
Expanding and collecting terms for $b$ yields
\\[
    {sym.latex(eq0)} = 0,
\\]"""
        )
    else:
        print(
f"""Squaring both sides (which may introduce spurious solutions)
and collecting terms yields the equation
\\(
    {sym.latex(eq0_sq)} = 0,
\\)"""
        )

    # Solve.
    soln = sym.solve(eq0_sq, b)
    assert len(soln) == 1
    soln = soln[0]
    # Coerce into a more elegant form.
    soln = -soln.subs(k**2 - 1, 1 - k**2)

    # Ensure we didn't find a spurious solution by squaring both sides.
    check_dratio = simplify_k1(eq0.subs(b, soln))
    assert check_dratio == 0

    print(
f"""with the solution $b = {{}}$ {sym.latex(soln, mode="inline", fold_short_frac=True)}.
This is the expression for $b_k$ from \\lemmaref{{lem:scalar-lqr-facts}}.
Note that it is only positive for $k < -1$.
If $k \in [-1, 0)$, then there are no stationary points in $\\bdomain_k$.
Otherwise, substitution into $\\partial r / \\partial b$ confirms that this solution is not spurious,
so it is the only stationary point of $r$ with respect to $b$.
We now must check the second-order condition for $k < -1$."""
    )

    # Another per-term sign-preserving change, this time for second deriv.
    def sign_preserve_2(term):
        term = term / (k**2 + 1)
        term = term * -2*(a + b*k)
        term = term.simplify()
        term = term * (a + sq)
        return term
    factor = sign_preserve_2(1)
    d2ratio = dratio.diff(b)
    assert isinstance(d2ratio, sym.Add)
    d2ratio = sym.Add(*(sign_preserve_2(arg) for arg in d2ratio.args))
    print(
f"""Evaluating $\\partial^2 r / \\partial b^2$ (not shown due to length)
and multiplying by the strictly positive factor
\\[
    {sym.latex(factor)},
\\]
we have
\\begin{{dmath*}}
    \\sign \\left( \\frac{{\\partial^2 r}}{{\\partial b^2}} \\right)
    =
    \\sign \\left( {sym.latex(d2ratio)} \\right).
\\end{{dmath*}}"""
    )

    # Evaluate the second derivative at the stationary point.
    d2r_opt = d2ratio.subs(b, soln).simplify()
    d2r_opt = d2r_opt.subs(abs(k**2 - 1), k**2 - 1).simplify()
    d2r_opt = d2r_opt.subs(sym.sqrt(k**4 + 2*k**2 + 1), k**2 + 1)
    d2r_opt = d2r_opt.simplify().factor()
    print(
f"""Evaluating at the stationary point $b_k$, this reduces to
\\begin{{dmath*}}
    \\left. \\sign \\left( \\frac{{\\partial^2 r}}{{\\partial b^2}} \\right) \\right|_{{b_k, k}}
    =
    \\sign \\left( {sym.latex(d2r_opt)} \\right).
\\end{{dmath*}}
Recalling that $k < -1$, the sign is positive."""
    )
    return


if __name__ == "__main__":
    main()
