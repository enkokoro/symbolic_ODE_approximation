import sympy
from numerical_methods import calculate_h, forward_euler

def ODE_approx(f, dim, T, h=None, num_method=forward_euler, spline_deg=1):
    """ 
    symbolic function approximation to solution of
    x'(t) = f(t, x) for 0 <= t <= T

    inputs
    f: RHS function of vector diff eq
    dim: dimension of vector
    T: maximum time
    eps: global error
    num_method: numerical method used for calculating discrete ODE approximation 
    approx_method: method used for converting discrete approximation to continuous approximation

    output
    x_approx(x0, t): function which approximates solution to x'(t) = f(t, x) within eps
    """
    _x0 = sympy.Matrix([sympy.symbols("_x"+str(i)) for i in range(1, dim+1)])
    ts, xs = num_method(f, _x0, T, h)

    xis = [[] for i in range(dim)]
    for x in xs:
        for i in range(dim):
            xis[i].append(x[i])

    _t = sympy.symbols("_t")

    f_approx = [sympy.interpolating_spline(spline_deg, _t, ts, xis[i]) for i in range(dim)]

    return sympy.lambdify([_x0, _t], f_approx)
