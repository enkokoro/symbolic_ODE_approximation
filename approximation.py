import sympy
from numerical_methods import calculate_h, forward_euler

def ODE_approx(f, T, eps=1e-5, num_method=forward_euler, spline_deg=1):
    """ 
    symbolic function approximation to solution of
    x'(t) = f(t, x) for 0 <= t <= T

    inputs
    f: RHS function of diff eq
    T: maximum time
    eps: global error
    num_method: numerical method used for calculating discrete ODE approximation 
    approx_method: method used for converting discrete approximation to continuous approximation

    output
    x_approx(x0, t): function which approximates solution to x'(t) = f(t, x) within eps
    """
    M = 1
    h = calculate_h(M, T, eps, num_method)
    _x0 = sympy.symbols("_x0")
    ts, xs = num_method(f, _x0, T, h)

    _t = sympy.symbols("_t")
    f_approx = sympy.interpolating_spline(spline_deg, _t, ts, xs)

    return sympy.lambdify([_x0, _t], f_approx)
