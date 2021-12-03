import sympy

def ODE_parser(ODE):
    """
    parse ODE string
    { x' = f(t, x), y' = g(t, x, y) & t <= T}
    { x' = y, y' = x & t <= 10}
    implicit t' = 1

    output
    successful: bool
    z: collection of sympy symbols
    f: function z' = f(t, z)
    T: maximum time > 0
    """
    ODE = ODE.replace(" ", "")
    if ODE[0] != "{":
        return False, None, None, None
    if ODE[-1] != "}":
        return False, None, None, None
    if "&" not in ODE:
        return False, None, None, None
    a = ODE.index("&")
    ode_part = ODE[1:a]
    ode_part_split = ode_part.split(",")
    ode_vars = []
    ode_expr = []
    for ode_p in ode_part_split:
        var_idx = ode_p.index("'=")
        ode_vars.append(sympy.symbols(ode_p[:var_idx]))
        ode_expr.append(sympy.sympify(ode_p[var_idx + len("'="):]))

    # z' = f(t, z) where z is a vector
    ode_func = sympy.lambdify([sympy.symbols("t"), ode_vars], sympy.Matrix(ode_expr))
    
    time_part = ODE[a+1:-1]
    T_idx = time_part.replace(" ", "").index("t<=") + len("t<=")
    T = sympy.sympify(time_part[T_idx:])
    return True, ode_vars, ode_func, T

