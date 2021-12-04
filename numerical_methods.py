"""
Numerical Methods

numerical_method(f, x0, T, h)
    calculates discrete approximation to the differential equation 
    x'(t) = f(t, x) for 0 <= t <= T and initial value x0 using timestep h

    input
    f: function f representing differential equation x'(t) = f(t, x)
    x0: initial value of x
    T: maximum time
    h: timestep

    output
    ts: discrete time values where we generated approximations
    xs: approximations at the discrete time values

numerical_method_h(M, T, eps)
    calculates timestep necessary to obtain error of at most eps

    input
    M: maximum magnitude of second derivative of the function over the time range
    L: maximum magnitude of first derivative of the function over the time range
    T: maximum time
    eps: desired error bound

    output
    h: timestep
"""
import numpy as np

def calculate_global_error(tau, C, T, h):
    """
    input
    tau: local truncation error as a function of h
    C: one-step stability constant as a function of h
    T: maximum timestep
    h: timestep

    output
    error: global error bound
    """
    tau_h = tau(h)
    C_h = C(h)
    error = tau_h*(C_h**(T/h)-1)/(C_h-1)
    return error

def calculate_h_lte_C(tau, C, T, eps, max_ite=10):
    """
    input
    tau: local truncation error as a function of h
    C: one-step stability constant as a function of h
    eps: desired error tolerance

    output
    h: timestep to generate the desired error tolerance

    solves via bisection method
    """
    h = T 
    error = eps*2
    # first find an h which has error < eps
    while error >= eps:
        h = h/2
        error = calculate_global_error(tau, C, T, h)
    # then try to find a larger h
    lower_h = h
    upper_h = 2*h 
    for i in range(max_ite):
        h = (lower_h + upper_h)/2
        error = calculate_global_error(tau, C, T, h)

        if error < eps:
            lower_h = h 
        else:
            upper_h = h
    return h

def calculate_h(M, L, T, eps, num_method):
    calc_h_func = None
    if num_method == forward_euler:
        calc_h_func = forward_euler_h
    
    if calc_h_func is None:
        return 0.1
    else:
        return calc_h_func(M, L, T, eps)


def ODE_onestep(f, x0, T, h, onestep):
    N = int(np.ceil(T/h)) 

    xs = [x0]
    ts = [0]

    for i in range(N-1):
        xs.append(onestep(f, ts[i], xs[i], h))
        ts.append(ts[i] + h)

    ts.append(T)
    h_last = ts[-1] - ts[-2]
    xs.append(xs[-1] + h_last*f(ts[-1], xs[-1]))
    return ts, xs

def forward_euler(f, x0, T, h):
    def forward_euler_onestep(f, ti, xi, h):
        return xi + h*f(ti, xi)

    return ODE_onestep(f, x0, T, h, forward_euler_onestep)

def forward_euler_h(M, L, T, eps):
    """
    error <= hM/(2L)*(exp(L(t-t0)-1) =: eps
    h = eps*2L/(M*(exp(L(t-t0)-1))
    """
    return eps*2*L/(M*np.exp(L*T)-1)

def forward_euler_C(L):
    """
    L : lipschitz constant for f in x'(t)=f(t,x(t))
    """
    return lambda h: 1+h*L

def forward_euler_tau(M):
    """
    M : maximum of abs(derivative of f/second derivative of x)
    LTE_i = (h^2)/2 * y^(2)(xi) 
    """
    return lambda h: (1/2)*(h**2)*M

def midpoint_method(f, x0, T, h):
    def midpoint_onestep(f, ti, xi, h):
        K1 = f(ti, xi)
        K2 = f(ti + h/2, xi + h/2 * K1)
        return xi + h*K2
    
    return ODE_onestep(f, x0, T, h, midpoint_onestep)

def midpoint_tau(F, L, M):
    """
    F : maximum of abs(f) for x'(t) = f(t, x(t))
    L : maximum{abs(f_t), abs(f_y)} (first partial derivatives)
    M : maximum{abs(f_tt), abs(f_ty), abs(f_yy)} (second partial derivatives)

    LTE_i = (h^3)/24*f_tt + (h^3)/12*f*f_ty + (h^3)/24*f^2*f_yy + (h^3)/6*(f_t+f*f_y)*f_y
    """
    return lambda h: (h**3)*M/24 + (h**3)*F*M/12 + (h**3)*F**2*M/24 + (h**3)*(L + F*L)*L

def modified_euler(f, x0, T, h):
    def modified_euler_onestep(f, ti, xi, h):
        K1 = f(ti, xi)
        K2 = f(ti + h, xi + h * K1)
        return xi + (h/2)*(K1+K2)
    
    return ODE_onestep(f, x0, T, h, modified_euler)

def modified_tau(F, L, M):
    """
    F : maximum of abs(f) for x'(t) = f(t, x(t))
    L : maximum{abs(f_t), abs(f_y)} (first partial derivatives)
    M : maximum{abs(f_tt), abs(f_ty), abs(f_yy)} (second partial derivatives)

    LTE_i = (h^3)/12*f_tt + (h^3)/6*f*f_ty + (h^3)/12*f^2*f_yy + (h^3)/6*(f_t+f*f_y)*f_y
    """
    return lambda h: (h**3)*M/12 + (h**3)*F*M/6 + (h**3)*F**2*M/12 + (h**3)*(L + F*L)*L


def modified_and_midpoint_method_C(L):
    """
    L: lipschitz constant for f in x'(t)=f(t,x(t))
    """
    return lambda h: 1+h*L+(1/2)*(h*L)**2
