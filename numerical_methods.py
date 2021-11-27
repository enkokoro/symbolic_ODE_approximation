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
    xs.append(xs[-2] + h_last*f(ts[-2], xs[-2]))
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

def midpoint_method(f, x0, T, h):
    def midpoint_onestep(f, ti, xi, h):
        K1 = f(ti, xi)
        K2 = f(ti + h/2, xi + h/2 * K1)
        return xi + h*K2
    
    return ODE_onestep(f, x0, T, h, midpoint_onestep)
