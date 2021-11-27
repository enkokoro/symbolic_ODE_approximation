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
    M: maximum derivative of the function over the time range
    T: maximum time
    eps: desired error bound

    output
    h: timestep
"""
import numpy as np

def calculate_h(M, T, eps, num_method):
    calc_h_func = None
    # if num_method == forward_euler:
    #     calc_h_func = forward_euler_h
    
    if calc_h_func is None:
        return 0.1
    else:
        return calc_h_func(M, T, eps)


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

def forward_euler_h(M, T, eps):
    # TODO
    pass

def runge_kutta_4(f, x0, T, h):
    def rk4_onestep(f, ti, xi, h):
        # TODO
        pass

def runge_kutta_4_h(M, T, eps):
    # TODO
    pass 