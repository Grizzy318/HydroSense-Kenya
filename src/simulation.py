import numpy as np

def moisture_change(rainfall, irrigation, et, drainage):
    return rainfall + irrigation - et - drainage

def euler_method(func, y0, t0, tf, h):
    t = np.arange(t0, tf + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i - 1] + h * func(t[i - 1], y[i - 1])
    return t, y

def rk4(func, y0, t0, tf, h):
    t = np.arange(t0, tf + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = func(t[i - 1], y[i - 1])
        k2 = func(t[i - 1] + h / 2, y[i - 1] + h * k1 / 2)
        k3 = func(t[i - 1] + h / 2, y[i - 1] + h * k2 / 2)
        k4 = func(t[i - 1] + h, y[i - 1] + h * k3)
        y[i] = y[i - 1] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return t, y
