import numpy as np

def evapotranspiration(temperature, wind_speed, solar_index, humidity):
    return np.maximum(
        0,
        0.12 * temperature + 0.35 * wind_speed + 2.4 * solar_index - 0.025 * humidity
    )

def water_balance(storage, rainfall, irrigation, et, drainage):
    return storage + rainfall + irrigation - et - drainage

def bisection(f, a, b, tol=1e-6):
    while abs(b - a) > tol:
        c = (a + b) / 2
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def trapezoidal(x, y):
    return np.trapz(y, x)

def solve_linear_system(A, b):
    return np.linalg.solve(A, b)
