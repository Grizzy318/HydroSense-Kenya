import numpy as np
from scipy.optimize import minimize

def objective(x):
    target = np.array([35, 40, 25])
    return np.sum((x - target) ** 2)

def optimize_water_allocation(total_water=100):
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - total_water}
    bounds = [(0, total_water)] * 3
    result = minimize(
        objective,
        [30, 35, 35],
        bounds=bounds,
        constraints=constraints
    )
    return result.x
