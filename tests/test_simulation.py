import numpy as np
import pytest

from src.simulation import moisture_change, euler_method, rk4
from src.optimization import objective, optimize_water_allocation


# ---------------------------------------------------------------------------
# Simulation & Differential Equations Tests
# ---------------------------------------------------------------------------

def test_moisture_change_basic():
    """Validates basic mass balance addition/subtraction for moisture change."""
    result = moisture_change(rainfall=5, irrigation=3, et=4, drainage=1)
    assert result == 3


def test_euler_method_output_shape_and_initial_value():
    """Ensures the Euler method returns arrays of expected length and initial conditions."""
    func = lambda t, y: -0.1 * y
    t, y = euler_method(func, y0=40.0, t0=0, tf=10, h=0.5)
    assert len(t) == len(y)
    assert y[0] == 40.0


def test_euler_method_approximates_exponential_decay():
    """Validates Euler method against the analytic solution for dy/dt = -k*y: y(t) = y0 * exp(-k*t)."""
    k, y0, t0, tf, h = 0.15, 40.0, 0, 20, 0.1
    func = lambda t, y: -k * y

    _, y = euler_method(func, y0, t0, tf, h)
    analytic_final = y0 * np.exp(-k * tf)

    assert y[-1] == pytest.approx(analytic_final, abs=0.1)


def test_rk4_approximates_exponential_decay_closely():
    """Validates RK4 against the analytic solution for dy/dt = -k*y with high precision."""
    k, y0, t0, tf, h = 0.15, 40.0, 0, 20, 0.1
    func = lambda t, y: -k * y

    _, y = rk4(func, y0, t0, tf, h)
    analytic_final = y0 * np.exp(-k * tf)

    assert y[-1] == pytest.approx(analytic_final, abs=1e-4)


def test_rk4_is_more_accurate_than_euler_for_coarse_steps():
    """
    Validates stability and error scaling: for a coarse step size, RK4 (4th-order) 
    tracks the analytic solution much more closely than Euler (1st-order).
    """
    k, y0, t0, tf, h = 0.15, 40.0, 0, 20, 1.0 
    func = lambda t, y: -k * y
    analytic_final = y0 * np.exp(-k * tf)

    _, y_euler = euler_method(func, y0, t0, tf, h)
    _, y_rk4 = rk4(func, y0, t0, tf, h)

    error_euler = abs(y_euler[-1] - analytic_final)
    error_rk4 = abs(y_rk4[-1] - analytic_final)

    assert error_rk4 < error_euler


# ---------------------------------------------------------------------------
# Optimization Tests
# ---------------------------------------------------------------------------

def test_objective_is_zero_at_target():
    """The objective function cost should be exactly 0 when inputs match target values perfectly."""
    assert objective(np.array([35, 40, 25])) == 0


def test_objective_is_positive_away_from_target():
    """The objective function cost should be > 0 for any non-target inputs."""
    assert objective(np.array([0, 0, 0])) > 0


def test_optimize_water_allocation_respects_total_water_constraint():
    """Ensures the optimizer respects the equality constraint for total water available."""
    result = optimize_water_allocation(total_water=100)
    assert np.sum(result) == pytest.approx(100.0, abs=1e-4)


def test_optimize_water_allocation_matches_target_when_water_is_sufficient():
    """When total water equals target demand, the optimizer should allocate exact target amounts."""
    result = optimize_water_allocation(total_water=100)
    np.testing.assert_allclose(result, [35, 40, 25], atol=1e-3)


def test_optimize_water_allocation_stays_within_bounds():
    """Ensures the optimizer respects the boundary constraints [0, total_water] for all zones."""
    result = optimize_water_allocation(total_water=60)
    assert np.all(result >= 0) and np.all(result <= 60)


def test_optimize_water_allocation_shares_shortfall_proportionally_by_deficit():
    """
    Validates allocation under scarcity. With total water < target sum, the optimum 
    shifts every zone's allocation by an equal deficit amount (x_i = target_i - lambda).
    """
    result = optimize_water_allocation(total_water=60)
    target = np.array([35.0, 40.0, 25.0])
    lam = (target.sum() - 60) / 3
    expected = target - lam
    np.testing.assert_allclose(result, expected, atol=1e-2)