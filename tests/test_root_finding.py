import numpy as np
import pytest

from src.numerical_methods import evapotranspiration, water_balance, bisection


# ---------------------------------------------------------------------------
# Core Functions Tests
# ---------------------------------------------------------------------------

def test_evapotranspiration_matches_formula_by_hand():
    """Validates ET calculation against standard formula for a typical day."""
    et = evapotranspiration(temperature=25, wind_speed=2, solar_index=0.7, humidity=65)
    expected = 0.12 * 25 + 0.35 * 2 + 2.4 * 0.7 - 0.025 * 65  
    assert et == pytest.approx(expected, abs=1e-9)


def test_evapotranspiration_is_floored_at_zero():
    """Ensures ET does not return negative values under extreme environmental inputs."""
    et = evapotranspiration(temperature=0, wind_speed=0, solar_index=0, humidity=1000)
    assert et == 0


def test_evapotranspiration_accepts_numpy_arrays():
    """Ensures the function is vectorized and handles NumPy array inputs."""
    temp = np.array([25.0, 30.0])
    wind = np.array([2.0, 3.0])
    solar = np.array([0.7, 0.8])
    hum = np.array([65.0, 60.0])

    result = evapotranspiration(temp, wind, solar, hum)
    expected = np.array([
        0.12 * 25 + 0.35 * 2 + 2.4 * 0.7 - 0.025 * 65,
        0.12 * 30 + 0.35 * 3 + 2.4 * 0.8 - 0.025 * 60,
    ])
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, expected)


def test_water_balance_basic_addition():
    """Validates the standard water balance equation: S_(t+1) = S_t + R_t + I_t - ET_t - D_t."""
    new_storage = water_balance(storage=30, rainfall=5, irrigation=10, et=4, drainage=2)
    assert new_storage == 39


def test_water_balance_can_go_negative():
    """
    Validates that storage can mathematically drop below zero if losses exceed inflows. 
    Note: Physical constraints (clipping at 0) are handled outside this base function.
    """
    result = water_balance(storage=10, rainfall=0, irrigation=0, et=5, drainage=8)
    assert result == -3


# ---------------------------------------------------------------------------
# Root Finding Tests
# ---------------------------------------------------------------------------

def test_bisection_finds_known_root():
    """Validates bisection method on f(x) = x^2 - 4 with a known root at x = 2."""
    root = bisection(lambda x: x**2 - 4, a=0, b=5, tol=1e-6)
    assert root == pytest.approx(2.0, abs=1e-4)


def test_bisection_solves_irrigation_for_target_moisture():
    """Uses root finding to determine required irrigation to hit a target moisture level."""
    target_moisture = 32.05

    def f(irrigation):
        return water_balance(27.4, 4.6, irrigation, 6.35, 1.2) - target_moisture

    root = bisection(f, a=0, b=20, tol=1e-6)
    assert root == pytest.approx(7.6, abs=1e-3)
    assert abs(f(root)) < 1e-3


@pytest.mark.xfail(
    reason=(
        "Midpoint exact zero evaluation flaw: f(a) * f(c) < 0 evaluates to 0 when f(c)==0, "
        "causing the algorithm to take the 'else' branch instead of returning c."
    ),
    strict=True,
)
def test_bisection_exact_root_on_midpoint():
    """Validates convergence when the true root lands exactly on a bisection midpoint."""
    def f(irrigation):
        return water_balance(28, 5, irrigation, 6.0, 1.5) - 33  

    root = bisection(f, a=0, b=20, tol=1e-6)
    assert root == pytest.approx(7.5, abs=1e-3)


def test_newton_raphson_known_root():
    try:
        from src.numerical_methods import newton_raphson
    except ImportError:
        pytest.skip("newton_raphson() is not implemented yet.")

    f = lambda x: x**2 - 4
    fprime = lambda x: 2 * x
    root = newton_raphson(f, fprime, x0=3.0, tol=1e-8)
    assert root == pytest.approx(2.0, abs=1e-6)


def test_secant_known_root():
    try:
        from src.numerical_methods import secant
    except ImportError:
        pytest.skip("secant() is not implemented yet.")

    f = lambda x: x**2 - 4
    root = secant(f, x0=1.0, x1=3.0, tol=1e-8)
    assert root == pytest.approx(2.0, abs=1e-6)


def test_root_finding_methods_agree_with_each_other():
    """Cross-validates that all root finding algorithms converge to the same value."""
    try:
        from src.numerical_methods import newton_raphson, secant
    except ImportError:
        pytest.skip("newton_raphson()/secant() not implemented yet.")

    f = lambda x: x**2 - 4
    fprime = lambda x: 2 * x

    r_bisect = bisection(f, 0, 5)
    r_newton = newton_raphson(f, fprime, x0=3.0)
    r_secant = secant(f, x0=1.0, x1=3.0)

    assert r_bisect == pytest.approx(r_newton, abs=1e-3)
    assert r_bisect == pytest.approx(r_secant, abs=1e-3)