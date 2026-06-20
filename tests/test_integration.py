import numpy as np
import pytest

from src.numerical_methods import derivative, trapezoidal


# ---------------------------------------------------------------------------
# Differentiation Tests
# ---------------------------------------------------------------------------

def test_central_difference_on_polynomial():
    """d/dx(x^2) at x=3 should be 6. Central difference is exact for quadratics."""
    result = derivative(lambda x: x**2, x=3, h=1e-5)
    assert result == pytest.approx(6.0, abs=1e-6)


def test_central_difference_on_sine():
    """d/dx(sin(x)) at x=0 should be cos(0) = 1."""
    result = derivative(np.sin, x=0, h=1e-5)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_central_difference_step_size_sensitivity():
    """Central difference error is O(h^2); shrinking h should maintain or improve accuracy."""
    f = lambda x: np.sin(x)
    true_value = np.cos(1.0)

    err_coarse = abs(derivative(f, x=1.0, h=1e-2) - true_value)
    err_fine = abs(derivative(f, x=1.0, h=1e-5) - true_value)

    assert err_fine <= err_coarse


def test_forward_difference_known_derivative():
    try:
        from src.numerical_methods import forward_difference
    except ImportError:
        pytest.skip("forward_difference() is not implemented yet.")

    result = forward_difference(lambda x: x**2, x=3, h=1e-5)
    assert result == pytest.approx(6.0, abs=1e-2)


def test_backward_difference_known_derivative():
    try:
        from src.numerical_methods import backward_difference
    except ImportError:
        pytest.skip("backward_difference() is not implemented yet.")

    result = backward_difference(lambda x: x**2, x=3, h=1e-5)
    assert result == pytest.approx(6.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

def test_trapezoidal_runs_without_numpy_version_error():
    """Validates compatibility with NumPy API changes (np.trapz vs np.trapezoid)."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    try:
        trapezoidal(x, y)
    except AttributeError as e:
        pytest.fail(f"trapezoidal() raised AttributeError ({e}). Update to use np.trapezoid.")


def test_trapezoidal_integrates_quadratic_accurately():
    """Integral of x^2 from 0 to 1 is 1/3."""
    x = np.linspace(0, 1, 1001)
    y = x**2
    result = trapezoidal(x, y)
    assert result == pytest.approx(1 / 3, abs=1e-4)


def test_trapezoidal_exact_for_constant_function():
    """Trapezoidal rule is exact for constant (degree-0) functions."""
    x = np.linspace(0, 10, 5)
    y = np.full_like(x, 5.0)
    result = trapezoidal(x, y)
    assert result == pytest.approx(50.0, abs=1e-9)


def test_trapezoidal_cumulative_rainfall_matches_manual_sum():
    """Integration over discrete sample data."""
    days = np.array([0, 1, 2, 3])
    rainfall = np.array([3.2, 2.2, 3.0, 1.6]) 
    result = trapezoidal(days, rainfall)
    assert result == pytest.approx(7.6, abs=1e-6)


def test_simpson_rule_known_integral():
    try:
        from src.numerical_methods import simpson
    except ImportError:
        pytest.skip("simpson() is not implemented yet.")

    x = np.linspace(0, 1, 101)
    y = x**2
    result = simpson(x, y)
    assert result == pytest.approx(1 / 3, abs=1e-6)