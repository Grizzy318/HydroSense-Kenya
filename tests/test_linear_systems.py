import numpy as np
import pytest

from src.numerical_methods import solve_linear_system


# ---------------------------------------------------------------------------
# Linear Systems Tests
# ---------------------------------------------------------------------------

def test_solve_linear_system_basic_2x2():
    """Validates the basic solver against Ax = b."""
    A = np.array([[2.0, 1.0], 
                  [1.0, 3.0]])
    b = np.array([3.0, 5.0])
    
    x = solve_linear_system(A, b)
    np.testing.assert_allclose(A @ x, b, atol=1e-9)


def test_solve_linear_system_three_zone_allocation():
    """
    Domain-specific test for three-zone water allocation.
    I_A + I_B + I_C = 100   (Total pump capacity)
    I_A - I_B = 5           (Zone A needs 5 more than Zone B)
    I_B - I_C = 8           (Zone B needs 8 more than Zone C)
    """
    A = np.array([
        [1.0,  1.0,  1.0],
        [1.0, -1.0,  0.0],
        [0.0,  1.0, -1.0],
    ])
    b = np.array([100.0, 5.0, 8.0])

    x = solve_linear_system(A, b)

    np.testing.assert_allclose(A @ x, b, atol=1e-6)
    assert x.sum() == pytest.approx(100.0, abs=1e-6)


def test_solve_linear_system_raises_on_singular_matrix():
    """A singular matrix has no unique solution and should raise a LinAlgError."""
    A = np.array([[1.0, 2.0], 
                  [2.0, 4.0]]) 
    b = np.array([1.0, 2.0])
    
    with pytest.raises(np.linalg.LinAlgError):
        solve_linear_system(A, b)


def test_gaussian_elimination_matches_numpy_solution():
    """Validates manual Gaussian elimination implementation against NumPy verification."""
    try:
        from src.numerical_methods import gaussian_elimination
    except ImportError:
        pytest.skip("gaussian_elimination() is not implemented yet. Required for manual solver validation.")

    A = np.array([[2.0, 1.0], 
                  [1.0, 3.0]])
    b = np.array([3.0, 5.0])

    manual = gaussian_elimination(A, b)
    reference = solve_linear_system(A, b) 

    np.testing.assert_allclose(manual, reference, atol=1e-6)