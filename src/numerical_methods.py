import numpy as np

# ---------------------------------------------------------
# Level 1: Core Mathematical Models
# ---------------------------------------------------------

def evapotranspiration(temperature, wind_speed, solar_index, humidity):
    return np.maximum(
        0,
        0.12 * temperature + 0.35 * wind_speed + 2.4 * solar_index - 0.025 * humidity
    )

def water_balance(storage, rainfall, irrigation, et, drainage):
    return storage + rainfall + irrigation - et - drainage


# ---------------------------------------------------------
# Level 3: Root Finding Methods
# ---------------------------------------------------------

def bisection(f, a, b, tol=1e-6):
    """Finds a root of f(x) within the interval [a, b] using the bisection method."""
    while abs(b - a) > tol:
        c = (a + b) / 2
        
        # FIX: Check if the midpoint is exactly the root to prevent the zero-product bug
        if f(c) == 0:
            return c
            
        if np.sign(f(a)) != np.sign(f(c)):
            b = c
        else:
            a = c
    return (a + b) / 2

def newton_raphson(f, fprime, x0, tol=1e-6, max_iter=100):
    """Finds a root of f(x) given its derivative fprime(x) and an initial guess x0."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        
        dfx = fprime(x)
        if dfx == 0:
            raise ValueError("Derivative is zero. Newton-Raphson fails.")
            
        x = x - fx / dfx
    return x

def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """Finds a root of f(x) using two initial guesses x0 and x1."""
    for _ in range(max_iter):
        f0 = f(x0)
        f1 = f(x1)
        
        if abs(f1) < tol:
            return x1
            
        if f1 - f0 == 0:
            raise ValueError("Denominator is zero. Secant method fails.")
            
        x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, x1 = x1, x_new
        
    return x1


# ---------------------------------------------------------
# Level 3: Differentiation and Integration
# ---------------------------------------------------------

def derivative(f, x, h=1e-5):
    """Central difference derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)

def forward_difference(f, x, h=1e-5):
    """Forward difference derivative."""
    return (f(x + h) - f(x)) / h

def backward_difference(f, x, h=1e-5):
    """Backward difference derivative."""
    return (f(x) - f(x - h)) / h

def trapezoidal(x, y):
    """Integrates using the trapezoidal rule, safe for NumPy 1.x and 2.x."""
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)

def simpson(x, y):
    """Computes the integral of y over x using Simpson's rule."""
    from scipy.integrate import simpson as scipy_simpson
    
    # Handle API variations across Scipy versions
    if hasattr(scipy_simpson, "__call__"):
        return scipy_simpson(y, x=x)
    else:
        from scipy.integrate import simps
        return simps(y, x=x)


# ---------------------------------------------------------
# Level 3: Linear Systems
# ---------------------------------------------------------

def solve_linear_system(A, b):
    """Verification solver using NumPy's built-in optimized algorithms."""
    return np.linalg.solve(A, b)

def gaussian_elimination(A, b):
    """
    Solves the linear system Ax = b using manual Gaussian elimination 
    without partial pivoting. Required by the Level 3 brief.
    """
    # Convert to float to avoid integer division issues
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    
    # Forward elimination
    for i in range(n):
        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
            
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        
    return x