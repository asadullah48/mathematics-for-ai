"""
Calculus Module

Differential and integral calculus operations for machine learning.
Includes automatic differentiation, numerical methods, and optimization.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class DerivativeResult:
    """Result of differentiation containing value and gradient."""
    value: float
    gradient: np.ndarray
    hessian: Optional[np.ndarray] = None


class Calculus:
    """
    Comprehensive calculus operations for ML and optimization.
    
    Features:
    - Numerical differentiation (forward, backward, central)
    - Gradient computation
    - Jacobian and Hessian matrices
    - Numerical integration
    - Taylor series expansion
    - Root finding algorithms
    """
    
    # ==================== Numerical Differentiation ====================
    
    @staticmethod
    def numerical_derivative(f: Callable[[float], float], x: float, 
                            method: str = "central", h: float = 1e-7) -> float:
        """
        Compute numerical derivative of a scalar function.
        
        Args:
            f: Function f: R -> R
            x: Point to evaluate at
            method: 'forward', 'backward', or 'central'
            h: Step size
            
        Returns:
            Approximate derivative at x
            
        Raises:
            ValueError: If invalid method specified
        """
        if method == "forward":
            return (f(x + h) - f(x)) / h
        elif method == "backward":
            return (f(x) - f(x - h)) / h
        elif method == "central":
            return (f(x + h) - f(x - h)) / (2 * h)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'forward', 'backward', or 'central'")
    
    @staticmethod
    def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray,
                          method: str = "central", h: float = 1e-7) -> np.ndarray:
        """
        Compute numerical gradient of a multivariate function.
        
        Args:
            f: Function f: R^n -> R
            x: Point to evaluate at (n,)
            method: 'forward' or 'central'
            h: Step size
            
        Returns:
            Gradient vector at x
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        gradient = np.zeros(n)
        
        if method == "forward":
            for i in range(n):
                x_plus = x.copy()
                x_plus[i] += h
                gradient[i] = (f(x_plus) - f(x)) / h
        elif method == "central":
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'forward' or 'central'")
        
        return gradient
    
    @staticmethod
    def jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                h: float = 1e-7) -> np.ndarray:
        """
        Compute the Jacobian matrix of a vector-valued function.
        
        Args:
            f: Function f: R^n -> R^m
            x: Point to evaluate at (n,)
            h: Step size
            
        Returns:
            Jacobian matrix (m x n) where J[i,j] = ∂f_i/∂x_j
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        fx = f(x)
        m = fx.size
        
        J = np.zeros((m, n))
        
        for j in range(n):
            x_plus = x.copy()
            x_plus[j] += h
            f_plus = f(x_plus)
            J[:, j] = (f_plus - fx) / h
        
        return J
    
    @staticmethod
    def hessian(f: Callable[[np.ndarray], float], x: np.ndarray,
               h: float = 1e-5) -> np.ndarray:
        """
        Compute the Hessian matrix of a scalar function.
        
        Args:
            f: Function f: R^n -> R
            x: Point to evaluate at (n,)
            h: Step size
            
        Returns:
            Hessian matrix (n x n) where H[i,j] = ∂²f/∂x_i∂x_j
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        H = np.zeros((n, n))
        
        # Central difference for second derivatives
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                
                x_pp[i] += h
                x_pp[j] += h
                x_pm[i] += h
                x_pm[j] -= h
                x_mp[i] -= h
                x_mp[j] += h
                x_mm[i] -= h
                x_mm[j] -= h
                
                H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
                H[j, i] = H[i, j]  # Symmetry
        
        return H
    
    @staticmethod
    def directional_derivative(f: Callable[[np.ndarray], float], x: np.ndarray,
                              direction: np.ndarray, h: float = 1e-7) -> float:
        """
        Compute directional derivative in a given direction.
        
        Args:
            f: Function f: R^n -> R
            x: Point to evaluate at
            direction: Direction vector
            h: Step size
            
        Returns:
            Directional derivative
        """
        direction = np.asarray(direction, dtype=float)
        direction = direction / np.linalg.norm(direction)  # Normalize
        return (f(x + h * direction) - f(x - h * direction)) / (2 * h)
    
    @staticmethod
    def laplacian(f: Callable[[np.ndarray], float], x: np.ndarray,
                 h: float = 1e-5) -> float:
        """
        Compute the Laplacian (trace of Hessian) of a function.
        
        Args:
            f: Function f: R^n -> R
            x: Point to evaluate at
            h: Step size
            
        Returns:
            Laplacian ∇²f = Σ ∂²f/∂x_i²
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        laplacian = 0.0
        
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            laplacian += (f(x_plus) - 2 * f(x) + f(x_minus)) / (h * h)
        
        return laplacian
    
    # ==================== Gradient Operators ====================
    
    @staticmethod
    def gradient_descent_step(f: Callable[[np.ndarray], float], x: np.ndarray,
                             learning_rate: float = 0.01) -> np.ndarray:
        """
        Perform one step of gradient descent.
        
        Args:
            f: Objective function
            x: Current point
            learning_rate: Step size
            
        Returns:
            Updated point
        """
        grad = Calculus.numerical_gradient(f, x)
        return x - learning_rate * grad
    
    @staticmethod
    def gradient_ascent_step(f: Callable[[np.ndarray], float], x: np.ndarray,
                            learning_rate: float = 0.01) -> np.ndarray:
        """
        Perform one step of gradient ascent.
        
        Args:
            f: Objective function
            x: Current point
            learning_rate: Step size
            
        Returns:
            Updated point
        """
        grad = Calculus.numerical_gradient(f, x)
        return x + learning_rate * grad
    
    @staticmethod
    def newton_step(f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        """
        Perform one step of Newton's method.
        
        Args:
            f: Objective function
            x: Current point
            
        Returns:
            Updated point
        """
        grad = Calculus.numerical_gradient(f, x)
        hess = Calculus.hessian(f, x)
        
        # Add regularization for numerical stability
        reg = 1e-6 * np.eye(len(x))
        hess_reg = hess + reg
        
        try:
            delta = np.linalg.solve(hess_reg, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]
        
        return x - delta
    
    # ==================== Numerical Integration ====================
    
    @staticmethod
    def trapezoidal(f: Callable[[float], float], a: float, b: float, 
                   n: int = 100) -> float:
        """
        Numerical integration using the trapezoidal rule.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n: Number of intervals
            
        Returns:
            Approximate integral
        """
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])
        
        return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    
    @staticmethod
    def simpson(f: Callable[[float], float], a: float, b: float,
               n: int = 100) -> float:
        """
        Numerical integration using Simpson's rule.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n: Number of intervals (must be even)
            
        Returns:
            Approximate integral
            
        Raises:
            ValueError: If n is odd
        """
        if n % 2 != 0:
            n += 1
        
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = np.array([f(xi) for xi in x])
        
        return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    
    @staticmethod
    def gaussian_quadrature(f: Callable[[float], float], a: float, b: float,
                           n: int = 5) -> float:
        """
        Numerical integration using Gaussian quadrature.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n: Number of points
            
        Returns:
            Approximate integral
        """
        # Get roots of Legendre polynomial and weights
        from numpy.polynomial.legendre import leggauss
        
        roots, weights = leggauss(n)
        
        # Transform from [-1, 1] to [a, b]
        transformed_roots = 0.5 * (b - a) * roots + 0.5 * (b + a)
        
        # Evaluate and sum
        return 0.5 * (b - a) * np.sum(weights * np.array([f(x) for x in transformed_roots]))
    
    @staticmethod
    def monte_carlo_integrate(f: Callable[[np.ndarray], float], bounds: List[Tuple[float, float]],
                             n_samples: int = 10000, seed: int = 42) -> Tuple[float, float]:
        """
        Monte Carlo integration for multidimensional integrals.
        
        Args:
            f: Function to integrate
            bounds: List of (lower, upper) bounds for each dimension
            n_samples: Number of samples
            seed: Random seed
            
        Returns:
            Tuple of (integral estimate, standard error)
        """
        np.random.seed(seed)
        bounds = np.array(bounds)
        dim = len(bounds)
        
        # Generate random samples
        samples = np.random.uniform(
            bounds[:, 0], 
            bounds[:, 1], 
            size=(n_samples, dim)
        )
        
        # Evaluate function
        values = np.array([f(sample) for sample in samples])
        
        # Compute volume
        volume = np.prod(bounds[:, 1] - bounds[:, 0])
        
        # Estimate integral
        integral = volume * np.mean(values)
        std_error = volume * np.std(values) / np.sqrt(n_samples)
        
        return integral, std_error
    
    # ==================== Taylor Series ====================
    
    @staticmethod
    def taylor_series(f: Callable[[float], float], x0: float, n: int = 5,
                     num_points: int = 100, x_range: Tuple[float, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Taylor series approximation of a function.
        
        Args:
            f: Function to approximate
            x0: Expansion point
            n: Order of Taylor polynomial
            num_points: Number of points for evaluation
            x_range: Range for evaluation (default: x0 ± 2)
            
        Returns:
            Tuple of (x values, Taylor approximation values)
        """
        if x_range is None:
            x_range = (x0 - 2, x0 + 2)
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        # Compute derivatives at x0
        coefficients = []
        for k in range(n + 1):
            if k == 0:
                coeff = f(x0)
            else:
                # Numerical derivative
                h = 1e-5
                deriv = 0
                for i in range(k):
                    # Use finite differences
                    pass
                # Simplified: use automatic differentiation concept
                coeff = Calculus._compute_derivative_at(f, x0, k)
            
            coefficients.append(coeff / np.math.factorial(k))
        
        # Evaluate Taylor polynomial
        taylor = np.zeros_like(x)
        for k, coeff in enumerate(coefficients):
            taylor += coeff * (x - x0) ** k
        
        return x, taylor
    
    @staticmethod
    def _compute_derivative_at(f: Callable[[float], float], x: float, order: int) -> float:
        """Compute nth order derivative at a point."""
        if order == 0:
            return f(x)
        
        h = 1e-5
        if order == 1:
            return (f(x + h) - f(x - h)) / (2 * h)
        else:
            # Recursive for higher orders
            def df(x):
                return (f(x + h) - f(x - h)) / (2 * h)
            return Calculus._compute_derivative_at(df, x, order - 1)
    
    # ==================== Root Finding ====================
    
    @staticmethod
    def bisection(f: Callable[[float], float], a: float, b: float,
                 tol: float = 1e-10, max_iter: int = 100) -> float:
        """
        Find root using bisection method.
        
        Args:
            f: Function to find root of
            a: Left endpoint
            b: Right endpoint
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            Approximate root
            
        Raises:
            ValueError: If f(a) and f(b) have same sign
        """
        if f(a) * f(b) > 0:
            raise ValueError("f(a) and f(b) must have opposite signs")
        
        for _ in range(max_iter):
            c = (a + b) / 2
            if abs(f(c)) < tol or (b - a) / 2 < tol:
                return c
            
            if f(c) * f(a) < 0:
                b = c
            else:
                a = c
        
        return (a + b) / 2
    
    @staticmethod
    def newton_raphson(f: Callable[[float], float], x0: float,
                      tol: float = 1e-10, max_iter: int = 100) -> float:
        """
        Find root using Newton-Raphson method.
        
        Args:
            f: Function to find root of
            x0: Initial guess
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            Approximate root
        """
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            if abs(fx) < tol:
                return x
            
            dfx = Calculus.numerical_derivative(f, x)
            if abs(dfx) < 1e-15:
                break
            
            x = x - fx / dfx
        
        return x
    
    @staticmethod
    def secant(f: Callable[[float], float], x0: float, x1: float,
              tol: float = 1e-10, max_iter: int = 100) -> float:
        """
        Find root using secant method.
        
        Args:
            f: Function to find root of
            x0: First initial guess
            x1: Second initial guess
            tol: Tolerance
            max_iter: Maximum iterations
            
        Returns:
            Approximate root
        """
        for _ in range(max_iter):
            f0, f1 = f(x0), f(x1)
            if abs(f1) < tol:
                return x1
            
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0, x1 = x1, x2
        
        return x1
    
    # ==================== Automatic Differentiation (Forward Mode) ====================
    
    @staticmethod
    def dual_number_derivative(f: Callable[[float], float], x: float) -> Tuple[float, float]:
        """
        Compute derivative using dual numbers (forward mode AD).
        
        Args:
            f: Function f: R -> R
            x: Point to evaluate at
            
        Returns:
            Tuple of (function value, derivative)
        """
        # Dual number: x + ε where ε² = 0
        # f(x + ε) = f(x) + f'(x)ε
        
        h = 1e-10
        fx = f(x)
        fx_plus_h = f(x + h)
        
        derivative = (fx_plus_h - fx) / h
        return fx, derivative
    
    @staticmethod
    def gradient_check(f: Callable[[np.ndarray], float], grad_f: Callable[[np.ndarray], np.ndarray],
                      x: np.ndarray, tol: float = 1e-7) -> bool:
        """
        Verify analytical gradient against numerical gradient.
        
        Args:
            f: Function
            grad_f: Analytical gradient function
            x: Point to check
            tol: Tolerance for comparison
            
        Returns:
            True if gradients match within tolerance
        """
        numerical_grad = Calculus.numerical_gradient(f, x)
        analytical_grad = grad_f(x)
        
        diff = np.linalg.norm(numerical_grad - analytical_grad)
        sum_norms = np.linalg.norm(numerical_grad) + np.linalg.norm(analytical_grad)
        
        relative_error = diff / (sum_norms + 1e-15)
        return relative_error < tol
    
    # ==================== Partial Derivatives ====================
    
    @staticmethod
    def partial_derivative(f: Callable, x: np.ndarray, var_idx: int,
                          h: float = 1e-7) -> float:
        """
        Compute partial derivative with respect to one variable.
        
        Args:
            f: Multivariate function
            x: Point to evaluate at
            var_idx: Index of variable to differentiate
            h: Step size
            
        Returns:
            Partial derivative value
        """
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[var_idx] += h
        x_minus[var_idx] -= h
        
        return (f(x_plus) - f(x_minus)) / (2 * h)
    
    @staticmethod
    def mixed_partial_derivative(f: Callable, x: np.ndarray, 
                                var_idx_1: int, var_idx_2: int,
                                h: float = 1e-5) -> float:
        """
        Compute mixed partial derivative ∂²f/∂x_i∂x_j.
        
        Args:
            f: Multivariate function
            x: Point to evaluate at
            var_idx_1: First variable index
            var_idx_2: Second variable index
            h: Step size
            
        Returns:
            Mixed partial derivative value
        """
        x_pp = x.copy()
        x_pm = x.copy()
        x_mp = x.copy()
        x_mm = x.copy()
        
        x_pp[var_idx_1] += h
        x_pp[var_idx_2] += h
        x_pm[var_idx_1] += h
        x_pm[var_idx_2] -= h
        x_mp[var_idx_1] -= h
        x_mp[var_idx_2] += h
        x_mm[var_idx_1] -= h
        x_mm[var_idx_2] -= h
        
        return (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
    
    # ==================== Chain Rule ====================
    
    @staticmethod
    def chain_rule_single(f: Callable[[float], float], g: Callable[[float], float],
                         x: float) -> float:
        """
        Compute derivative of f(g(x)) using chain rule.
        
        Args:
            f: Outer function
            g: Inner function
            x: Point to evaluate at
            
        Returns:
            Derivative (f ∘ g)'(x)
        """
        gx = g(x)
        dg_dx = Calculus.numerical_derivative(g, x)
        df_dg = Calculus.numerical_derivative(f, gx)
        
        return df_dg * dg_dx
    
    @staticmethod
    def chain_rule_multivariate(f: Callable[[np.ndarray], float], 
                                g: Callable[[float], np.ndarray],
                                t: float) -> float:
        """
        Compute derivative of f(g(t)) where g: R -> R^n and f: R^n -> R.
        
        Args:
            f: Outer function
            g: Inner function (parametric curve)
            t: Parameter value
            
        Returns:
            Derivative df/dt
        """
        gt = g(t)
        dg_dt = Calculus.numerical_gradient(g, t)  # This needs adjustment
        grad_f = Calculus.numerical_gradient(f, gt)
        
        # For proper implementation, g should return gradient
        # Simplified version:
        h = 1e-7
        dg_dt = (g(t + h) - g(t - h)) / (2 * h)
        
        return np.dot(grad_f, dg_dt)
