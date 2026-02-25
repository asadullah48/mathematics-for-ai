"""
Tests for Calculus module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from math_utils.calculus import Calculus


class TestNumericalDerivatives:
    """Test numerical differentiation."""
    
    def test_numerical_derivative_forward(self):
        """Test forward difference derivative."""
        f = lambda x: x ** 2
        result = Calculus.numerical_derivative(f, 3.0, method='forward')
        expected = 6.0  # d/dx(x^2) = 2x, at x=3 is 6
        assert np.isclose(result, expected, atol=1e-5)
    
    def test_numerical_derivative_central(self):
        """Test central difference derivative (more accurate)."""
        f = lambda x: x ** 2
        result = Calculus.numerical_derivative(f, 3.0, method='central')
        expected = 6.0
        assert np.isclose(result, expected, atol=1e-8)
    
    def test_numerical_derivative_sin(self):
        """Test derivative of sin(x)."""
        f = lambda x: np.sin(x)
        result = Calculus.numerical_derivative(f, np.pi / 4, method='central')
        expected = np.cos(np.pi / 4)  # d/dx(sin(x)) = cos(x)
        assert np.isclose(result, expected, atol=1e-8)
    
    def test_numerical_gradient(self):
        """Test gradient computation."""
        f = lambda x: x[0] ** 2 + x[1] ** 2
        x = np.array([3.0, 4.0])
        grad = Calculus.numerical_gradient(f, x)
        expected = np.array([6.0, 8.0])
        assert np.allclose(grad, expected, atol=1e-6)
    
    def test_numerical_gradient_rosenbrock(self):
        """Test gradient of Rosenbrock function."""
        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        
        x = np.array([1.0, 1.0])
        grad = Calculus.numerical_gradient(f, x)
        expected = np.array([0.0, 0.0])  # Minimum at (1, 1)
        assert np.allclose(grad, expected, atol=1e-6)


class TestJacobianAndHessian:
    """Test Jacobian and Hessian computations."""
    
    def test_jacobian_linear(self):
        """Test Jacobian of linear function."""
        f = lambda x: np.array([2 * x[0], 3 * x[1], x[0] + x[1]])
        x = np.array([1.0, 2.0])
        J = Calculus.jacobian(f, x)
        expected = np.array([[2, 0], [0, 3], [1, 1]])
        assert np.allclose(J, expected, atol=1e-6)
    
    def test_hessian_quadratic(self):
        """Test Hessian of quadratic function."""
        f = lambda x: x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[0] * x[1]
        x = np.array([1.0, 2.0])
        H = Calculus.hessian(f, x)
        expected = np.array([[2, 3], [3, 4]])
        assert np.allclose(H, expected, atol=1e-5)
    
    def test_hessian_rosenbrock(self):
        """Test Hessian of Rosenbrock function at minimum."""
        def f(x):
            return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        
        x = np.array([1.0, 1.0])
        H = Calculus.hessian(f, x)
        expected = np.array([[802, -400], [-400, 200]])
        assert np.allclose(H, expected, atol=1e-4)


class TestNumericalIntegration:
    """Test numerical integration methods."""
    
    def test_trapezoidal_polynomial(self):
        """Test trapezoidal rule on polynomial."""
        f = lambda x: x ** 2
        result = Calculus.trapezoidal(f, 0, 1, n=1000)
        expected = 1 / 3  # ∫x²dx from 0 to 1 = 1/3
        assert np.isclose(result, expected, atol=1e-5)
    
    def test_simpson_polynomial(self):
        """Test Simpson's rule on polynomial."""
        f = lambda x: x ** 2
        result = Calculus.simpson(f, 0, 1, n=100)
        expected = 1 / 3
        assert np.isclose(result, expected, atol=1e-8)
    
    def test_simpson_sin(self):
        """Test Simpson's rule on sin(x)."""
        f = lambda x: np.sin(x)
        result = Calculus.simpson(f, 0, np.pi, n=100)
        expected = 2.0  # ∫sin(x)dx from 0 to π = 2
        assert np.isclose(result, expected, atol=1e-6)
    
    def test_gaussian_quadrature(self):
        """Test Gaussian quadrature."""
        f = lambda x: np.exp(x)
        result = Calculus.gaussian_quadrature(f, -1, 1, n=5)
        expected = np.exp(1) - np.exp(-1)  # ∫e^x dx from -1 to 1
        assert np.isclose(result, expected, atol=1e-6)
    
    def test_monte_carlo_integration(self):
        """Test Monte Carlo integration."""
        f = lambda x: x ** 2
        result, error = Calculus.monte_carlo_integrate(f, [(0, 1)], n_samples=100000)
        expected = 1 / 3
        assert abs(result - expected) < 0.01  # Within 1% for large sample


class TestRootFinding:
    """Test root-finding algorithms."""
    
    def test_bisection(self):
        """Test bisection method."""
        f = lambda x: x ** 2 - 2
        root = Calculus.bisection(f, 0, 2)
        expected = np.sqrt(2)
        assert np.isclose(root, expected, atol=1e-8)
    
    def test_newton_raphson(self):
        """Test Newton-Raphson method."""
        f = lambda x: x ** 2 - 2
        root = Calculus.newton_raphson(f, 1.5)
        expected = np.sqrt(2)
        assert np.isclose(root, expected, atol=1e-10)
    
    def test_secant(self):
        """Test secant method."""
        f = lambda x: x ** 3 - x - 2
        root = Calculus.secant(f, 1.0, 2.0)
        expected = 1.5213797068045676  # Known root
        assert np.isclose(root, expected, atol=1e-8)


class TestOptimization:
    """Test optimization methods."""
    
    def test_gradient_descent_step(self):
        """Test single gradient descent step."""
        f = lambda x: x ** 2
        x = np.array([3.0])
        new_x = Calculus.gradient_descent_step(f, x, learning_rate=0.1)
        expected = 3.0 - 0.1 * 6.0  # x - lr * 2x
        assert np.isclose(new_x, expected)
    
    def test_gradient_ascent_step(self):
        """Test single gradient ascent step."""
        f = lambda x: -x ** 2  # Maximizing negative quadratic
        x = np.array([3.0])
        new_x = Calculus.gradient_ascent_step(f, x, learning_rate=0.1)
        # Gradient is -2x = -6, ascent adds: 3 + 0.1 * 6 = 3.6
        expected = 3.0 + 0.1 * 6.0
        assert np.isclose(new_x, expected)
    
    def test_newton_step(self):
        """Test Newton's method step."""
        f = lambda x: (x - 2) ** 2
        x = np.array([4.0])
        new_x = Calculus.newton_step(f, x)
        expected = 2.0  # Newton should find minimum in one step for quadratic
        assert np.isclose(new_x, expected, atol=1e-6)


class TestTaylorSeries:
    """Test Taylor series approximation."""
    
    def test_taylor_expansion_point(self):
        """Test Taylor series at expansion point."""
        f = lambda x: np.exp(x)
        x, taylor = Calculus.taylor_series(f, x0=0, n=5, num_points=100, x_range=(0, 0))
        # At x0=0, Taylor series of e^x should give 1
        assert np.isclose(taylor[0], 1.0, atol=1e-6)


class TestChainRule:
    """Test chain rule implementations."""
    
    def test_chain_rule_single(self):
        """Test chain rule for single variable."""
        f = lambda x: x ** 2  # outer
        g = lambda x: 3 * x + 1  # inner
        x = 2.0
        
        # d/dx f(g(x)) = f'(g(x)) * g'(x) = 2*g(x) * 3 = 2*(3x+1)*3
        result = Calculus.chain_rule_single(f, g, x)
        expected = 2 * (3 * x + 1) * 3  # 42 at x=2
        assert np.isclose(result, expected, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
