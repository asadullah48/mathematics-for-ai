"""
Tests for Linear Models.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_models.linear_models import LinearRegression, LogisticRegression, Ridge, Lasso


class TestLinearRegression:
    """Test Linear Regression."""
    
    def test_fit_predict_perfect(self):
        """Test perfect fit on linear data."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_weights = np.array([1.5, -2.0, 0.5])
        y = X @ true_weights + 0.1 * np.random.randn(100)
        
        model = LinearRegression(n_iterations=1000, learning_rate=0.1)
        model.fit(X, y)
        
        # Predictions should be close to actual
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 0.1
    
    def test_fit_normal_equation(self):
        """Test normal equation solution."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_weights = np.array([1.0, 2.0, 3.0])
        y = X @ true_weights
        
        model = LinearRegression(method='normal_equation')
        model.fit(X, y)
        
        # Should recover exact weights (no noise)
        assert np.allclose(model.weights, true_weights, atol=1e-10)
    
    def test_r2_score(self):
        """Test R² score computation."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
        
        model = LinearRegression(n_iterations=1000, learning_rate=0.1)
        model.fit(X, y)
        
        r2 = model.score(X, y)
        assert r2 > 0.9  # Should explain most variance
    
    def test_fit_intercept(self):
        """Test fitting with intercept."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([3, 5, 7, 9])  # y = 2x + 1
        
        model = LinearRegression(fit_intercept=True, n_iterations=1000, learning_rate=0.1)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        assert np.allclose(y_pred, y, atol=0.5)


class TestLogisticRegression:
    """Test Logistic Regression."""
    
    def test_binary_classification(self):
        """Test binary classification."""
        np.random.seed(42)
        
        # Generate linearly separable data
        X1 = np.random.randn(50, 2) + np.array([2, 2])
        X2 = np.random.randn(50, 2) + np.array([-2, -2])
        X = np.vstack([X1, X2])
        y = np.array([1] * 50 + [0] * 50)
        
        model = LogisticRegression(n_iterations=1000, learning_rate=0.1)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        assert accuracy > 0.9
    
    def test_predict_proba(self):
        """Test probability predictions."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(n_iterations=500, learning_rate=0.1)
        model.fit(X, y)
        
        probs = model.predict_proba(X)
        
        # Probabilities should be in [0, 1]
        assert np.all(probs >= 0) and np.all(probs <= 1)
    
    def test_sigmoid(self):
        """Test sigmoid function."""
        z = np.array([-10, 0, 10])
        result = LogisticRegression.sigmoid(z)
        expected = np.array([4.54e-5, 0.5, 0.99995])
        assert np.allclose(result, expected, atol=1e-4)


class TestRidge:
    """Test Ridge Regression."""
    
    def test_ridge_regularization(self):
        """Test Ridge regularization effect."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X @ np.random.randn(10) + np.random.randn(50) * 0.1
        
        # Fit with different regularization strengths
        model_no_reg = Ridge(alpha=0, n_iterations=1000, learning_rate=0.1)
        model_high_reg = Ridge(alpha=100, n_iterations=1000, learning_rate=0.1)
        
        model_no_reg.fit(X, y)
        model_high_reg.fit(X, y)
        
        # Higher regularization should give smaller weights
        norm_no_reg = np.linalg.norm(model_no_reg.weights)
        norm_high_reg = np.linalg.norm(model_high_reg.weights)
        
        assert norm_high_reg < norm_no_reg


class TestLasso:
    """Test Lasso Regression."""
    
    def test_lasso_sparsity(self):
        """Test Lasso produces sparse solutions."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        # Only first 3 features are relevant
        true_weights = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0])
        y = X @ true_weights + np.random.randn(100) * 0.1
        
        model = Lasso(alpha=1.0, n_iterations=500)
        model.fit(X, y)
        
        # Many weights should be close to zero
        near_zero = np.sum(np.abs(model.weights) < 0.1)
        assert near_zero >= 5  # At least 5 weights should be near zero


class TestElasticNet:
    """Test Elastic Net."""
    
    def test_elastic_net(self):
        """Test Elastic Net combines L1 and L2."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = X @ np.random.randn(10) + np.random.randn(100) * 0.1
        
        model = ElasticNet(alpha=1.0, l1_ratio=0.5, n_iterations=500, learning_rate=0.1)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        assert mse < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
