"""
Linear Models Module

Implementation of linear models from scratch including:
- Linear Regression (Ordinary Least Squares)
- Logistic Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from abc import ABC, abstractmethod


class BaseLinearModel(ABC):
    """Base class for linear models."""
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 fit_intercept: bool = True, random_state: Optional[int] = None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column to features."""
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X
    
    def _initialize_weights(self, n_features: int):
        """Initialize weights."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.fit_intercept:
            self.weights = np.random.randn(n_features + 1) * 0.01
            self.bias = 0
        else:
            self.weights = np.random.randn(n_features) * 0.01
            self.bias = 0
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute cost function."""
        pass
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients."""
        pass


class LinearRegression(BaseLinearModel):
    """
    Linear Regression using Gradient Descent.
    
    Can also solve using Normal Equation for closed-form solution.
    
    Parameters:
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of iterations for gradient descent
        fit_intercept: Whether to fit intercept term
        random_state: Random seed for reproducibility
        method: 'gradient_descent' or 'normal_equation'
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 fit_intercept: bool = True, random_state: Optional[int] = None,
                 method: str = 'gradient_descent'):
        super().__init__(learning_rate, n_iterations, fit_intercept, random_state)
        self.method = method
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
            
        Returns:
            Self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.method == 'normal_equation':
            self._fit_normal_equation(X, y)
        else:
            self._fit_gradient_descent(X, y)
        
        return self
    
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """Fit using normal equation: w = (X^T X)^(-1) X^T y"""
        X_processed = self._add_intercept(X)
        
        # Add regularization for numerical stability
        identity = np.eye(X_processed.shape[1])
        identity[0, 0] = 0  # Don't regularize intercept
        
        XtX = X_processed.T @ X_processed
        XtX_reg = XtX + 1e-10 * identity
        Xty = X_processed.T @ y
        
        self.weights = np.linalg.solve(XtX_reg, Xty)
        if self.fit_intercept:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """Fit using gradient descent."""
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        X_processed = self._add_intercept(X)
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            predictions = X_processed @ self.weights if self.fit_intercept else X_processed @ self.weights + self.bias
            if self.fit_intercept:
                predictions = X_processed @ self.weights
            else:
                predictions = X @ self.weights + self.bias
            
            # Compute cost (MSE)
            errors = predictions - y
            cost = np.mean(errors ** 2)
            self.cost_history.append(cost)
            
            # Compute gradients
            if self.fit_intercept:
                gradients = (2 / n_samples) * X_processed.T @ errors
                self.weights -= self.learning_rate * gradients
            else:
                dw = (2 / n_samples) * X.T @ errors
                db = (2 / n_samples) * np.sum(errors)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        X = np.asarray(X)
        if self.fit_intercept:
            return X @ self.weights + self.bias
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.
        
        Args:
            X: Features
            y: True target values
            
        Returns:
            R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE cost."""
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)


class Ridge(BaseLinearModel):
    """
    Ridge Regression (L2 regularization).
    
    Parameters:
        alpha: Regularization strength
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of iterations
        fit_intercept: Whether to fit intercept term
        random_state: Random seed
    """
    
    def __init__(self, alpha: float = 1.0, learning_rate: float = 0.01,
                 n_iterations: int = 1000, fit_intercept: bool = True,
                 random_state: Optional[int] = None):
        super().__init__(learning_rate, n_iterations, fit_intercept, random_state)
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Ridge':
        """Fit Ridge regression model."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        X_processed = self._add_intercept(X)
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            predictions = X_processed @ self.weights if self.fit_intercept else X @ self.weights + self.bias
            if not self.fit_intercept:
                predictions = X @ self.weights + self.bias
            
            # Compute cost with L2 regularization
            errors = predictions - y
            if self.fit_intercept:
                reg_term = self.alpha * np.sum(self.weights[1:] ** 2)
            else:
                reg_term = self.alpha * np.sum(self.weights ** 2)
            cost = np.mean(errors ** 2) + reg_term
            self.cost_history.append(cost)
            
            # Compute gradients with regularization
            if self.fit_intercept:
                gradients = (2 / n_samples) * X_processed.T @ errors
                gradients[1:] += 2 * self.alpha * self.weights[1:]  # Don't regularize intercept
                self.weights -= self.learning_rate * gradients
            else:
                dw = (2 / n_samples) * X.T @ errors + 2 * self.alpha * self.weights
                db = (2 / n_samples) * np.sum(errors)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.asarray(X)
        if self.fit_intercept:
            return X @ self.weights + self.bias
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class Lasso(BaseLinearModel):
    """
    Lasso Regression (L1 regularization).
    
    Uses coordinate descent optimization.
    
    Parameters:
        alpha: Regularization strength
        learning_rate: Learning rate
        n_iterations: Number of iterations
        fit_intercept: Whether to fit intercept term
        random_state: Random seed
        tol: Tolerance for convergence
    """
    
    def __init__(self, alpha: float = 1.0, learning_rate: float = 0.01,
                 n_iterations: int = 1000, fit_intercept: bool = True,
                 random_state: Optional[int] = None, tol: float = 1e-6):
        super().__init__(learning_rate, n_iterations, fit_intercept, random_state)
        self.alpha = alpha
        self.tol = tol
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Lasso':
        """Fit Lasso regression using coordinate descent."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        if self.fit_intercept:
            self.weights = np.zeros(n_features + 1)
            self.weights[0] = 0  # Intercept
        else:
            self.weights = np.zeros(n_features)
        
        self.cost_history = []
        
        # Coordinate descent
        for iteration in range(self.n_iterations):
            weights_old = self.weights.copy()
            
            for j in range(len(self.weights)):
                if self.fit_intercept and j == 0:
                    # Update intercept (no regularization)
                    residual = y - X @ self.weights[1:] - self.weights[0]
                    self.weights[0] = np.mean(residual)
                else:
                    # Update weight with soft thresholding
                    idx = j - 1 if self.fit_intercept else j
                    
                    if self.fit_intercept:
                        residual = y - self.weights[0] - X @ self.weights[1:]
                        x_j = X[:, idx]
                        w_j = self.weights[j]
                        self.weights[j] = 0
                        residual += w_j * x_j
                        
                        rho = x_j @ residual
                        z = x_j @ x_j
                        
                        if z == 0:
                            self.weights[j] = 0
                        else:
                            # Soft thresholding
                            if rho < -self.alpha * n_samples:
                                self.weights[j] = (rho + self.alpha * n_samples) / z
                            elif rho > self.alpha * n_samples:
                                self.weights[j] = (rho - self.alpha * n_samples) / z
                            else:
                                self.weights[j] = 0
                    else:
                        residual = y - X @ self.weights
                        x_j = X[:, j]
                        w_j = self.weights[j]
                        self.weights[j] = 0
                        residual += w_j * x_j
                        
                        rho = x_j @ residual
                        z = x_j @ x_j
                        
                        if z == 0:
                            self.weights[j] = 0
                        else:
                            if rho < -self.alpha * n_samples:
                                self.weights[j] = (rho + self.alpha * n_samples) / z
                            elif rho > self.alpha * n_samples:
                                self.weights[j] = (rho - self.alpha * n_samples) / z
                            else:
                                self.weights[j] = 0
            
            # Check convergence
            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                break
            
            # Record cost
            predictions = self.predict(X)
            cost = np.mean((predictions - y) ** 2) + self.alpha * np.sum(np.abs(self.weights))
            self.cost_history.append(cost)
        
        if self.fit_intercept:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.asarray(X)
        if self.fit_intercept:
            return X @ self.weights + self.bias
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LogisticRegression(BaseLinearModel):
    """
    Logistic Regression for binary classification.
    
    Parameters:
        learning_rate: Learning rate for gradient descent
        n_iterations: Number of iterations
        fit_intercept: Whether to fit intercept term
        random_state: Random seed
        regularization: 'none', 'l1', or 'l2'
        reg_alpha: Regularization strength
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 fit_intercept: bool = True, random_state: Optional[int] = None,
                 regularization: str = 'none', reg_alpha: float = 0.01):
        super().__init__(learning_rate, n_iterations, fit_intercept, random_state)
        self.regularization = regularization
        self.reg_alpha = reg_alpha
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit logistic regression model.
        
        Args:
            X: Features (n_samples, n_features)
            y: Binary target (n_samples,)
            
        Returns:
            Self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        X_processed = self._add_intercept(X)
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            z = X_processed @ self.weights if self.fit_intercept else X @ self.weights + self.bias
            predictions = self.sigmoid(z)
            
            # Compute cost (binary cross-entropy)
            epsilon = 1e-15  # Avoid log(0)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            
            if self.fit_intercept:
                cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            else:
                cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            # Add regularization
            if self.regularization == 'l2':
                reg_term = (self.reg_alpha / 2) * np.sum(self.weights ** 2)
                cost += reg_term
            elif self.regularization == 'l1':
                reg_term = self.reg_alpha * np.sum(np.abs(self.weights))
                cost += reg_term
            
            self.cost_history.append(cost)
            
            # Compute gradients
            error = predictions - y
            
            if self.fit_intercept:
                gradients = (1 / n_samples) * X_processed.T @ error
                
                # Add regularization gradient
                if self.regularization == 'l2':
                    gradients[1:] += self.reg_alpha * self.weights[1:]
                elif self.regularization == 'l1':
                    gradients[1:] += self.reg_alpha * np.sign(self.weights[1:])
                
                self.weights -= self.learning_rate * gradients
            else:
                dw = (1 / n_samples) * X.T @ error
                db = (1 / n_samples) * np.sum(error)
                
                if self.regularization == 'l2':
                    dw += self.reg_alpha * self.weights
                elif self.regularization == 'l1':
                    dw += self.reg_alpha * np.sign(self.weights)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates.
        
        Args:
            X: Features
            
        Returns:
            Probability of positive class (n_samples,)
        """
        X = np.asarray(X)
        if self.fit_intercept:
            z = X @ self.weights + self.bias
        else:
            z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            threshold: Classification threshold
            
        Returns:
            Predicted labels (0 or 1)
        """
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function (log-odds)."""
        X = np.asarray(X)
        if self.fit_intercept:
            return X @ self.weights + self.bias
        return X @ self.weights + self.bias


class ElasticNet(BaseLinearModel):
    """
    Elastic Net Regression (L1 + L2 regularization).
    
    Parameters:
        alpha: Regularization strength
        l1_ratio: Ratio of L1 regularization (0 = pure L2, 1 = pure L1)
        learning_rate: Learning rate
        n_iterations: Number of iterations
        fit_intercept: Whether to fit intercept term
        random_state: Random seed
    """
    
    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 learning_rate: float = 0.01, n_iterations: int = 1000,
                 fit_intercept: bool = True, random_state: Optional[int] = None):
        super().__init__(learning_rate, n_iterations, fit_intercept, random_state)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNet':
        """Fit Elastic Net model."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)
        
        X_processed = self._add_intercept(X)
        self.cost_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            predictions = X_processed @ self.weights if self.fit_intercept else X @ self.weights + self.bias
            if not self.fit_intercept:
                predictions = X @ self.weights + self.bias
            
            errors = predictions - y
            
            # Compute cost
            mse = np.mean(errors ** 2)
            if self.fit_intercept:
                l1_term = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights[1:]))
                l2_term = (self.alpha * (1 - self.l1_ratio) / 2) * np.sum(self.weights[1:] ** 2)
            else:
                l1_term = self.alpha * self.l1_ratio * np.sum(np.abs(self.weights))
                l2_term = (self.alpha * (1 - self.l1_ratio) / 2) * np.sum(self.weights ** 2)
            
            cost = mse + l1_term + l2_term
            self.cost_history.append(cost)
            
            # Compute gradients
            if self.fit_intercept:
                gradients = (2 / n_samples) * X_processed.T @ errors
                
                # L2 regularization gradient
                l2_grad = self.alpha * (1 - self.l1_ratio) * self.weights
                l2_grad[0] = 0  # Don't regularize intercept
                
                # L1 regularization gradient (subgradient)
                l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights)
                l1_grad[0] = 0
                
                gradients += l2_grad + l1_grad
                self.weights -= self.learning_rate * gradients
            else:
                dw = (2 / n_samples) * X.T @ errors
                dw += self.alpha * (1 - self.l1_ratio) * self.weights
                dw += self.alpha * self.l1_ratio * np.sign(self.weights)
                
                db = (2 / n_samples) * np.sum(errors)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.asarray(X)
        if self.fit_intercept:
            return X @ self.weights + self.bias
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
