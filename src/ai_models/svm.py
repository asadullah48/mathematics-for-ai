"""
Support Vector Machine Module

Implementation of SVM from scratch including:
- Linear SVM
- Kernel SVM (RBF, Polynomial, Sigmoid)
- SMO (Sequential Minimal Optimization) algorithm
"""

import numpy as np
from typing import Optional, Tuple, Callable
from abc import ABC, abstractmethod


class Kernel(ABC):
    """Base class for kernel functions."""
    
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        pass


class LinearKernel(Kernel):
    """Linear kernel: K(x, y) = x^T y"""
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return X1 @ X2.T


class RBFKernel(Kernel):
    """
    Radial Basis Function (Gaussian) kernel.
    
    K(x, y) = exp(-gamma * ||x - y||^2)
    """
    
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Compute squared Euclidean distances
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances = X1_sq + X2_sq - 2 * X1 @ X2.T
        
        return np.exp(-self.gamma * distances)


class PolynomialKernel(Kernel):
    """
    Polynomial kernel.
    
    K(x, y) = (gamma * x^T y + coef0)^degree
    """
    
    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return (self.gamma * X1 @ X2.T + self.coef0) ** self.degree


class SigmoidKernel(Kernel):
    """
    Sigmoid kernel.
    
    K(x, y) = tanh(gamma * x^T y + coef0)
    """
    
    def __init__(self, gamma: float = 1.0, coef0: float = 0.0):
        self.gamma = gamma
        self.coef0 = coef0
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.tanh(self.gamma * X1 @ X2.T + self.coef0)


class SupportVectorMachine:
    """
    Support Vector Machine using SMO algorithm.
    
    Parameters:
        C: Regularization parameter (trade-off between margin and errors)
        kernel: Kernel function or string ('linear', 'rbf', 'poly', 'sigmoid')
        gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
        degree: Degree for polynomial kernel
        coef0: Independent term in kernel function
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        random_state: Random seed
    """
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf',
                 gamma: float = 'scale', degree: int = 3, coef0: float = 1.0,
                 tol: float = 1e-3, max_iter: int = 1000,
                 random_state: Optional[int] = None):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.alphas = None
        self.b = None
        self.support_vectors = None
        self.support_vector_indices = None
        self.n_support_vectors = None
        
        # Set up kernel
        self._setup_kernel()
    
    def _setup_kernel(self):
        """Initialize kernel function."""
        if isinstance(self.kernel_name, Kernel):
            self.kernel = self.kernel_name
        elif self.kernel_name == 'linear':
            self.kernel = LinearKernel()
        elif self.kernel_name == 'rbf':
            if self.gamma == 'scale':
                self.gamma = 1.0  # Will be updated in fit
            self.kernel = RBFKernel(self.gamma)
        elif self.kernel_name == 'poly':
            self.kernel = PolynomialKernel(self.degree, self.gamma, self.coef0)
        elif self.kernel_name == 'sigmoid':
            self.kernel = SigmoidKernel(self.gamma, self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_name}")
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix for training data."""
        return self.kernel(X, X)
    
    def _initialize_alphas(self, n_samples: int):
        """Initialize Lagrange multipliers."""
        self.alphas = np.zeros(n_samples)
    
    def _compute_decision_function(self, X: np.ndarray, K: np.ndarray = None) -> np.ndarray:
        """Compute decision function values."""
        if K is None:
            K = self.kernel(X, self.support_vectors)
        
        return K @ (self.alphas[self.support_vector_indices] * self.y[self.support_vector_indices]) + self.b
    
    def _take_step(self, i1: int, i2: int, K: np.ndarray, y: np.ndarray) -> bool:
        """
        Take optimization step for alpha_i1 and alpha_i2.
        
        Implements SMO algorithm.
        """
        if i1 == i2:
            return False
        
        alpha1, alpha2 = self.alphas[i1], self.alphas[i2]
        y1, y2 = y[i1], y[i2]
        K11, K22 = K[i1, i1], K[i2, i2]
        K12 = K[i1, i2]
        
        # Compute eta
        eta = K11 + K22 - 2 * K12
        
        if eta <= 0:
            return False
        
        # Compute bounds
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        
        if L == H:
            return False
        
        # Compute optimal alpha2
        a2 = alpha2 + y2 * (self.E[i1] - self.E[i2]) / eta
        
        # Clip alpha2
        if a2 < L:
            a2 = L
        elif a2 > H:
            a2 = H
        
        if abs(a2 - alpha2) < 1e-5:
            return False
        
        # Update alpha2
        self.alphas[i2] = a2
        
        # Update alpha1
        self.alphas[i1] = alpha1 + y1 * y2 * (alpha2 - a2)
        
        # Update bias
        b1 = self.b - self.E[i1] - y1 * (a2 - alpha2) * K12 - y1 * (self.alphas[i1] - alpha1) * K11
        b2 = self.b - self.E[i2] - y2 * (a2 - alpha2) * K22 - y2 * (self.alphas[i1] - alpha1) * K12
        
        if 0 < self.alphas[i1] < self.C:
            self.b = b1
        elif 0 < self.alphas[i2] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # Update error cache
        for i in range(len(self.E)):
            if i != i1 and i != i2:
                self.E[i] += y1 * (self.alphas[i1] - alpha1) * K[i1, i]
                self.E[i] += y2 * (self.alphas[i2] - alpha2) * K[i2, i]
        
        self.E[i1] = 0
        self.E[i2] = 0
        
        return True
    
    def _examine_example(self, i2: int, K: np.ndarray, y: np.ndarray) -> int:
        """Examine a single example for optimization."""
        y2 = y[i2]
        E2 = self.E[i2]
        r2 = E2 * y2
        
        # Check KKT conditions
        if ((r2 < -self.tol and self.alphas[i2] < self.C) or
            (r2 > self.tol and self.alphas[i2] > 0)):
            
            # Look for i1 that maximizes |E1 - E2|
            if np.sum(self.alphas > 0) > 1:
                if E2 > 0:
                    i1 = np.argmin(self.E)
                else:
                    i1 = np.argmax(self.E)
                
                if self._take_step(i1, i2, K, y):
                    return 1
            
            # Try all support vectors
            for i1 in np.where(self.alphas > 0)[0]:
                if i1 == i2:
                    continue
                if self._take_step(i1, i2, K, y):
                    return 1
            
            # Try all examples
            n = len(self.alphas)
            start = np.random.randint(0, n)
            for i in range(n):
                i1 = (i + start) % n
                if i1 == i2:
                    continue
                if self._take_step(i1, i2, K, y):
                    return 1
        
        return 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupportVectorMachine':
        """
        Fit SVM using SMO algorithm.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) with values in {-1, 1}
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Convert labels to {-1, 1}
        y = np.asarray(y)
        y_unique = np.unique(y)
        if len(y_unique) == 2:
            # Map to {-1, 1}
            self.classes = y_unique
            y = np.where(y == y_unique[1], 1, -1)
        else:
            raise ValueError("SVM is for binary classification")
        
        self.y = y
        
        # Set gamma if 'scale'
        if self.gamma == 'scale':
            self.gamma = 1.0 / (n_features * np.var(X))
            self._setup_kernel()
        
        # Initialize
        self._initialize_alphas(n_samples)
        self.b = 0.0
        self.E = np.zeros(n_samples)  # Error cache
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # SMO algorithm
        num_changed_alphas = 0
        examine_all = True
        iteration = 0
        
        while iteration < self.max_iter and (examine_all or num_changed_alphas > 0):
            num_changed_alphas = 0
            
            if examine_all:
                # Examine all examples
                for i in range(n_samples):
                    num_changed_alphas += self._examine_example(i, K, y)
            else:
                # Examine only support vectors
                for i in np.where((self.alphas > 0) & (self.alphas < self.C))[0]:
                    num_changed_alphas += self._examine_example(i, K, y)
            
            examine_all = not examine_all
            iteration += 1
        
        # Find support vectors
        sv_threshold = 1e-5
        sv_mask = self.alphas > sv_threshold
        self.support_vector_indices = np.where(sv_mask)[0]
        self.support_vectors = X[sv_mask]
        self.n_support_vectors = len(self.support_vector_indices)
        
        # Compute final bias using support vectors on margin
        margin_mask = (self.alphas > sv_threshold) & (self.alphas < self.C - sv_threshold)
        if np.sum(margin_mask) > 0:
            margin_sv_indices = np.where(margin_mask)[0]
            b_values = []
            for i in margin_sv_indices:
                b_values.append(y[i] - np.sum(self.alphas[sv_mask] * y[sv_mask] * K[sv_mask, i]))
            self.b = np.mean(b_values)
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function (distance from hyperplane)."""
        K = self.kernel(X, self.support_vectors)
        return self._compute_decision_function(X, K)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        decisions = self.decision_function(X)
        predictions = np.where(decisions >= 0, 1, -1)
        
        # Map back to original labels
        if hasattr(self, 'classes'):
            predictions = np.where(predictions == 1, self.classes[1], self.classes[0])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability estimates using Platt scaling.
        
        Returns:
            Probability of positive class
        """
        decisions = self.decision_function(X)
        
        # Platt scaling (simplified)
        prob = 1 / (1 + np.exp(-decisions))
        
        return np.column_stack([1 - prob, prob])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'C': self.C,
            'kernel': self.kernel_name,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0
        }


class SVR:
    """
    Support Vector Regression using epsilon-insensitive loss.
    
    Parameters:
        C: Regularization parameter
        epsilon: Epsilon in epsilon-SVR model
        kernel: Kernel function or string
        gamma: Kernel coefficient
        tol: Convergence tolerance
        max_iter: Maximum iterations
    """
    
    def __init__(self, C: float = 1.0, epsilon: float = 0.1,
                 kernel: str = 'rbf', gamma: float = 'scale',
                 tol: float = 1e-3, max_iter: int = 1000):
        self.C = C
        self.epsilon = epsilon
        self.kernel_name = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        
        self.alphas = None
        self.alphas_star = None
        self.b = None
        self.support_vectors = None
    
    def _setup_kernel(self, X: np.ndarray):
        """Initialize kernel."""
        n_features = X.shape[1]
        
        if self.gamma == 'scale':
            self.gamma = 1.0 / (n_features * np.var(X))
        
        if self.kernel_name == 'rbf':
            self.kernel = RBFKernel(self.gamma)
        elif self.kernel_name == 'linear':
            self.kernel = LinearKernel()
        elif self.kernel_name == 'poly':
            self.kernel = PolynomialKernel(3, self.gamma, 1.0)
        else:
            self.kernel = RBFKernel(self.gamma)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVR':
        """
        Fit SVR model.
        
        Uses dual coordinate descent method.
        """
        n_samples = X.shape[0]
        self._setup_kernel(X)
        
        # Compute kernel matrix
        K = self._setup_kernel(X)
        K = self.kernel(X, X)
        
        # Initialize alphas
        self.alphas = np.zeros(n_samples)
        self.alphas_star = np.zeros(n_samples)
        
        # Simplified gradient descent approach
        learning_rate = 0.01
        
        for iteration in range(self.max_iter):
            for i in range(n_samples):
                # Compute prediction
                f_i = np.sum((self.alphas - self.alphas_star) * K[:, i]) + self.b
                
                # Compute gradients
                if f_i > y[i] + self.epsilon:
                    grad = -1
                elif f_i < y[i] - self.epsilon:
                    grad = 1
                else:
                    grad = 0
                
                # Update alphas
                if grad != 0:
                    self.alphas[i] += learning_rate * grad
                    self.alphas[i] = np.clip(self.alphas[i], 0, self.C)
        
        # Find support vectors
        sv_threshold = 1e-5
        sv_mask = (np.abs(self.alphas - self.alphas_star) > sv_threshold)
        self.support_vectors = X[sv_mask]
        
        # Compute bias
        if np.sum(sv_mask) > 0:
            self.b = np.mean(y[sv_mask] - np.sum((self.alphas - self.alphas_star) * K[sv_mask, :].T, axis=1))
        else:
            self.b = np.mean(y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values."""
        K = self.kernel(X, self.support_vectors)
        return K @ (self.alphas - self.alphas_star) + self.b
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
