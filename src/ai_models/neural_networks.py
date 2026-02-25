"""
Neural Networks Module

Implementation of neural networks from scratch including:
- Autograd engine for automatic differentiation
- Various layer types (Dense, Conv2D, RNN, LSTM)
- Activation functions
- Loss functions
- Optimizers (SGD, Adam, RMSprop)
"""

import numpy as np
from typing import Optional, List, Tuple, Union, Callable, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json


# ==================== Autograd Engine ====================

class Tensor:
    """
    Tensor class with automatic differentiation.
    
    Supports basic operations for building neural networks.
    """
    
    def __init__(self, data: Union[np.ndarray, float], 
                 requires_grad: bool = False, 
                 _prev: List['Tensor'] = None,
                 _op: str = '', label: str = ''):
        if isinstance(data, (int, float)):
            data = np.array(data)
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = _prev or []
        self._op = _op
        self.label = label
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad,
                    _prev=[self, other], _op='+')
        
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
            other.grad += np.ones_like(other.data) * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return Tensor(other) - self
    
    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad, _prev=[self], _op='neg')
        def _backward():
            self.grad -= out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad,
                    _prev=[self, other], _op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return Tensor(other) * (self ** -1)
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError("Power only supports scalar exponents")
        
        out = Tensor(self.data ** other, requires_grad=self.requires_grad, _prev=[self], _op=f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    
    def __matmul__(self, other):
        """Matrix multiplication."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad,
                    _prev=[self, other], _op='@')
        
        def _backward():
            if self.data.ndim == 1 and other.data.ndim == 1:
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad
            elif self.data.ndim == 2 and other.data.ndim == 2:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
            else:
                # Handle broadcasting cases
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad, _prev=[self], _op='exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad, _prev=[self], _op='log')
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                    requires_grad=self.requires_grad, _prev=[self], _op='sum')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad, _prev=[self], _op='mean')
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad / n
        out._backward = _backward
        return out
    
    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _prev=[self], _op='relu')
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def sigmoid(self):
        out = Tensor(1 / (1 + np.exp(-self.data)), requires_grad=self.requires_grad, _prev=[self], _op='sigmoid')
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        out = Tensor(np.tanh(self.data), requires_grad=self.requires_grad, _prev=[self], _op='tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    
    def softmax(self, axis=-1):
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        out = Tensor(exp_data / np.sum(exp_data, axis=axis, keepdims=True),
                    requires_grad=self.requires_grad, _prev=[self], _op='softmax')
        def _backward():
            # Softmax gradient
            s = out.data
            grad_input = s * (out.grad - np.sum(s * out.grad, axis=axis, keepdims=True))
            self.grad += grad_input
        out._backward = _backward
        return out
    
    def backward(self):
        """Compute gradients using backpropagation."""
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        
        build_topo(self)
        
        # Set gradient of output to 1
        self.grad = np.ones_like(self.data)
        
        # Backpropagate
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad = np.zeros_like(self.data)
    
    def item(self):
        """Get scalar value."""
        return self.data.item()


# ==================== Activation Functions ====================

class Activation:
    """Activation functions."""
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        s = Activation.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
    @staticmethod
    def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x > 0, 1, alpha * np.exp(x))
    
    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        return x * Activation.sigmoid(x)
    
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


# ==================== Layer Types ====================

@dataclass
class Layer:
    """Base layer configuration."""
    input_size: int
    output_size: int
    activation: str = 'relu'
    use_bias: bool = True


class Dense:
    """
    Fully connected (Dense) layer.
    """
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: str = 'relu', use_bias: bool = True,
                 weight_init: str = 'he'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        
        # Weight initialization
        if weight_init == 'he':
            scale = np.sqrt(2.0 / input_size)
        elif weight_init == 'xavier':
            scale = np.sqrt(2.0 / (input_size + output_size))
        elif weight_init == 'glorot':
            scale = np.sqrt(6.0 / (input_size + output_size))
        else:
            scale = 0.01
        
        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((1, output_size)) if use_bias else None
        
        # Cache for backprop
        self.input = None
        self.z = None
        self.dW = None
        self.db = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self.input = X
        self.z = X @ self.weights
        if self.use_bias:
            self.z += self.bias
        
        # Apply activation
        if self.activation == 'relu':
            return Activation.relu(self.z)
        elif self.activation == 'sigmoid':
            return Activation.sigmoid(self.z)
        elif self.activation == 'tanh':
            return Activation.tanh(self.z)
        elif self.activation == 'softmax':
            return Activation.softmax(self.z)
        elif self.activation == 'leaky_relu':
            return Activation.leaky_relu(self.z)
        elif self.activation == 'identity':
            return self.z
        else:
            return self.z
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # Gradient through activation
        if self.activation == 'relu':
            dz = dout * Activation.relu_derivative(self.z)
        elif self.activation == 'sigmoid':
            dz = dout * Activation.sigmoid_derivative(self.z)
        elif self.activation == 'tanh':
            dz = dout * Activation.tanh_derivative(self.z)
        elif self.activation == 'leaky_relu':
            dz = dout * Activation.leaky_relu_derivative(self.z)
        else:
            dz = dout
        
        # Compute gradients
        self.dW = self.input.T @ dz
        if self.use_bias:
            self.db = np.sum(dz, axis=0, keepdims=True)
        
        # Gradient for previous layer
        dinput = dz @ self.weights.T
        return dinput
    
    def update(self, learning_rate: float):
        """Update weights using gradients."""
        self.weights -= learning_rate * self.dW
        if self.use_bias:
            self.bias -= learning_rate * self.db


class Conv2D:
    """
    2D Convolutional layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1,
                 activation: str = 'relu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        
        # He initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, 
                                       kernel_size, kernel_size) * scale
        self.bias = np.zeros((1, out_channels, 1, 1))
        
        self.input = None
        self.z = None
        self.dW = None
        self.db = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Input tensor (batch, channels, height, width)
        """
        self.input = X
        batch_size, _, h, w = X.shape
        
        # Add padding
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            X_padded = X
        
        # Output dimensions
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Convolution
        self.z = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
        for b in range(batch_size):
            for c in range(self.out_channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        self.z[b, c, i, j] = np.sum(
                            X_padded[b, :, h_start:h_end, w_start:w_end] * 
                            self.weights[c]
                        ) + self.bias[0, c, 0, 0]
        
        # Activation
        if self.activation == 'relu':
            return Activation.relu(self.z)
        return self.z
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass."""
        batch_size = self.input.shape[0]
        
        # Gradient through activation
        if self.activation == 'relu':
            dz = dout * Activation.relu_derivative(self.z)
        else:
            dz = dout
        
        # Initialize gradients
        self.dW = np.zeros_like(self.weights)
        self.db = np.sum(dz, axis=(0, 2, 3), keepdims=True)
        dinput = np.zeros_like(self.input)
        
        # Add padding to input
        if self.padding > 0:
            input_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding),
                                               (self.padding, self.padding)), mode='constant')
            dinput_padded = np.zeros_like(input_padded)
        else:
            input_padded = self.input
            dinput_padded = dinput
        
        # Compute gradients
        for b in range(batch_size):
            for c in range(self.out_channels):
                for i in range(dz.shape[2]):
                    for j in range(dz.shape[3]):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        self.dW[c] += input_padded[b, :, h_start:h_end, w_start:w_end] * dz[b, c, i, j]
                        dinput_padded[b, :, h_start:h_end, w_start:w_end] += self.weights[c] * dz[b, c, i, j]
        
        # Remove padding from gradient
        if self.padding > 0:
            dinput = dinput_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return dinput
    
    def update(self, learning_rate: float):
        """Update weights."""
        self.weights -= learning_rate * self.dW
        self.bias -= learning_rate * self.db


class RNN:
    """
    Recurrent Neural Network layer.
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 activation: str = 'tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Initialize weights
        scale = 0.1
        self.Wxh = np.random.randn(input_size, hidden_size) * scale  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale  # Hidden to hidden
        self.Why = np.random.randn(hidden_size, hidden_size) * scale  # Hidden to output
        self.bias_h = np.zeros((1, hidden_size))
        self.bias_y = np.zeros((1, hidden_size))
        
        # Cache
        self.inputs = []
        self.hidden_states = []
        self.dWxh = None
        self.dWhh = None
        self.dWhy = None
    
    def forward(self, X: np.ndarray, hidden: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through time.
        
        Args:
            X: Input sequence (batch, seq_len, input_size)
            hidden: Initial hidden state (batch, hidden_size)
            
        Returns:
            Tuple of (outputs, final_hidden_state)
        """
        batch_size, seq_len, _ = X.shape
        
        if hidden is None:
            hidden = np.zeros((batch_size, self.hidden_size))
        
        self.inputs = []
        self.hidden_states = [hidden]
        outputs = []
        
        for t in range(seq_len):
            x_t = X[:, t, :]
            h_prev = hidden
            
            # Compute new hidden state
            hidden = np.tanh(x_t @ self.Wxh + h_prev @ self.Whh + self.bias_h)
            
            # Compute output
            y_t = hidden @ self.Why + self.bias_y
            
            self.inputs.append(x_t)
            self.hidden_states.append(hidden)
            outputs.append(y_t)
        
        return np.stack(outputs, axis=1), hidden
    
    def backward(self, dout: np.ndarray):
        """Backpropagation through time."""
        batch_size, seq_len, _ = dout.shape
        
        self.dWxh = np.zeros_like(self.Wxh)
        self.dWhh = np.zeros_like(self.Whh)
        self.dWhy = np.zeros_like(self.Why)
        
        dh_next = np.zeros((batch_size, self.hidden_size))
        
        for t in reversed(range(seq_len)):
            dy = dout[:, t, :]
            h_t = self.hidden_states[t + 1]
            h_prev = self.hidden_states[t]
            x_t = self.inputs[t]
            
            # Gradient for output weights
            self.dWhy += h_t.T @ dy
            
            # Gradient for hidden state
            dh = dy @ self.Why.T + dh_next
            
            # Gradient through tanh
            dh_raw = dh * (1 - h_t ** 2)
            
            # Gradient for recurrent weights
            self.dWxh += x_t.T @ dh_raw
            self.dWhh += h_prev.T @ dh_raw
            
            # Gradient for previous hidden state
            dh_next = dh_raw @ self.Whh.T
    
    def update(self, learning_rate: float):
        """Update weights."""
        self.Wxh -= learning_rate * self.dWxh
        self.Whh -= learning_rate * self.dWhh
        self.Why -= learning_rate * self.dWhy


# ==================== Loss Functions ====================

class Loss(ABC):
    """Base class for loss functions."""
    
    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass


class MSELoss(Loss):
    """Mean Squared Error loss."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropyLoss(Loss):
    """Cross-Entropy loss for classification."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if len(y_true.shape) == 1:
            # Integer labels
            batch_size = len(y_true)
            log_probs = -np.log(y_pred[np.arange(batch_size), y_true.astype(int)])
        else:
            # One-hot encoded
            log_probs = -np.sum(y_true * np.log(y_pred), axis=1)
        
        return np.mean(log_probs)
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        if len(y_true.shape) == 1:
            batch_size = len(y_true)
            grad = y_pred.copy()
            grad[np.arange(batch_size), y_true.astype(int)] -= 1
        else:
            grad = y_pred - y_true
        
        return grad / y_true.size


class BinaryCrossEntropyLoss(Loss):
    """Binary Cross-Entropy loss."""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + epsilon) / y_true.size


# ==================== Optimizers ====================

class Optimizer(ABC):
    """Base class for optimizers."""
    
    @abstractmethod
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]):
        pass


class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = None
    
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self.velocities is None:
            self.velocities = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * grad
            param += self.velocities[i]


class Adam:
    """Adam optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSprop:
    """RMSprop optimizer."""
    
    def __init__(self, learning_rate: float = 0.001, decay: float = 0.9,
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None
    
    def step(self, params: List[np.ndarray], grads: List[np.ndarray]):
        if self.cache is None:
            self.cache = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.cache[i] = self.decay * self.cache[i] + (1 - self.decay) * (grad ** 2)
            param -= self.learning_rate * grad / (np.sqrt(self.cache[i]) + self.epsilon)


# ==================== Neural Network ====================

class NeuralNetwork:
    """
    Feedforward Neural Network.
    
    Parameters:
        layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        activations: List of activation functions for each layer
        learning_rate: Learning rate
        optimizer: Optimizer name ('sgd', 'adam', 'rmsprop')
        loss: Loss function ('mse', 'cross_entropy', 'binary_cross_entropy')
    """
    
    def __init__(self, layer_sizes: List[int], 
                 activations: Optional[List[str]] = None,
                 learning_rate: float = 0.001,
                 optimizer: str = 'adam',
                 loss: str = 'cross_entropy'):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # Default activations
        if activations is None:
            activations = ['relu'] * (self.n_layers - 1) + ['softmax']
        
        self.activations = activations
        self.learning_rate = learning_rate
        
        # Initialize layers
        self.layers = []
        for i in range(self.n_layers):
            layer = Dense(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i]
            )
            self.layers.append(layer)
        
        # Initialize optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(learning_rate)
        else:
            self.optimizer = SGD(learning_rate)
        
        # Loss function
        if loss == 'mse':
            self.loss_fn = MSELoss()
        elif loss == 'binary_cross_entropy':
            self.loss_fn = BinaryCrossEntropyLoss()
        else:
            self.loss_fn = CrossEntropyLoss()
        
        self.training_history = {'loss': [], 'accuracy': []}
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Backward pass."""
        # Loss gradient
        dout = self.loss_fn.backward(y_pred, y_true)
        
        # Backprop through layers
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
    
    def update(self):
        """Update weights."""
        params = []
        grads = []
        
        for layer in self.layers:
            params.append(layer.weights)
            grads.append(layer.dW)
            if layer.use_bias:
                params.append(layer.bias)
                grads.append(layer.db)
        
        self.optimizer.step(params, grads)
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            epochs: int = 100, batch_size: int = 32,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True):
        """
        Train the network.
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            batch_size: Batch size
            validation_data: Optional (X_val, y_val) tuple
            verbose: Print training progress
        """
        n_samples = X.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.loss_fn.forward(y_pred, y_batch)
                epoch_loss += loss
                
                # Backward pass
                self.backward(y_pred, y_batch)
                
                # Update weights
                self.update()
            
            epoch_loss /= n_batches
            self.training_history['loss'].append(epoch_loss)
            
            # Compute accuracy
            if len(y.shape) == 1:
                y_pred_class = np.argmax(self.forward(X), axis=1)
                accuracy = np.mean(y_pred_class == y)
            else:
                y_pred_class = np.argmax(self.forward(X), axis=1)
                y_true_class = np.argmax(y, axis=1)
                accuracy = np.mean(y_pred_class == y_true_class)
            
            self.training_history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                val_msg = ""
                if validation_data:
                    X_val, y_val = validation_data
                    y_val_pred = self.forward(X_val)
                    val_loss = self.loss_fn.forward(y_val_pred, y_val)
                    val_msg = f" val_loss: {val_loss:.4f}"
                
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {accuracy:.4f}{val_msg}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)
    
    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        y_pred = self.predict_classes(X)
        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)
        return np.mean(y_pred == y)
    
    def save(self, filepath: str):
        """Save model weights."""
        weights = []
        for layer in self.layers:
            weights.append({
                'weights': layer.weights.tolist(),
                'bias': layer.bias.tolist() if layer.bias is not None else None,
                'activation': layer.activation
            })
        
        with open(filepath, 'w') as f:
            json.dump({
                'layer_sizes': self.layer_sizes,
                'activations': self.activations,
                'weights': weights
            }, f)
    
    def load(self, filepath: str):
        """Load model weights."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for i, layer_data in enumerate(data['weights']):
            self.layers[i].weights = np.array(layer_data['weights'])
            if layer_data['bias'] is not None:
                self.layers[i].bias = np.array(layer_data['bias'])
