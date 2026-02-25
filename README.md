# Mathematics for AI

A comprehensive educational repository and Python library for understanding and implementing the mathematical foundations of Artificial Intelligence and Machine Learning.

## 🎯 Overview

This project provides:
- **Interactive implementations** of core mathematical concepts used in AI/ML
- **From-scratch implementations** of algorithms to build intuition
- **Visualizations** to understand abstract mathematical concepts
- **Jupyter notebooks** with hands-on examples and exercises
- **Production-ready code** that can be used in real applications

## 📚 Topics Covered

### Linear Algebra
- Vectors, matrices, and tensors operations
- Matrix decompositions (LU, QR, Cholesky, SVD, Eigendecomposition)
- Vector spaces, basis, and transformations
- Eigenvalues and eigenvectors
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)

### Calculus
- Derivatives and gradients
- Partial derivatives and Jacobians
- Chain rule and backpropagation
- Taylor series expansions
- Optimization techniques (Gradient Descent, Adam, RMSprop)
- Constrained optimization (Lagrange multipliers)

### Probability & Statistics
- Probability distributions (discrete and continuous)
- Bayes' theorem and applications
- Expectation, variance, and covariance
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP)
- Hypothesis testing
- Markov chains and Monte Carlo methods

### Optimization
- Convex optimization
- Gradient-based methods
- Second-order methods (Newton, BFGS)
- Constrained optimization
- Linear programming
- Dynamic programming

### Machine Learning Algorithms (From Scratch)
- Linear Regression (with regularization)
- Logistic Regression
- Support Vector Machines
- K-Means Clustering
- Gaussian Mixture Models
- Neural Networks (Feedforward, CNN, RNN)
- Attention mechanisms and Transformers

## 🚀 Installation

### From PyPI (coming soon)
```bash
pip install mathematics-for-ai
```

### From Source
```bash
git clone https://github.com/yourusername/mathematics-for-ai.git
cd mathematics-for-ai
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

## 📖 Quick Start

### Basic Usage

```python
from math_utils import LinearAlgebra, Calculus, Probability
from ai_models import LinearRegression, NeuralNetwork
from visualization import plot_function, plot_decision_boundary

# Linear Algebra Example
import numpy as np
A = np.array([[1, 2], [3, 4]])
eigenvalues, eigenvectors = LinearAlgebra.eigendecomposition(A)

# Calculus Example
def f(x):
    return x**2 + 2*x + 1

gradient = Calculus.numerical_gradient(f, np.array([3.0]))

# Probability Example
mean, std = Probability.fit_gaussian(data)
likelihood = Probability.gaussian_pdf(data, mean, std)

# Train a model from scratch
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Using Jupyter Notebooks

Explore the `notebooks/` directory for interactive tutorials:

```bash
jupyter notebook notebooks/basics/introduction.ipynb
```

## 📁 Project Structure

```
mathematics-for-ai/
├── src/
│   ├── math_utils/          # Core mathematical operations
│   │   ├── linear_algebra.py
│   │   ├── calculus.py
│   │   ├── probability.py
│   │   └── statistics.py
│   ├── ai_models/           # ML algorithms from scratch
│   │   ├── linear_models.py
│   │   ├── neural_networks.py
│   │   ├── clustering.py
│   │   └── svm.py
│   └── visualization/       # Mathematical visualizations
│       ├── plots_2d.py
│       ├── plots_3d.py
│       └── animations.py
├── notebooks/               # Interactive tutorials
│   ├── basics/
│   ├── linear-algebra/
│   ├── calculus/
│   ├── probability/
│   └── applications/
├── tests/                   # Test suite
├── docs/                    # Documentation
│   ├── theory/              # Mathematical theory
│   ├── examples/            # Code examples
│   └── tutorials/           # Step-by-step guides
├── data/                    # Sample datasets
└── config/                  # Configuration files
```

## 🎓 Learning Path

### Beginner
1. Start with `notebooks/basics/introduction.ipynb`
2. Learn linear algebra fundamentals
3. Understand basic calculus concepts
4. Explore probability basics

### Intermediate
1. Study optimization algorithms
2. Implement ML algorithms from scratch
3. Work through statistical inference
4. Build neural networks from scratch

### Advanced
1. Deep dive into matrix decompositions
2. Advanced optimization techniques
3. Attention mechanisms and transformers
4. Research-level applications

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_linear_algebra.py
```

## 📊 Features

### Math Utils Module
- ✅ Vector and matrix operations
- ✅ Matrix decompositions (LU, QR, SVD, Cholesky)
- ✅ Numerical differentiation and integration
- ✅ Probability distributions (20+ distributions)
- ✅ Statistical tests and measures

### AI Models Module
- ✅ Linear/Logistic Regression with regularization
- ✅ Neural Networks (custom autograd engine)
- ✅ Support Vector Machines
- ✅ Clustering algorithms (K-Means, GMM, DBSCAN)
- ✅ Decision Trees and Random Forests

### Visualization Module
- ✅ 2D/3D function plotting
- ✅ Decision boundary visualization
- ✅ Gradient descent animation
- ✅ Interactive dashboards (Plotly)
- ✅ Mathematical concept illustrations

## 🔧 Configuration

Create a `config.yaml` file:

```yaml
random_seed: 42
precision: float64
default_learning_rate: 0.001
max_iterations: 1000
tolerance: 1e-6
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/mathematics-for-ai.git

# Install development dependencies
pip install -e ".[dev]"

# Run pre-commit hooks
pre-commit install

# Run tests before committing
pytest tests/
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Linear Algebra by Gilbert Strang
- Pattern Recognition and Machine Learning by Christopher Bishop
- Deep Learning by Ian Goodfellow
- Mathematics for Machine Learning by Deisenroth, Faisal, and Ong

## 📧 Contact

For questions and suggestions, please open an issue or contact the maintainers.
