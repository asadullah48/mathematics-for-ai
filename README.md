# Mathematics for AI

[![Build](https://github.com/asadullah48/mathematics-for-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/asadullah48/mathematics-for-ai/actions/workflows/ci.yml)
[![Resources](https://img.shields.io/badge/Resources-📚-blue)](resources/README.md)
[![AI-ready](https://img.shields.io/badge/AI--ready-🚀-brightgreen)](#-ai-integration-notebooks--scripts)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Empowering AI through Mathematics.** A comprehensive educational repository and Python library for understanding and implementing the mathematical foundations of Artificial Intelligence and Machine Learning.

## 🎯 Overview

This project provides:
- **Interactive implementations** of core mathematical concepts used in AI/ML
- **From-scratch implementations** of algorithms to build intuition
- **Visualizations** to understand abstract mathematical concepts
- **Jupyter notebooks** with hands-on examples and exercises
- **Production-ready code** that can be used in real applications

## 🗺️ Roadmap: Foundations → Applications → AI Integration

This repo is organized to move you through three stages, each building on
the last:

1. **Foundations** - the math itself, implemented from scratch and
   testable in isolation: `src/math_utils/` (linear algebra, calculus,
   probability, statistics) plus the theory pointers in
   [`resources/`](resources/README.md).
2. **Applications** - the classic ML algorithms these foundations build:
   `src/ai_models/` (linear/logistic regression, SVM, clustering, neural
   networks) and the worked notebooks in `notebooks/`.
3. **AI Integration** - the math wired into AI-shaped tasks you'll
   actually hit building with LLMs/agents today: `scripts/` (retrieval via
   cosine similarity - the mechanism behind RAG - and optimizer
   comparisons). See [AI Integration: Notebooks & Scripts](#-ai-integration-notebooks--scripts)
   below.

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

## 🚀 AI Integration: Notebooks & Scripts

Worked examples that take the math in `src/math_utils/` and point it at
AI-relevant problems:

| File | Demonstrates |
|---|---|
| [`notebooks/linear-algebra/01_vectors_matrices.ipynb`](notebooks/linear-algebra/01_vectors_matrices.ipynb) | Vector/matrix operations from first principles |
| [`notebooks/linear-algebra/02_eigendecomposition.ipynb`](notebooks/linear-algebra/02_eigendecomposition.ipynb) | Eigen decomposition, geometric intuition, PCA via covariance eigenvectors |
| [`notebooks/calculus/01_gradient_descent.ipynb`](notebooks/calculus/01_gradient_descent.ipynb) | Gradient descent vs. Newton's method, visualized convergence paths |
| [`scripts/rag_embeddings.py`](scripts/rag_embeddings.py) | Cosine similarity as retrieval - the math behind RAG's "search" step |
| [`scripts/optimization_routines.py`](scripts/optimization_routines.py) | Gradient descent vs. Newton's method on the Rosenbrock benchmark, runnable from the CLI |

Run any script directly, e.g. `python scripts/rag_embeddings.py`. For the
theory behind any of these, see [`resources/README.md`](resources/README.md).

## 🚀 Installation

### From PyPI (coming soon)
```bash
pip install mathematics-for-ai
```

### From Source
```bash
git clone https://github.com/asadullah48/mathematics-for-ai.git
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
jupyter notebook notebooks/basics/01_introduction.ipynb
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
├── scripts/                 # Runnable practical-application demos
│   ├── rag_embeddings.py    # Cosine similarity as retrieval (RAG)
│   └── optimization_routines.py
├── resources/                # Curated links to books/papers/courses
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

## ⚠️ Known Gaps

- **mypy type-checking is advisory, not enforced, in CI.** `src/` has
  400+ pre-existing mypy findings (mostly numpy/scipy return-type
  inference mismatches) that predate this repo's public-facing polish -
  every CI run had been failing here since the workflow was first added.
  Rather than block the Build badge on debt nobody has paid down yet
  (or blind-fix 400+ findings across code I didn't write, risking
  silently changing behavior), CI now runs mypy non-blocking so the
  signal stays visible for a dedicated type-cleanup pass. See
  `.github/workflows/ci.yml`.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/asadullah48/mathematics-for-ai.git

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

## ✍️ Author

Built by **Asadullah Shafique**.

🔗 Portfolio - Agentic AI projects and real-world applications: [asadullahshafique-devunity.vercel.app](https://asadullahshafique-devunity.vercel.app)
🐙 GitHub: [github.com/asadullah48](https://github.com/asadullah48)
