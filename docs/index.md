# Mathematics for AI Documentation

## Welcome

Welcome to the Mathematics for AI documentation. This comprehensive library provides educational implementations of mathematical concepts essential for understanding artificial intelligence and machine learning.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Tutorials](#tutorials)
5. [Theory](#theory)

## Installation {#installation}

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

## Quick Start {#quick-start}

```python
from math_utils import LinearAlgebra, Calculus, Probability, Statistics
from ai_models import LinearRegression, NeuralNetwork
from visualization import Plot2D

# Linear Algebra
import numpy as np
A = np.array([[1, 2], [3, 4]])
eigenvalues, eigenvectors = LinearAlgebra.eigendecomposition(A)

# Calculus
f = lambda x: x ** 2
gradient = Calculus.numerical_gradient(f, np.array([3.0]))

# Probability
data = np.random.randn(1000)
mean, std = Probability.mle_gaussian(data)

# Train a model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
```

## API Reference {#api-reference}

### Math Utils

#### LinearAlgebra

Core linear algebra operations:

- `dot_product(v1, v2)` - Vector dot product
- `cross_product(v1, v2)` - Vector cross product (3D)
- `matrix_multiply(A, B)` - Matrix multiplication
- `determinant(A)` - Matrix determinant
- `trace(A)` - Matrix trace
- `rank(A)` - Matrix rank
- `lu_decomposition(A)` - LU decomposition with pivoting
- `qr_decomposition(A)` - QR decomposition
- `cholesky_decomposition(A)` - Cholesky decomposition
- `svd(A)` - Singular Value Decomposition
- `eigendecomposition(A)` - Eigenvalue decomposition
- `solve(A, b)` - Solve linear system Ax = b
- `least_squares(A, b)` - Least squares solution
- `norm(v, ord)` - Vector norm
- `cosine_similarity(v1, v2)` - Cosine similarity

#### Calculus

Differential and integral calculus:

- `numerical_derivative(f, x)` - Numerical derivative
- `numerical_gradient(f, x)` - Gradient computation
- `jacobian(f, x)` - Jacobian matrix
- `hessian(f, x)` - Hessian matrix
- `trapezoidal(f, a, b)` - Trapezoidal integration
- `simpson(f, a, b)` - Simpson's rule integration
- `bisection(f, a, b)` - Bisection root finding
- `newton_raphson(f, x0)` - Newton's method
- `gradient_descent_step(f, x)` - Gradient descent step

#### Probability

Probability theory and distributions:

- `gaussian_pdf(x, mu, sigma)` - Gaussian PDF
- `binomial_pmf(k, n, p)` - Binomial PMF
- `poisson_pmf(k, lambda)` - Poisson PMF
- `bayes_theorem(p_a, p_b_given_a, p_b)` - Bayes' theorem
- `mle_gaussian(data)` - MLE for Gaussian
- `entropy(probabilities)` - Shannon entropy
- `kl_divergence(p, q)` - KL divergence

#### Statistics

Statistical methods:

- `summary(data)` - Comprehensive summary statistics
- `confidence_interval_mean(data)` - Confidence interval for mean
- `t_test_one_sample(sample, pop_mean)` - One-sample t-test
- `anova_one_way(*groups)` - One-way ANOVA
- `pearson_correlation(x, y)` - Pearson correlation
- `chi_squared_test(observed, expected)` - Chi-squared test

### AI Models

#### Linear Models

- `LinearRegression` - Linear regression with gradient descent
- `LogisticRegression` - Logistic regression for classification
- `Ridge` - Ridge regression (L2 regularization)
- `Lasso` - Lasso regression (L1 regularization)
- `ElasticNet` - Elastic Net (L1 + L2)

#### Neural Networks

- `NeuralNetwork` - Feedforward neural network
- `Dense` - Fully connected layer
- `Conv2D` - 2D convolutional layer
- `RNN` - Recurrent neural network layer
- `Activation` - Activation functions

#### Clustering

- `KMeans` - K-Means clustering
- `GaussianMixtureModel` - GMM with EM algorithm
- `DBSCAN` - Density-based clustering

#### Support Vector Machines

- `SupportVectorMachine` - SVM with various kernels
- `SVR` - Support Vector Regression

### Visualization

#### Plot2D

2D plotting utilities:

- `plot_function(f)` - Plot 1D function
- `plot_scatter(X, y)` - Scatter plot
- `plot_decision_boundary(classifier, X, y)` - Decision boundary
- `plot_contour(f)` - Contour plot
- `plot_gradient_field(f)` - Gradient field
- `plot_gradient_descent(f, gradient, x0)` - GD visualization

#### Plot3D

3D plotting utilities:

- `plot_surface(f)` - 3D surface plot
- `plot_scatter_3d(X)` - 3D scatter plot
- `plot_loss_landscape(loss_fn, center)` - Loss landscape

#### Animation

Animated visualizations:

- `gradient_descent_animation(f, gradient, x0)` - GD animation
- `kmeans_animation(X, n_clusters)` - K-Means animation
- `pca_animation(X)` - PCA visualization

## Tutorials {#tutorials}

See the `notebooks/` directory for interactive tutorials:

- `notebooks/basics/01_introduction.ipynb` - Getting started
- `notebooks/linear-algebra/01_vectors_matrices.ipynb` - Linear algebra
- `notebooks/calculus/01_derivatives.ipynb` - Calculus basics
- `notebooks/probability/01_distributions.ipynb` - Probability distributions

## Theory {#theory}

### Linear Algebra

Linear algebra is the study of vectors, matrices, and linear transformations. It forms the foundation for:

- Neural network computations
- Dimensionality reduction (PCA, SVD)
- Word embeddings in NLP
- Computer graphics transformations

### Calculus

Calculus deals with rates of change and accumulation. Key concepts:

- **Derivatives**: Rate of change of a function
- **Gradients**: Direction of steepest ascent
- **Optimization**: Finding minima/maxima
- **Integration**: Computing areas and volumes

### Probability

Probability theory quantifies uncertainty:

- **Distributions**: Describe random variables
- **Bayes' Theorem**: Update beliefs with evidence
- **Expectation**: Average value of random variable
- **Variance**: Spread of distribution

### Statistics

Statistics involves data analysis and inference:

- **Descriptive**: Summarize data
- **Inferential**: Draw conclusions from samples
- **Hypothesis Testing**: Test claims about populations
- **Regression**: Model relationships between variables

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

MIT License - see LICENSE file for details.
