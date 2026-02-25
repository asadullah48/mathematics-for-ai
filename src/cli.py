#!/usr/bin/env python3
"""
Command Line Interface for Mathematics for AI

Usage:
    math-ai demo [options]
    math-ai test [options]
    math-ai generate-data [options]
    math-ai --help
    math-ai --version

Commands:
    demo            Run a demonstration
    test            Run tests
    generate-data   Generate sample datasets

Options:
    -h, --help              Show this help message
    --version               Show version
    -t, --topic TOPIC       Topic for demo (linear-algebra, calculus, probability, ml)
    -n, --samples N         Number of samples [default: 100]
    -o, --output PATH       Output path for generated data
    -v, --verbose           Verbose output
"""

import sys
import argparse
import numpy as np
from pathlib import Path


def demo_linear_algebra(verbose: bool = False):
    """Demonstrate linear algebra capabilities."""
    from math_utils import LinearAlgebra
    
    print("=" * 60)
    print("LINEAR ALGEBRA DEMONSTRATION")
    print("=" * 60)
    
    # Matrix operations
    A = np.array([[4, 2], [1, 3]], dtype=float)
    print(f"\nMatrix A:\n{A}")
    
    # Determinant
    det = LinearAlgebra.determinant(A)
    print(f"\nDeterminant: {det:.4f}")
    
    # Eigen decomposition
    eigenvalues, eigenvectors = LinearAlgebra.eigendecomposition(A)
    print(f"\nEigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # SVD
    U, S, Vt = LinearAlgebra.svd(A)
    print(f"\nSingular Values: {S}")
    
    # Solve linear system
    b = np.array([1, 2], dtype=float)
    x = LinearAlgebra.solve(A, b)
    print(f"\nSolving Ax = b where b = {b}")
    print(f"Solution: x = {x}")
    print(f"Verification: Ax = {A @ x}")


def demo_calculus(verbose: bool = False):
    """Demonstrate calculus capabilities."""
    from math_utils import Calculus
    
    print("=" * 60)
    print("CALCULUS DEMONSTRATION")
    print("=" * 60)
    
    # Derivative
    f = lambda x: x ** 3 - 2 * x + 1
    x = 2.0
    deriv = Calculus.numerical_derivative(f, x)
    print(f"\nf(x) = x³ - 2x + 1")
    print(f"f'({x}) = {deriv:.4f} (exact: {3*x**2 - 2})")
    
    # Gradient
    g = lambda x: x[0] ** 2 + x[1] ** 2
    point = np.array([3.0, 4.0])
    grad = Calculus.numerical_gradient(g, point)
    print(f"\ng(x,y) = x² + y²")
    print(f"∇g(3,4) = {grad} (exact: [6, 8])")
    
    # Integration
    f = lambda x: np.sin(x)
    integral = Calculus.simpson(f, 0, np.pi, n=100)
    print(f"\n∫sin(x)dx from 0 to π = {integral:.6f} (exact: 2)")
    
    # Root finding
    f = lambda x: x ** 2 - 2
    root = Calculus.newton_raphson(f, 1.5)
    print(f"\nRoot of x² - 2 = 0: {root:.10f} (exact: {np.sqrt(2):.10f})")


def demo_probability(verbose: bool = False):
    """Demonstrate probability capabilities."""
    from math_utils import Probability
    
    print("=" * 60)
    print("PROBABILITY DEMONSTRATION")
    print("=" * 60)
    
    # Gaussian distribution
    x = np.linspace(-3, 3, 10)
    pdf = Probability.gaussian_pdf(x, mu=0, sigma=1)
    print(f"\nGaussian PDF N(0,1):")
    print(f"x: {x[:5]}...")
    print(f"PDF: {pdf[:5]}...")
    
    # Bayes' theorem
    p_disease = 0.01
    p_positive_given_disease = 0.99
    p_positive_given_no_disease = 0.05
    
    p_positive = p_positive_given_disease * p_disease + p_positive_given_no_disease * (1 - p_disease)
    p_disease_given_positive = Probability.bayes_theorem(p_disease, p_positive_given_disease, p_positive)
    
    print(f"\nBayes' Theorem Example:")
    print(f"P(Disease) = {p_disease}")
    print(f"P(+|Disease) = {p_positive_given_disease}")
    print(f"P(+|No Disease) = {p_positive_given_no_disease}")
    print(f"P(Disease|+) = {p_disease_given_positive:.4f}")
    
    # MLE
    data = np.random.randn(1000) * 2 + 5
    mu_mle, sigma_mle = Probability.mle_gaussian(data)
    print(f"\nMLE for Gaussian:")
    print(f"True: μ=5, σ=2")
    print(f"Estimated: μ={mu_mle:.4f}, σ={sigma_mle:.4f}")


def demo_ml(verbose: bool = False):
    """Demonstrate machine learning capabilities."""
    from ai_models import LinearRegression, LogisticRegression, NeuralNetwork
    from data_utils import DataGenerator
    
    print("=" * 60)
    print("MACHINE LEARNING DEMONSTRATION")
    print("=" * 60)
    
    # Linear Regression
    print("\n--- Linear Regression ---")
    generator = DataGenerator()
    dataset = generator.make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
    
    model = LinearRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(dataset.X, dataset.y)
    
    print(f"True coefficients: [2.0, 1.0] (approximately)")
    print(f"Learned coefficients: {model.weights}")
    print(f"R² Score: {model.score(dataset.X, dataset.y):.4f}")
    
    # Logistic Regression
    print("\n--- Logistic Regression ---")
    dataset = generator.make_classification(n_samples=200, n_features=2, random_state=42)
    
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(dataset.X, dataset.y)
    
    print(f"Training Accuracy: {model.score(dataset.X, dataset.y):.4f}")
    
    # Neural Network
    print("\n--- Neural Network ---")
    dataset = generator.make_classification(n_samples=500, n_features=10, random_state=42)
    
    nn = NeuralNetwork(
        layer_sizes=[10, 32, 16, 2],
        activations=['relu', 'relu', 'softmax'],
        learning_rate=0.001,
        optimizer='adam'
    )
    
    nn.fit(dataset.X, dataset.y, epochs=100, batch_size=32, verbose=False)
    print(f"Training Accuracy: {nn.score(dataset.X, dataset.y):.4f}")


def generate_data(args):
    """Generate sample datasets."""
    from data_utils import DataGenerator, load_sample_dataset
    
    generator = DataGenerator()
    
    print("Generating sample datasets...")
    
    # Generate and save datasets
    datasets = {
        'regression': generator.make_regression(n_samples=args.samples, random_state=42),
        'classification': generator.make_classification(n_samples=args.samples, random_state=42),
        'moons': generator.make_moons(n_samples=args.samples, random_state=42),
        'circles': generator.make_circles(n_samples=args.samples, random_state=42),
        'clusters': generator.make_gaussian_clusters(n_samples_per_cluster=args.samples//4, random_state=42)
    }
    
    output_dir = Path(args.output) if args.output else Path('data/generated')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, dataset in datasets.items():
        np.savez(output_dir / f'{name}.npz', X=dataset.X, y=dataset.y)
        print(f"  ✓ {name}: X={dataset.X.shape}, y={dataset.y.shape}")
    
    print(f"\nDatasets saved to: {output_dir}")


def run_tests():
    """Run test suite."""
    import subprocess
    
    print("Running tests...")
    result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], cwd=Path(__file__).parent.parent)
    sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='Mathematics for AI - Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstrations')
    demo_parser.add_argument('-t', '--topic', choices=['linear-algebra', 'calculus', 'probability', 'ml', 'all'],
                            default='all', help='Topic to demonstrate')
    demo_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate sample datasets')
    gen_parser.add_argument('-n', '--samples', type=int, default=100, help='Number of samples')
    gen_parser.add_argument('-o', '--output', type=str, default='data/generated', help='Output directory')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        if args.topic == 'linear-algebra' or args.topic == 'all':
            demo_linear_algebra(args.verbose)
        if args.topic == 'calculus' or args.topic == 'all':
            demo_calculus(args.verbose)
        if args.topic == 'probability' or args.topic == 'all':
            demo_probability(args.verbose)
        if args.topic == 'ml' or args.topic == 'all':
            demo_ml(args.verbose)
    
    elif args.command == 'test':
        run_tests()
    
    elif args.command == 'generate-data':
        generate_data(args)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
