"""Tests for scripts/optimization_routines.py."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from optimization_routines import compare, rosenbrock, run_gradient_descent, run_newton  # noqa: E402


def test_rosenbrock_minimum_is_zero_at_one_one():
    assert rosenbrock(np.array([1.0, 1.0])) == 0.0


def test_rosenbrock_is_positive_away_from_minimum():
    assert rosenbrock(np.array([0.0, 0.0])) > 0.0


def test_gradient_descent_reduces_loss_on_a_bowl():
    def bowl(x):
        return x[0] ** 2 + x[1] ** 2

    path = run_gradient_descent(bowl, x0=[3.0, 4.0], learning_rate=0.1, n_steps=50)

    assert bowl(path[-1]) < bowl(path[0])
    assert bowl(path[-1]) < 1e-3


def test_newton_converges_to_rosenbrock_minimum_in_few_steps():
    path = run_newton(rosenbrock, x0=[1.2, 1.2], n_steps=15)

    assert np.allclose(path[-1], [1.0, 1.0], atol=1e-3)


def test_compare_returns_both_methods_with_lower_final_loss_than_start():
    x0 = [-1.5, 2.0]
    results = compare(rosenbrock, x0)

    assert set(results) == {"gradient_descent", "newton"}
    start_loss = rosenbrock(np.array(x0))
    for method_result in results.values():
        assert method_result["final_loss"] < start_loss
        assert method_result["steps"] > 0
