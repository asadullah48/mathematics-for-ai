"""Practical application: comparing optimization routines on a shared test
function - the runnable-script counterpart to
`notebooks/calculus/01_gradient_descent.ipynb`.

Uses this repo's own `Calculus.gradient_descent_step` and
`Calculus.newton_step` (`src/math_utils/calculus.py`) - no new math here,
just wiring the existing library into a small, reusable comparison you can
run from the command line or import into other scripts/notebooks.

Run directly:
    python scripts/optimization_routines.py
"""
from __future__ import annotations  # list[float]/dict[...] annotations need this on Python 3.8

import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from math_utils.calculus import Calculus  # noqa: E402

LossFn = Callable[[np.ndarray], float]


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """Classic optimizer benchmark: a curved, narrow valley. Much harder
    to minimize than a plain bowl - gradient descent tends to zig-zag
    along the valley floor, which is exactly why this function is the
    standard stress test for optimizers."""
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def run_gradient_descent(
    f: LossFn, x0: list[float], learning_rate: float = 0.001, n_steps: int = 500
) -> np.ndarray:
    """Trace the path gradient descent takes from x0 toward a minimum of f."""
    path = [np.array(x0, dtype=float)]
    x = np.array(x0, dtype=float)
    for _ in range(n_steps):
        x = Calculus.gradient_descent_step(f, x, learning_rate=learning_rate)
        path.append(x.copy())
    return np.array(path)


def run_newton(f: LossFn, x0: list[float], n_steps: int = 20) -> np.ndarray:
    """Trace the path Newton's method takes from x0 toward a minimum of f."""
    path = [np.array(x0, dtype=float)]
    x = np.array(x0, dtype=float)
    for _ in range(n_steps):
        x = Calculus.newton_step(f, x)
        path.append(x.copy())
    return np.array(path)


def compare(f: LossFn, x0: list[float]) -> dict[str, dict]:
    """Run both optimizers from the same start point and summarize how far
    each got. Newton's method needs a positive-definite Hessian to behave
    well; on a function like Rosenbrock it can misbehave far from the
    minimum, which the comparison intentionally exposes rather than hides.
    """
    gd_path = run_gradient_descent(f, x0)
    newton_path = run_newton(f, x0)
    return {
        "gradient_descent": {"final_x": gd_path[-1], "final_loss": f(gd_path[-1]), "steps": len(gd_path) - 1},
        "newton": {"final_x": newton_path[-1], "final_loss": f(newton_path[-1]), "steps": len(newton_path) - 1},
    }


def main() -> None:
    x0 = [-1.5, 2.0]
    results = compare(rosenbrock, x0)

    print(f"Minimizing Rosenbrock's function from x0={x0}")
    print("(true minimum is at (1, 1), loss=0)\n")
    for name, result in results.items():
        print(
            f"  {name:>17s}: {result['steps']:4d} steps -> "
            f"x={np.round(result['final_x'], 4)}, loss={result['final_loss']:.6g}"
        )


if __name__ == "__main__":
    main()
