"""
Mathematics for AI - Core Mathematical Utilities

This package provides implementations of fundamental mathematical operations
used in artificial intelligence and machine learning.
"""

from .linear_algebra import LinearAlgebra, PCA
from .calculus import Calculus
from .probability import Probability
from .statistics import Statistics

__version__ = "0.2.0"
__all__ = ["LinearAlgebra", "PCA", "Calculus", "Probability", "Statistics"]
