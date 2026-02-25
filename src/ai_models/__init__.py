"""
AI Models Module

Machine learning algorithms implemented from scratch for educational purposes.
Includes linear models, neural networks, clustering, and more.
"""

from .linear_models import LinearRegression, LogisticRegression, Ridge, Lasso
from .neural_networks import NeuralNetwork, Layer, Activation
from .clustering import KMeans, GaussianMixtureModel, DBSCAN
from .svm import SupportVectorMachine

__version__ = "0.1.0"
__all__ = [
    "LinearRegression", "LogisticRegression", "Ridge", "Lasso",
    "NeuralNetwork", "Layer", "Activation",
    "KMeans", "GaussianMixtureModel", "DBSCAN",
    "SupportVectorMachine"
]
