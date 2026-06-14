"""
AI Models Module

Machine learning algorithms implemented from scratch for educational purposes.
Includes linear models, neural networks, clustering, and more.
"""

from .linear_models import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from .neural_networks import (
    NeuralNetwork, Layer, Activation,
    LSTM,
    ScaledDotProductAttention, MultiHeadAttention, TransformerBlock,
)
from .clustering import KMeans, GaussianMixtureModel, DBSCAN
from .svm import SupportVectorMachine

__version__ = "0.2.0"
__all__ = [
    "LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet",
    "NeuralNetwork", "Layer", "Activation",
    "LSTM",
    "ScaledDotProductAttention", "MultiHeadAttention", "TransformerBlock",
    "KMeans", "GaussianMixtureModel", "DBSCAN",
    "SupportVectorMachine",
]
