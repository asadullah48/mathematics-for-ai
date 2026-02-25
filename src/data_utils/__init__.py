"""
Data utilities for generating and loading sample datasets.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class Dataset:
    """Dataset container."""
    X: np.ndarray
    y: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    target_names: Optional[List[str]] = None
    description: str = ""


class DataGenerator:
    """Generate synthetic datasets for testing and demonstration."""
    
    @staticmethod
    def make_regression(n_samples: int = 100, n_features: int = 1,
                       n_informative: int = 1, noise: float = 0.1,
                       bias: float = 0.0, coef: bool = False,
                       random_state: Optional[int] = None) -> Dataset:
        """
        Generate random regression data.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_informative: Number of informative features
            noise: Standard deviation of Gaussian noise
            bias: Bias term
            coef: If True, return true coefficients
            random_state: Random seed
            
        Returns:
            Dataset object
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        X = np.random.randn(n_samples, n_features)
        
        # Generate true coefficients
        true_coef = np.zeros(n_features)
        informative_idx = np.random.choice(n_features, n_informative, replace=False)
        true_coef[informative_idx] = np.random.randn(n_informative)
        
        y = X @ true_coef + bias + np.random.randn(n_samples) * noise
        
        dataset = Dataset(
            X=X, y=y,
            feature_names=[f'feature_{i}' for i in range(n_features)],
            description=f'Regression dataset with {n_samples} samples, {n_features} features'
        )
        
        if coef:
            dataset.target_names = [f'coef_{i}' for i in range(n_features)]
        
        return dataset
    
    @staticmethod
    def make_classification(n_samples: int = 100, n_features: int = 2,
                           n_classes: int = 2, n_clusters_per_class: int = 1,
                           class_sep: float = 1.0, noise: float = 0.1,
                           random_state: Optional[int] = None) -> Dataset:
        """
        Generate random classification data.
        
        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes
            n_clusters_per_class: Number of clusters per class
            class_sep: Separation between classes
            noise: Noise standard deviation
            random_state: Random seed
            
        Returns:
            Dataset object
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        samples_per_class = n_samples // n_classes
        X = []
        y = []
        
        for class_idx in range(n_classes):
            # Generate cluster centers for this class
            centers = []
            for _ in range(n_clusters_per_class):
                center = np.random.randn(n_features) * class_sep
                center += class_idx * class_sep * 2
                centers.append(center)
            
            # Generate samples around centers
            samples_per_cluster = samples_per_class // n_clusters_per_class
            for center in centers:
                cluster_samples = center + np.random.randn(samples_per_cluster, n_features) * noise
                X.append(cluster_samples)
                y.extend([class_idx] * samples_per_cluster)
        
        # Handle remainder
        remainder = n_samples - len(y)
        if remainder > 0:
            extra = centers[0] + np.random.randn(remainder, n_features) * noise
            X.append(extra)
            y.extend([0] * remainder)
        
        X = np.vstack(X)
        y = np.array(y)
        
        return Dataset(
            X=X, y=y,
            feature_names=[f'feature_{i}' for i in range(n_features)],
            target_names=[f'class_{i}' for i in range(n_classes)],
            description=f'Classification dataset with {n_samples} samples, {n_classes} classes'
        )
    
    @staticmethod
    def make_moons(n_samples: int = 100, noise: float = 0.1,
                  random_state: Optional[int] = None) -> Dataset:
        """
        Generate two interleaving half circles (moons dataset).
        
        Args:
            n_samples: Number of samples
            noise: Noise standard deviation
            random_state: Random seed
            
        Returns:
            Dataset object
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # Generate outer moon
        outer_linspace = np.linspace(0, np.pi, n_samples_out)
        X_out = np.column_stack([np.cos(outer_linspace), np.sin(outer_linspace)])
        
        # Generate inner moon
        inner_linspace = np.linspace(0, np.pi, n_samples_in)
        X_in = np.column_stack([np.cos(inner_linspace), np.sin(inner_linspace)])
        X_in -= np.array([1, 0.5])
        
        X = np.vstack([X_out, X_in])
        y = np.array([0] * n_samples_out + [1] * n_samples_in)
        
        # Add noise
        X += np.random.randn(n_samples, 2) * noise
        
        return Dataset(
            X=X, y=y,
            feature_names=['feature_0', 'feature_1'],
            target_names=['class_0', 'class_1'],
            description='Two moons dataset'
        )
    
    @staticmethod
    def make_circles(n_samples: int = 100, noise: float = 0.1,
                    factor: float = 0.5, random_state: Optional[int] = None) -> Dataset:
        """
        Generate concentric circles.
        
        Args:
            n_samples: Number of samples
            noise: Noise standard deviation
            factor: Ratio between inner and outer circle radii
            random_state: Random seed
            
        Returns:
            Dataset object
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
        
        # Outer circle
        outer_angle = np.linspace(0, 2 * np.pi, n_samples_out)
        X_out = np.column_stack([np.cos(outer_angle), np.sin(outer_angle)])
        
        # Inner circle
        inner_angle = np.linspace(0, 2 * np.pi, n_samples_in)
        X_in = np.column_stack([np.cos(inner_angle), np.sin(inner_angle)]) * factor
        
        X = np.vstack([X_out, X_in])
        y = np.array([0] * n_samples_out + [1] * n_samples_in)
        
        # Add noise
        X += np.random.randn(n_samples, 2) * noise
        
        return Dataset(
            X=X, y=y,
            feature_names=['feature_0', 'feature_1'],
            target_names=['class_0', 'class_1'],
            description='Concentric circles dataset'
        )
    
    @staticmethod
    def make_gaussian_clusters(n_clusters: int = 3, n_samples_per_cluster: int = 100,
                              n_features: int = 2, cluster_std: float = 1.0,
                              random_state: Optional[int] = None) -> Dataset:
        """
        Generate Gaussian clusters.
        
        Args:
            n_clusters: Number of clusters
            n_samples_per_cluster: Samples per cluster
            n_features: Number of features
            cluster_std: Standard deviation of clusters
            random_state: Random seed
            
        Returns:
            Dataset object
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        X = []
        y = []
        
        # Generate cluster centers
        centers = np.random.randn(n_clusters, n_features) * 5
        
        for i in range(n_clusters):
            cluster_samples = centers[i] + np.random.randn(n_samples_per_cluster, n_features) * cluster_std
            X.append(cluster_samples)
            y.extend([i] * n_samples_per_cluster)
        
        X = np.vstack(X)
        y = np.array(y)
        
        return Dataset(
            X=X, y=y,
            feature_names=[f'feature_{i}' for i in range(n_features)],
            target_names=[f'cluster_{i}' for i in range(n_clusters)],
            description=f'Gaussian clusters dataset with {n_clusters} clusters'
        )
    
    @staticmethod
    def make_spiral(n_samples: int = 100, noise: float = 0.1,
                   n_turns: int = 3, random_state: Optional[int] = None) -> Dataset:
        """
        Generate spiral dataset.
        
        Args:
            n_samples: Number of samples per spiral
            noise: Noise standard deviation
            n_turns: Number of turns in the spiral
            random_state: Random seed
            
        Returns:
            Dataset object
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n = np.sqrt(np.random.rand(n_samples, 1)) * n_turns * 2 * np.pi
        X1 = np.column_stack([n * np.cos(n), n * np.sin(n)])
        X2 = np.column_stack([n * np.cos(n + np.pi), n * np.sin(n + np.pi)])
        
        X = np.vstack([X1, X2])
        y = np.array([0] * n_samples + [1] * n_samples)
        
        # Add noise
        X += np.random.randn(len(X), 2) * noise
        
        # Shift second spiral
        X[n_samples:] += np.array([1, 1])
        
        return Dataset(
            X=X, y=y,
            feature_names=['feature_0', 'feature_1'],
            target_names=['class_0', 'class_1'],
            description='Spiral dataset'
        )


def load_sample_dataset(name: str) -> Dataset:
    """
    Load a sample dataset by name.
    
    Args:
        name: Dataset name ('regression', 'classification', 'moons', 'circles', 'spiral')
        
    Returns:
        Dataset object
    """
    generator = DataGenerator()
    
    if name == 'regression':
        return generator.make_regression(n_samples=200, n_features=2, noise=0.5)
    elif name == 'classification':
        return generator.make_classification(n_samples=200, n_features=2, n_classes=2)
    elif name == 'moons':
        return generator.make_moons(n_samples=200, noise=0.1)
    elif name == 'circles':
        return generator.make_circles(n_samples=200, noise=0.05)
    elif name == 'spiral':
        return generator.make_spiral(n_samples=200, noise=0.1)
    elif name == 'clusters':
        return generator.make_gaussian_clusters(n_clusters=4, n_samples_per_cluster=50)
    else:
        raise ValueError(f"Unknown dataset: {name}")
