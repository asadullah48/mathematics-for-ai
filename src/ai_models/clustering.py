"""
Clustering Module

Unsupervised learning algorithms implemented from scratch:
- K-Means Clustering
- Gaussian Mixture Models (GMM)
- DBSCAN
- Hierarchical Clustering
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist


class BaseClustering(ABC):
    """Base class for clustering algorithms."""
    
    @abstractmethod
    def fit(self, X: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class KMeans(BaseClustering):
    """
    K-Means Clustering algorithm.
    
    Parameters:
        n_clusters: Number of clusters
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        init: Initialization method ('kmeans++' or 'random')
        random_state: Random seed
        n_init: Number of times to run with different initializations
    """
    
    def __init__(self, n_clusters: int = 8, max_iter: int = 300,
                 tol: float = 1e-6, init: str = 'kmeans++',
                 random_state: Optional[int] = None, n_init: int = 10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.n_init = n_init
        
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter = 0
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids using k-means++ or random."""
        if self.init == 'random':
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            return X[indices]
        
        elif self.init == 'kmeans++':
            centroids = []
            
            # Choose first centroid randomly
            first_idx = np.random.randint(len(X))
            centroids.append(X[first_idx])
            
            # Choose remaining centroids
            for _ in range(1, self.n_clusters):
                # Compute distances to nearest centroid
                distances = np.array([
                    min(np.linalg.norm(x - c) ** 2 for c in centroids)
                    for x in X
                ])
                
                # Choose next centroid with probability proportional to distance squared
                probabilities = distances / distances.sum()
                next_idx = np.random.choice(len(X), p=probabilities)
                centroids.append(X[next_idx])
            
            return np.array(centroids)
        
        else:
            raise ValueError(f"Unknown init method: {self.init}")
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid."""
        distances = cdist(X, centroids, metric='euclidean')
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Update centroids as mean of assigned points."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                centroids[k] = X[mask].mean(axis=0)
            else:
                # If no points assigned, reinitialize randomly
                centroids[k] = X[np.random.randint(len(X))]
        
        return centroids
    
    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, 
                        centroids: np.ndarray) -> float:
        """Compute within-cluster sum of squares."""
        inertia = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                inertia += np.sum((X[mask] - centroids[k]) ** 2)
        return inertia
    
    def _fit_single(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Run single k-means fit."""
        centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, centroids)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Check convergence
            if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < self.tol:
                break
            
            centroids = new_centroids
        
        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration + 1
    
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-Means clustering.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        
        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._fit_single(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                self.n_iter = n_iter
        
        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        return self._assign_clusters(X, self.centroids)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform X to cluster distance space."""
        return cdist(X, self.centroids, metric='euclidean')
    
    def score(self, X: np.ndarray) -> float:
        """Compute negative inertia (for compatibility with sklearn)."""
        labels = self.predict(X)
        return -self._compute_inertia(X, labels, self.centroids)


class GaussianMixtureModel(BaseClustering):
    """
    Gaussian Mixture Model using EM algorithm.
    
    Parameters:
        n_components: Number of mixture components
        max_iter: Maximum number of iterations
        tol: Convergence tolerance
        covariance_type: 'full', 'tied', 'diag', or 'spherical'
        random_state: Random seed
    """
    
    def __init__(self, n_components: int = 8, max_iter: int = 100,
                 tol: float = 1e-6, covariance_type: str = 'full',
                 random_state: Optional[int] = None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.covariance_type = covariance_type
        self.random_state = random_state
        
        self.weights = None  # Mixing coefficients
        self.means = None  # Component means
        self.covariances = None  # Component covariances
        self.converged = False
        self.n_iter = 0
        self.log_likelihood = None
    
    def _initialize(self, X: np.ndarray):
        """Initialize parameters."""
        n_samples, n_features = X.shape
        
        # Initialize weights uniformly
        self.weights = np.ones(self.n_components) / self.n_components
        
        # Initialize means using k-means++
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
        kmeans.fit(X)
        self.means = kmeans.centroids
        
        # Initialize covariances
        if self.covariance_type == 'full':
            self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)])
        elif self.covariance_type == 'tied':
            self.covariances = np.cov(X.T)
        elif self.covariance_type == 'diag':
            self.covariances = np.array([np.var(X, axis=0) for _ in range(self.n_components)])
        elif self.covariance_type == 'spherical':
            self.covariances = np.array([np.var(X) * np.eye(n_features) 
                                        for _ in range(self.n_components)])
    
    def _multivariate_gaussian_pdf(self, X: np.ndarray, mean: np.ndarray,
                                   cov: np.ndarray) -> np.ndarray:
        """Compute multivariate Gaussian PDF."""
        n_features = len(mean)
        
        if self.covariance_type == 'spherical':
            var = cov[0, 0]
            return (1 / np.sqrt(2 * np.pi * var) ** n_features) * \
                   np.exp(-np.sum((X - mean) ** 2, axis=1) / (2 * var))
        
        # Add regularization for numerical stability
        cov_reg = cov + 1e-6 * np.eye(n_features)
        
        try:
            cov_inv = np.linalg.inv(cov_reg)
            cov_det = np.linalg.det(cov_reg)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_reg)
            cov_det = np.prod(np.linalg.eigvalsh(cov_reg))
        
        if cov_det <= 0:
            cov_det = 1e-10
        
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        
        return (1 / np.sqrt((2 * np.pi) ** n_features * cov_det)) * np.exp(exponent)
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """E-step: compute responsibilities."""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            if self.covariance_type == 'tied':
                responsibilities[:, k] = self.weights[k] * \
                    self._multivariate_gaussian_pdf(X, self.means[k], self.covariances)
            else:
                responsibilities[:, k] = self.weights[k] * \
                    self._multivariate_gaussian_pdf(X, self.means[k], self.covariances[k])
        
        # Normalize
        total = responsibilities.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1e-10)  # Avoid division by zero
        responsibilities /= total
        
        return responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """M-step: update parameters."""
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Effective number of points assigned to each component
        Nk = responsibilities.sum(axis=0)
        Nk = np.maximum(Nk, 1e-10)  # Avoid division by zero
        
        # Update weights
        self.weights = Nk / n_samples
        
        # Update means
        self.means = (responsibilities.T @ X) / Nk[:, np.newaxis]
        
        # Update covariances
        if self.covariance_type == 'full':
            self.covariances = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = (responsibilities[:, k][:, np.newaxis] * diff).T @ diff / Nk[k]
        
        elif self.covariance_type == 'tied':
            self.covariances = np.zeros((n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances += (responsibilities[:, k][:, np.newaxis] * diff).T @ diff
            self.covariances /= n_samples
        
        elif self.covariance_type == 'diag':
            self.covariances = np.zeros((self.n_components, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                self.covariances[k] = (responsibilities[:, k] * diff ** 2).sum(axis=0) / Nk[k]
        
        elif self.covariance_type == 'spherical':
            self.covariances = np.zeros((self.n_components, n_features, n_features))
            for k in range(self.n_components):
                diff = X - self.means[k]
                var = (responsibilities[:, k] * np.sum(diff ** 2, axis=1)).sum() / (Nk[k] * n_features)
                self.covariances[k] = var * np.eye(n_features)
    
    def _compute_log_likelihood(self, X: np.ndarray) -> float:
        """Compute log-likelihood of the data."""
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)
        
        for k in range(self.n_components):
            if self.covariance_type == 'tied':
                likelihood += self.weights[k] * \
                    self._multivariate_gaussian_pdf(X, self.means[k], self.covariances)
            else:
                likelihood += self.weights[k] * \
                    self._multivariate_gaussian_pdf(X, self.means[k], self.covariances[k])
        
        return np.sum(np.log(np.maximum(likelihood, 1e-10)))
    
    def fit(self, X: np.ndarray) -> 'GaussianMixtureModel':
        """
        Fit GMM using EM algorithm.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            Self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        self._initialize(X)
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self._e_step(X)
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check convergence
            self.log_likelihood = self._compute_log_likelihood(X)
            if abs(self.log_likelihood - prev_log_likelihood) < self.tol:
                self.converged = True
                self.n_iter = iteration + 1
                break
            
            prev_log_likelihood = self.log_likelihood
            self.n_iter = iteration + 1
        
        # Assign labels
        self.labels = np.argmax(self._e_step(X), axis=1)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels."""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignment probabilities."""
        return self._e_step(X)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels
    
    def score(self, X: np.ndarray) -> float:
        """Compute log-likelihood."""
        return self._compute_log_likelihood(X)
    
    def bic(self, X: np.ndarray) -> float:
        """Compute Bayesian Information Criterion."""
        n_samples, n_features = X.shape
        n_params = self._count_parameters(n_features)
        return np.log(n_samples) * n_params - 2 * self.log_likelihood
    
    def aic(self, X: np.ndarray) -> float:
        """Compute Akaike Information Criterion."""
        n_params = self._count_parameters(X.shape[1])
        return 2 * n_params - 2 * self.log_likelihood
    
    def _count_parameters(self, n_features: int) -> int:
        """Count number of free parameters."""
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        else:  # spherical
            cov_params = self.n_components
        
        mean_params = self.n_components * n_features
        weight_params = self.n_components - 1
        
        return int(cov_params + mean_params + weight_params)


class DBSCAN(BaseClustering):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    
    Parameters:
        eps: Maximum distance between two samples to be considered neighbors
        min_samples: Minimum number of samples in a neighborhood to be a core point
        metric: Distance metric
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        self.labels = None
        self.core_sample_indices = None
        self.n_clusters = 0
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        return cdist(X, X, metric=self.metric)
    
    def _get_neighbors(self, distances: np.ndarray, point_idx: int) -> np.ndarray:
        """Get indices of neighbors within eps distance."""
        return np.where(distances[point_idx] <= self.eps)[0]
    
    def _expand_cluster(self, X: np.ndarray, distances: np.ndarray,
                       point_idx: int, neighbors: np.ndarray,
                       cluster_id: int, labels: np.ndarray,
                       visited: np.ndarray) -> np.ndarray:
        """Expand cluster from core point."""
        labels[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if not visited[neighbor_idx]:
                visited[neighbor_idx] = True
                neighbor_neighbors = self._get_neighbors(distances, neighbor_idx)
                
                if len(neighbor_neighbors) >= self.min_samples:
                    # This is a core point, add its neighbors
                    neighbors = np.concatenate([neighbors, neighbor_neighbors])
                    neighbors = np.unique(neighbors)
            
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            
            i += 1
        
        return labels
    
    def fit(self, X: np.ndarray) -> 'DBSCAN':
        """
        Fit DBSCAN clustering.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            Self
        """
        n_samples = X.shape[0]
        
        # Compute distance matrix
        distances = self._compute_distances(X)
        
        # Initialize labels (-1 means noise)
        labels = np.full(n_samples, -1)
        visited = np.zeros(n_samples, dtype=bool)
        
        # Find core points
        core_samples = []
        for i in range(n_samples):
            neighbors = self._get_neighbors(distances, i)
            if len(neighbors) >= self.min_samples:
                core_samples.append(i)
        
        self.core_sample_indices = np.array(core_samples)
        
        # Expand clusters
        cluster_id = 0
        for i in range(n_samples):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = self._get_neighbors(distances, i)
            
            if len(neighbors) >= self.min_samples:
                # Core point - start new cluster
                labels = self._expand_cluster(X, distances, i, neighbors,
                                             cluster_id, labels, visited)
                cluster_id += 1
        
        self.labels = labels
        self.n_clusters = cluster_id
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Note: DBSCAN doesn't naturally handle new data points.
        This assigns points to the nearest cluster if within eps distance.
        """
        distances = cdist(X, self.core_sample_indices, metric=self.metric)
        labels = np.full(len(X), -1)
        
        for i in range(len(X)):
            # Find closest core point
            min_dist_idx = np.argmin(distances[i])
            if distances[i, min_dist_idx] <= self.eps:
                core_idx = self.core_sample_indices[min_dist_idx]
                labels[i] = self.labels[core_idx]
        
        return labels
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels
    
    def get_noise_indices(self) -> np.ndarray:
        """Get indices of noise points."""
        return np.where(self.labels == -1)[0]


class HierarchicalClustering:
    """
    Agglomerative Hierarchical Clustering.
    
    Parameters:
        n_clusters: Number of clusters
        linkage: Linkage method ('single', 'complete', 'average', 'ward')
    """
    
    def __init__(self, n_clusters: int = 8, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        
        self.labels = None
        self.children = None
        self.distances = None
    
    def _compute_linkage_distance(self, cluster_i: List[int], cluster_j: List[int],
                                  X: np.ndarray, Z: np.ndarray) -> float:
        """Compute distance between clusters based on linkage method."""
        points_i = X[cluster_i]
        points_j = X[cluster_j]
        
        if self.linkage == 'single':
            # Minimum distance
            return np.min(cdist(points_i, points_j))
        elif self.linkage == 'complete':
            # Maximum distance
            return np.max(cdist(points_i, points_j))
        elif self.linkage == 'average':
            # Average distance
            return np.mean(cdist(points_i, points_j))
        elif self.linkage == 'ward':
            # Ward's method (minimize variance)
            combined = np.vstack([points_i, points_j])
            var_combined = np.sum(np.var(combined, axis=0)) * len(combined)
            var_i = np.sum(np.var(points_i, axis=0)) * len(points_i)
            var_j = np.sum(np.var(points_j, axis=0)) * len(points_j)
            return var_combined - var_i - var_j
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
    
    def fit(self, X: np.ndarray) -> 'HierarchicalClustering':
        """
        Fit hierarchical clustering.
        
        Args:
            X: Data matrix (n_samples, n_features)
            
        Returns:
            Self
        """
        n_samples = X.shape[0]
        
        # Initialize each point as its own cluster
        clusters = {i: [i] for i in range(n_samples)}
        cluster_labels = {i: i for i in range(n_samples)}
        
        self.children = []
        self.distances = []
        
        # Agglomerative clustering
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = np.inf
            merge_i, merge_j = None, None
            
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    ci, cj = cluster_ids[i], cluster_ids[j]
                    dist = self._compute_linkage_distance(clusters[ci], clusters[cj], X, self.children)
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = ci, cj
            
            # Merge clusters
            new_cluster_id = max(clusters.keys()) + 1
            clusters[new_cluster_id] = clusters[merge_i] + clusters[merge_j]
            
            self.children.append([merge_i, merge_j])
            self.distances.append(min_dist)
            
            del clusters[merge_i]
            del clusters[merge_j]
        
        # Assign labels
        self.labels = np.zeros(n_samples, dtype=int)
        for label, cluster in enumerate(clusters.values()):
            for idx in cluster:
                self.labels[idx] = label
        
        return self
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels
    
    def get_dendrogram_data(self) -> Dict:
        """Get data for plotting dendrogram."""
        return {
            'children': np.array(self.children),
            'distances': np.array(self.distances)
        }
