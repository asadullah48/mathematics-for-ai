"""
Animations Module

Animated visualizations for mathematical concepts and ML algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, List, Tuple, Callable
import warnings


class Animation:
    """
    Animation utilities for mathematics and machine learning.
    
    Features:
    - Gradient descent animation
    - Newton's method animation
    - K-means clustering animation
    - Function transformation animations
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), fps: int = 30):
        self.figsize = figsize
        self.fps = fps
        self.animations = []
    
    def gradient_descent_animation(self, f: Callable[[np.ndarray], float],
                                   gradient: Callable[[np.ndarray], np.ndarray],
                                   x0: np.ndarray, n_steps: int = 50,
                                   learning_rate: float = 0.1,
                                   x_range: Tuple[float, float] = (-5, 5),
                                   y_range: Tuple[float, float] = (-5, 5),
                                   title: str = 'Gradient Descent Animation',
                                   save_path: Optional[str] = None) -> FuncAnimation:
        """
        Animate gradient descent optimization.
        
        Args:
            f: Objective function
            gradient: Gradient function
            x0: Starting point
            n_steps: Number of steps
            learning_rate: Step size
            x_range: X axis range
            y_range: Y axis range
            title: Plot title
            save_path: Path to save animation
            
        Returns:
            Matplotlib animation object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create contour
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = f(np.stack([X, Y], axis=-1))
        
        contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.3)
        plt.colorbar(contour, ax=ax)
        
        # Compute path
        path = [x0.copy()]
        current = x0.copy()
        
        for _ in range(n_steps):
            grad = gradient(current)
            current = current - learning_rate * grad
            path.append(current.copy())
        
        path = np.array(path)
        
        # Animation elements
        path_line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.5)
        point, = ax.plot([], [], 'ro', markersize=10)
        start_point = ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start')[0]
        end_point = ax.plot(path[-1, 0], path[-1, 1], 'yx', markersize=12, label='End')[0]
        
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        def init():
            path_line.set_data([], [])
            point.set_data([], [])
            step_text.set_text('')
            return path_line, point, step_text, start_point, end_point
        
        def animate(i):
            path_line.set_data(path[:i+1, 0], path[:i+1, 1])
            point.set_data(path[i, 0], path[i, 1])
            step_text.set_text(f'Step: {i}/{n_steps}')
            return path_line, point, step_text, start_point, end_point
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_steps+1,
                           interval=200, blit=True)
        
        if save_path:
            anim.save(save_path, writer=PillowWriter(fps=self.fps))
            print(f"Animation saved to {save_path}")
        
        self.animations.append(anim)
        return anim
    
    def newton_method_animation(self, f: Callable[[np.ndarray], float],
                               gradient: Callable[[np.ndarray], np.ndarray],
                               hessian: Callable[[np.ndarray], np.ndarray],
                               x0: np.ndarray, n_steps: int = 20,
                               x_range: Tuple[float, float] = (-5, 5),
                               y_range: Tuple[float, float] = (-5, 5),
                               title: str = "Newton's Method Animation",
                               save_path: Optional[str] = None) -> FuncAnimation:
        """
        Animate Newton's method optimization.
        
        Args:
            f: Objective function
            gradient: Gradient function
            hessian: Hessian function
            x0: Starting point
            n_steps: Number of steps
            x_range: X axis range
            y_range: Y axis range
            title: Plot title
            save_path: Path to save animation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create contour
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = f(np.stack([X, Y], axis=-1))
        
        contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.3)
        plt.colorbar(contour, ax=ax)
        
        # Compute path using Newton's method
        path = [x0.copy()]
        current = x0.copy()
        
        for _ in range(n_steps):
            grad = gradient(current)
            hess = hessian(current)
            
            # Add regularization for stability
            hess_reg = hess + 1e-6 * np.eye(2)
            
            try:
                delta = np.linalg.solve(hess_reg, grad)
            except:
                delta = np.linalg.lstsq(hess_reg, grad, rcond=None)[0]
            
            current = current - delta
            path.append(current.copy())
        
        path = np.array(path)
        
        # Animation elements
        path_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.5)
        point, = ax.plot([], [], 'bo', markersize=10)
        start_point = ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start')[0]
        
        step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend()
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        def init():
            path_line.set_data([], [])
            point.set_data([], [])
            step_text.set_text('')
            return path_line, point, step_text, start_point
        
        def animate(i):
            path_line.set_data(path[:i+1, 0], path[:i+1, 1])
            point.set_data(path[i, 0], path[i, 1])
            step_text.set_text(f'Iteration: {i}/{n_steps}')
            return path_line, point, step_text, start_point
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_steps+1,
                           interval=300, blit=True)
        
        if save_path:
            anim.save(save_path, writer=PillowWriter(fps=self.fps))
            print(f"Animation saved to {save_path}")
        
        self.animations.append(anim)
        return anim
    
    def kmeans_animation(self, X: np.ndarray, n_clusters: int = 4,
                        n_iterations: int = 20,
                        title: str = 'K-Means Clustering Animation',
                        save_path: Optional[str] = None) -> FuncAnimation:
        """
        Animate K-Means clustering.
        
        Args:
            X: Data points
            n_clusters: Number of clusters
            n_iterations: Number of iterations
            title: Plot title
            save_path: Path to save animation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        np.random.seed(42)
        
        # Initialize centroids randomly
        indices = np.random.choice(len(X), n_clusters, replace=False)
        centroids = X[indices].copy()
        
        # Pre-compute all iterations
        all_centroids = [centroids.copy()]
        all_labels = []
        
        for _ in range(n_iterations):
            # Assign clusters
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            all_labels.append(labels.copy())
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]
            
            centroids = new_centroids
            all_centroids.append(centroids.copy())
        
        # Animation elements
        scatter = ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=50)
        centroid_scatter = ax.scatter([], [], c='red', s=200, marker='X', 
                                      edgecolors='black', linewidths=2)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        
        iter_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top')
        
        def init():
            scatter.set_facecolors('gray')
            centroid_scatter.set_offsets(np.empty((0, 2)))
            iter_text.set_text('')
            return scatter, centroid_scatter, iter_text
        
        def animate(i):
            # Update point colors
            if i > 0:
                labels = all_labels[i-1]
                scatter.set_array(labels)
                scatter.set_cmap('tab10')
            
            # Update centroids
            centroid_scatter.set_offsets(all_centroids[i])
            
            iter_text.set_text(f'Iteration: {i}/{n_iterations}')
            return scatter, centroid_scatter, iter_text
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_iterations+1,
                           interval=500, blit=False)
        
        if save_path:
            anim.save(save_path, writer=PillowWriter(fps=self.fps))
            print(f"Animation saved to {save_path}")
        
        self.animations.append(anim)
        return anim
    
    def function_transformation_animation(self, 
                                          transformations: List[Callable[[np.ndarray], np.ndarray]],
                                          X: np.ndarray,
                                          titles: Optional[List[str]] = None,
                                          save_path: Optional[str] = None) -> FuncAnimation:
        """
        Animate data transformations.
        
        Args:
            transformations: List of transformation functions
            X: Input data
            titles: Titles for each transformation
            save_path: Path to save animation
        """
        fig, axes = plt.subplots(1, len(transformations), figsize=(5 * len(transformations), 5))
        if len(transformations) == 1:
            axes = [axes]
        
        if titles is None:
            titles = [f'Transformation {i+1}' for i in range(len(transformations))]
        
        # Apply transformations
        transformed_data = [X]
        for transform in transformations:
            transformed_data.append(transform(transformed_data[-1]))
        
        # Initialize plots
        scatters = []
        for i, (ax, title) in enumerate(zip(axes, titles)):
            scatter = ax.scatter([], [], alpha=0.6, s=50, c='blue')
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('X1', fontsize=10)
            ax.set_ylabel('X2', fontsize=10)
            ax.grid(True, alpha=0.3)
            scatters.append(scatter)
        
        def init():
            for scatter in scatters:
                scatter.set_offsets(np.empty((0, 2)))
            return tuple(scatters)
        
        def animate(i):
            for j, (scatter, data) in enumerate(zip(scatters, transformed_data)):
                if i < len(data):
                    # Show progressive reveal
                    scatter.set_offsets(data[:max(1, int(len(data) * i / len(transformations)))])
            return tuple(scatters)
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=len(transformations)+1,
                           interval=300, blit=False)
        
        if save_path:
            anim.save(save_path, writer=PillowWriter(fps=self.fps))
            print(f"Animation saved to {save_path}")
        
        self.animations.append(anim)
        return anim
    
    def pca_animation(self, X: np.ndarray, n_components: int = 2,
                     title: str = 'PCA Animation',
                     save_path: Optional[str] = None) -> FuncAnimation:
        """
        Animate PCA dimensionality reduction.
        
        Args:
            X: Data points
            n_components: Number of principal components
            title: Plot title
            save_path: Path to save animation
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Center data
        X_centered = X - X.mean(axis=0)
        
        # Compute covariance and eigenvectors
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Animation
        scatter = ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, c='blue')
        
        # Principal component arrows
        arrows = []
        for i in range(n_components):
            arrow = ax.arrow(0, 0, 0, 0, head_width=0.5, head_length=0.3,
                           fc='red', ec='red', linewidth=2, alpha=0.8)
            arrows.append(arrow)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add eigenvalue text
        text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                      fontsize=10, verticalalignment='top')
        
        def init():
            for arrow in arrows:
                arrow.set_data((0, 0, 0, 0))
            text.set_text('')
            return tuple([scatter] + arrows + [text])
        
        def animate(i):
            if i < n_components:
                # Show principal components one by one
                scale = np.sqrt(eigenvalues[i]) * 3
                for j in range(i + 1):
                    arrows[j].set_data((0, eigenvectors[0, j] * scale * (i >= j),
                                       0, eigenvectors[1, j] * scale * (i >= j)))
            
            text.set_text(f'Component: {min(i+1, n_components)}/{n_components}\n' +
                         f'Explained Variance: {eigenvalues[min(i, n_components-1)]/eigenvalues.sum()*100:.1f}%')
            
            return tuple([scatter] + arrows + [text])
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=n_components,
                           interval=1000, blit=False)
        
        if save_path:
            anim.save(save_path, writer=PillowWriter(fps=self.fps))
            print(f"Animation saved to {save_path}")
        
        self.animations.append(anim)
        return anim
    
    def show(self):
        """Display all animations."""
        plt.show()
    
    def clear(self):
        """Clear all stored animations."""
        self.animations = []
        plt.close('all')
