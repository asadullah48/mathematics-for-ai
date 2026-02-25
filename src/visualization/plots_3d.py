"""
3D Plotting Module

3D visualization functions for mathematical concepts and ML results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import Optional, Tuple, Callable, List
import warnings


class Plot3D:
    """
    3D plotting utilities for mathematics and machine learning.
    
    Features:
    - 3D surface plots
    - 3D scatter plots
    - 3D contour plots
    - Vector field visualization
    - Loss landscape visualization
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 10), style: str = 'default'):
        self.figsize = figsize
        self.style = style
        
        try:
            plt.style.use(style)
        except:
            pass
    
    def plot_surface(self, f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    x_range: Tuple[float, float] = (-5, 5),
                    y_range: Tuple[float, float] = (-5, 5),
                    n_points: int = 100,
                    cmap: str = 'viridis',
                    alpha: float = 0.9,
                    title: str = '3D Surface',
                    view_angle: Tuple[float, float] = (30, 45),
                    ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot 3D surface of a function f(x, y).
        
        Args:
            f: Function f(x, y) -> z
            x_range: X axis range
            y_range: Y axis range
            n_points: Grid resolution
            cmap: Colormap
            alpha: Surface transparency
            title: Plot title
            view_angle: (elevation, azimuth) angles
            ax: Existing 3D axes
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha,
                              linewidth=0, antialiased=True)
        
        # Add colorbar
        if fig:
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='f(x,y)')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        return ax
    
    def plot_wireframe(self, f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                      x_range: Tuple[float, float] = (-5, 5),
                      y_range: Tuple[float, float] = (-5, 5),
                      n_points: int = 50,
                      color: str = 'blue',
                      title: str = '3D Wireframe',
                      ax: Optional[Axes3D] = None) -> Axes3D:
        """Plot 3D wireframe of a function."""
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        ax.plot_wireframe(X, Y, Z, color=color, linewidth=0.5, alpha=0.7)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        return ax
    
    def plot_scatter_3d(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       classes: Optional[np.ndarray] = None,
                       title: str = '3D Scatter Plot',
                       ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot 3D scatter plot.
        
        Args:
            X: Data points (n_samples, 3)
            y: Optional values for coloring
            classes: Optional class labels
            title: Plot title
            ax: Existing 3D axes
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        if classes is not None:
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], 
                               c=classes, cmap='tab10', 
                               alpha=0.6, s=50, edgecolors='k')
            if fig:
                fig.colorbar(scatter, ax=ax, label='Class')
        elif y is not None:
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                               c=y, cmap='viridis',
                               alpha=0.6, s=50, edgecolors='k')
            if fig:
                fig.colorbar(scatter, ax=ax, label='Value')
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.6, s=50)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.set_zlabel('X3', fontsize=12)
        
        return ax
    
    def plot_contour_3d(self, f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                       x_range: Tuple[float, float] = (-5, 5),
                       y_range: Tuple[float, float] = (-5, 5),
                       n_points: int = 100,
                       levels: int = 20,
                       title: str = '3D Contour',
                       ax: Optional[Axes3D] = None) -> Axes3D:
        """Plot 3D surface with contour lines."""
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        # Surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        # Contour on surface
        contour = ax.contour(X, Y, Z, levels=levels, colors='black', 
                            linewidths=0.5, alpha=0.5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        return ax
    
    def plot_loss_landscape(self, loss_fn: Callable[[np.ndarray], float],
                           center_point: np.ndarray,
                           x_range: Tuple[float, float] = (-5, 5),
                           y_range: Tuple[float, float] = (-5, 5),
                           n_points: int = 100,
                           title: str = 'Loss Landscape',
                           ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot loss landscape around a point in parameter space.
        
        Args:
            loss_fn: Loss function
            center_point: Center point in parameter space
            x_range: Range for first direction
            y_range: Range for second direction
            n_points: Grid resolution
            title: Plot title
            ax: Existing 3D axes
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        # Generate random directions
        np.random.seed(42)
        n_params = len(center_point)
        dir1 = np.random.randn(n_params)
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = np.random.randn(n_params)
        dir2 = dir2 - np.dot(dir2, dir1) * dir1  # Make orthogonal
        dir2 = dir2 / np.linalg.norm(dir2)
        
        # Create grid in direction space
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        
        # Compute loss at each point
        Z = np.zeros_like(X)
        for i in range(n_points):
            for j in range(n_points):
                point = center_point + X[i, j] * dir1 + Y[i, j] * dir2
                Z[i, j] = loss_fn(point)
        
        # Plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
        if fig:
            fig.colorbar(surf, ax=ax, label='Loss')
        
        # Mark center
        ax.scatter(0, 0, loss_fn(center_point), color='red', s=100, label='Center')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Direction 1', fontsize=12)
        ax.set_ylabel('Direction 2', fontsize=12)
        ax.set_zlabel('Loss', fontsize=12)
        ax.legend()
        
        return ax
    
    def plot_decision_surface_3d(self, classifier, X: np.ndarray, y: np.ndarray,
                                resolution: int = 50,
                                title: str = '3D Decision Surface',
                                ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot 3D decision boundary for a classifier.
        
        Args:
            classifier: Trained classifier
            X: Data points (n_samples, 3)
            y: Labels
            resolution: Grid resolution
            title: Plot title
            ax: Existing 3D axes
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        # Create grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        z_min, z_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
        
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution),
            np.linspace(z_min, z_max, resolution)
        )
        
        # For visualization, show 2D slices
        slice_idx = resolution // 2
        
        # Plot data points
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='tab10', 
                  alpha=0.8, s=50, edgecolors='k')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.set_zlabel('X3', fontsize=12)
        
        return ax
    
    def plot_vector_field_3d(self, Fx: Callable, Fy: Callable, Fz: Callable,
                            x_range: Tuple[float, float] = (-3, 3),
                            y_range: Tuple[float, float] = (-3, 3),
                            z_range: Tuple[float, float] = (-3, 3),
                            density: int = 10,
                            title: str = '3D Vector Field',
                            ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot 3D vector field.
        
        Args:
            Fx: X-component function
            Fy: Y-component function
            Fz: Z-component function
            x_range: X axis range
            y_range: Y axis range
            z_range: Z axis range
            density: Grid density
            title: Plot title
            ax: Existing 3D axes
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        z = np.linspace(z_range[0], z_range[1], density)
        X, Y, Z = np.meshgrid(x, y, z)
        
        U = Fx(X, Y, Z)
        V = Fy(X, Y, Z)
        W = Fz(X, Y, Z)
        
        # Normalize for better visualization
        magnitude = np.sqrt(U**2 + V**2 + W**2)
        magnitude = np.where(magnitude == 0, 1, magnitude)
        
        ax.quiver(X, Y, Z, U/magnitude, V/magnitude, W/magnitude,
                 length=0.5, normalize=True, alpha=0.5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        return ax
    
    def plot_parametric_surface(self, fx: Callable, fy: Callable, fz: Callable,
                               u_range: Tuple[float, float] = (0, 2*np.pi),
                               v_range: Tuple[float, float] = (0, np.pi),
                               n_u: int = 50, n_v: int = 50,
                               cmap: str = 'viridis',
                               title: str = 'Parametric Surface',
                               ax: Optional[Axes3D] = None) -> Axes3D:
        """
        Plot parametric surface.
        
        Args:
            fx: X(u, v) function
            fy: Y(u, v) function
            fz: Z(u, v) function
            u_range: U parameter range
            v_range: V parameter range
            n_u: Number of u points
            n_v: Number of v points
            cmap: Colormap
            title: Plot title
            ax: Existing 3D axes
        """
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(u_range[0], u_range[1], n_u)
        v = np.linspace(v_range[0], v_range[1], n_v)
        U, V = np.meshgrid(u, v)
        
        X = fx(U, V)
        Y = fy(U, V)
        Z = fz(U, V)
        
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9)
        if fig:
            fig.colorbar(surf, ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        return ax
    
    def plot_sphere(self, center: Tuple[float, float, float] = (0, 0, 0),
                   radius: float = 1.0, n_points: int = 50,
                   cmap: str = 'viridis', title: str = 'Sphere',
                   ax: Optional[Axes3D] = None) -> Axes3D:
        """Plot a sphere."""
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        surf = ax.plot_surface(x, y, z, cmap=cmap, alpha=0.8)
        if fig:
            fig.colorbar(surf, ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        # Equal aspect ratio
        max_range = radius
        ax.set_xlim(center[0] - max_range, center[0] + max_range)
        ax.set_ylim(center[1] - max_range, center[1] + max_range)
        ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        return ax
    
    def plot_ellipsoid(self, center: Tuple[float, float, float] = (0, 0, 0),
                      radii: Tuple[float, float, float] = (1, 1, 1),
                      n_points: int = 50, cmap: str = 'viridis',
                      title: str = 'Ellipsoid',
                      ax: Optional[Axes3D] = None) -> Axes3D:
        """Plot an ellipsoid."""
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        
        x = center[0] + radii[0] * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radii[1] * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))
        
        surf = ax.plot_surface(x, y, z, cmap=cmap, alpha=0.8)
        if fig:
            fig.colorbar(surf, ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        return ax
    
    def plot_cone(self, height: float = 3, radius: float = 1,
                 n_points: int = 50, cmap: str = 'viridis',
                 title: str = 'Cone', ax: Optional[Axes3D] = None) -> Axes3D:
        """Plot a cone."""
        fig = None
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
        
        theta = np.linspace(0, 2 * np.pi, n_points)
        z = np.linspace(0, height, n_points)
        Theta, Z = np.meshgrid(theta, z)
        
        R = radius * Z / height
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8)
        if fig:
            fig.colorbar(surf, ax=ax)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
        return ax
    
    def save(self, filepath: str, dpi: int = 300):
        """Save the current figure."""
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
    
    def show(self):
        """Display the plot."""
        plt.show()
