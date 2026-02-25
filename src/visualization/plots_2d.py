"""
2D Plotting Module

2D visualization functions for mathematical concepts and ML results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple, Union, Callable
import warnings


class Plot2D:
    """
    2D plotting utilities for mathematics and machine learning.
    
    Features:
    - Function plotting
    - Scatter plots with decision boundaries
    - Contour plots
    - Vector field visualization
    - Gradient descent visualization
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = 'seaborn-v0_8'):
        self.figsize = figsize
        self.style = style
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
    
    def plot_function(self, f: Callable[[np.ndarray], np.ndarray],
                     x_range: Tuple[float, float] = (-10, 10),
                     n_points: int = 1000,
                     title: str = 'Function Plot',
                     xlabel: str = 'x', ylabel: str = 'f(x)',
                     show_grid: bool = True,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot a 1D function.
        
        Args:
            f: Function to plot
            x_range: Range of x values
            n_points: Number of points for smooth curve
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_grid: Show grid
            ax: Existing axes to plot on
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = f(x)
        
        ax.plot(x, y, 'b-', linewidth=2, label='f(x)')
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        ax.legend()
        return ax
    
    def plot_multiple_functions(self, functions: List[Callable],
                               labels: Optional[List[str]] = None,
                               x_range: Tuple[float, float] = (-10, 10),
                               n_points: int = 1000,
                               title: str = 'Functions',
                               colors: Optional[List[str]] = None,
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot multiple functions on the same axes."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        
        if labels is None:
            labels = [f'f{i+1}(x)' for i in range(len(functions))]
        
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(functions)))
        
        for i, (f, label, color) in enumerate(zip(functions, labels, colors)):
            y = f(x)
            ax.plot(x, y, color=color, linewidth=2, label=label)
        
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_scatter(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                    classes: Optional[np.ndarray] = None,
                    title: str = 'Scatter Plot',
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot 2D scatter plot with optional class coloring.
        
        Args:
            X: Data points (n_samples, 2)
            y: Optional target values for colorbar
            classes: Optional class labels for discrete coloring
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        X = np.asarray(X)
        
        if classes is not None:
            # Color by class
            scatter = ax.scatter(X[:, 0], X[:, 1], c=classes, 
                               cmap='tab10', alpha=0.6, s=50, edgecolors='k')
            plt.colorbar(scatter, ax=ax, label='Class')
        elif y is not None:
            # Continuous coloring
            scatter = ax.scatter(X[:, 0], X[:, 1], c=y, 
                               cmap='viridis', alpha=0.6, s=50, edgecolors='k')
            plt.colorbar(scatter, ax=ax, label='Value')
        else:
            ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=50, edgecolors='k')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_decision_boundary(self, classifier, X: np.ndarray, y: np.ndarray,
                              resolution: int = 300,
                              title: str = 'Decision Boundary',
                              show_points: bool = True,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot decision boundary for a 2D classifier.
        
        Args:
            classifier: Trained classifier with predict method
            X: Training features (n_samples, 2)
            y: Training labels
            resolution: Grid resolution
            title: Plot title
            show_points: Show training points
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        # Predict on grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        try:
            Z = classifier.predict(grid_points)
        except:
            # Try decision function for smoother boundaries
            try:
                Z = classifier.decision_function(grid_points)
            except:
                Z = classifier.predict(grid_points)
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        if Z.dtype in [bool, int] or len(np.unique(Z)) < 10:
            # Discrete classes
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
        else:
            # Continuous values
            contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
            plt.colorbar(contour, ax=ax)
        
        # Plot training points
        if show_points:
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='tab10', 
                      edgecolors='k', alpha=0.8, s=50)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X1', fontsize=12)
        ax.set_ylabel('X2', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_contour(self, f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    x_range: Tuple[float, float] = (-5, 5),
                    y_range: Tuple[float, float] = (-5, 5),
                    n_points: int = 100,
                    levels: int = 20,
                    filled: bool = True,
                    title: str = 'Contour Plot',
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot 2D contour of a function f(x, y).
        
        Args:
            f: Function f(x, y) -> z
            x_range: X axis range
            y_range: Y axis range
            n_points: Grid resolution
            levels: Number of contour levels
            filled: Filled contour
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.linspace(x_range[0], x_range[1], n_points)
        y = np.linspace(y_range[0], y_range[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        if filled:
            contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
            plt.colorbar(contour, ax=ax, label='f(x,y)')
        else:
            contour = ax.contour(X, Y, Z, levels=levels, cmap='viridis')
            ax.clabel(contour, inline=True, fontsize=8)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal')
        
        return ax
    
    def plot_vector_field(self, Fx: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         Fy: Callable[[np.ndarray, np.ndarray], np.ndarray],
                         x_range: Tuple[float, float] = (-5, 5),
                         y_range: Tuple[float, float] = (-5, 5),
                         density: int = 20,
                         normalize: bool = False,
                         title: str = 'Vector Field',
                         ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot 2D vector field.
        
        Args:
            Fx: X-component function Fx(x, y)
            Fy: Y-component function Fy(x, y)
            x_range: X axis range
            y_range: Y axis range
            density: Grid density
            normalize: Normalize vector lengths
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.linspace(x_range[0], x_range[1], density)
        y = np.linspace(y_range[0], y_range[1], density)
        X, Y = np.meshgrid(x, y)
        
        U = Fx(X, Y)
        V = Fy(X, Y)
        
        if normalize:
            magnitude = np.sqrt(U**2 + V**2)
            magnitude = np.where(magnitude == 0, 1, magnitude)
            U = U / magnitude
            V = V / magnitude
        
        ax.quiver(X, Y, U, V, alpha=0.6, scale=50)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_gradient_field(self, f: Callable[[np.ndarray, np.ndarray], np.ndarray],
                           x_range: Tuple[float, float] = (-5, 5),
                           y_range: Tuple[float, float] = (-5, 5),
                           density: int = 20,
                           show_contours: bool = True,
                           title: str = 'Gradient Field',
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot gradient field of a scalar function.
        
        Args:
            f: Scalar function f(x, y)
            x_range: X axis range
            y_range: Y axis range
            density: Grid density
            show_contours: Show contour lines
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        # Compute gradient
        grad_x, grad_y = np.gradient(Z)
        
        # Downsample for quiver plot
        step = max(1, 100 // density)
        ax.quiver(X[::step, ::step], Y[::step, ::step], 
                 grad_x[::step, ::step], grad_y[::step, ::step],
                 alpha=0.6, color='red', scale=50)
        
        if show_contours:
            ax.contour(X, Y, Z, levels=20, alpha=0.5, cmap='gray')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal')
        
        return ax
    
    def plot_gradient_descent(self, f: Callable, gradient: Callable,
                             x0: np.ndarray, n_steps: int = 50,
                             learning_rate: float = 0.1,
                             x_range: Tuple[float, float] = (-5, 5),
                             y_range: Tuple[float, float] = (-5, 5),
                             title: str = 'Gradient Descent',
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Visualize gradient descent optimization path.
        
        Args:
            f: Objective function f(x, y)
            gradient: Gradient function
            x0: Starting point
            n_steps: Number of steps
            learning_rate: Step size
            x_range: X axis range
            y_range: Y axis range
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot contour
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        
        ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
        ax.contour(X, Y, Z, levels=20, colors='white', linewidths=0.5, alpha=0.5)
        
        # Run gradient descent
        path = [x0.copy()]
        current = x0.copy()
        
        for _ in range(n_steps):
            grad = gradient(current)
            current = current - learning_rate * grad
            path.append(current.copy())
        
        path = np.array(path)
        
        # Plot path
        ax.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6, 
               label='GD Path', alpha=0.8)
        ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start', alpha=0.8)
        ax.plot(path[-1, 0], path[-1, 1], 'yx', markersize=12, label='End', alpha=0.8)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend()
        
        return ax
    
    def plot_histogram(self, data: np.ndarray, bins: int = 30,
                      density: bool = False, cumulative: bool = False,
                      title: str = 'Histogram',
                      ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot histogram with optional density and cumulative options."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(data, bins=bins, density=density, cumulative=cumulative,
               alpha=0.7, edgecolor='black')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency' if not density else 'Density', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_distribution_comparison(self, data: np.ndarray,
                                    theoretical_dist: Callable,
                                    dist_params: Tuple,
                                    title: str = 'Distribution Comparison',
                                    ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Compare empirical data with theoretical distribution.
        
        Args:
            data: Empirical data
            theoretical_dist: Theoretical PDF function
            dist_params: Parameters for theoretical distribution
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Histogram
        ax.hist(data, bins=50, density=True, alpha=0.6, label='Empirical',
               edgecolor='black')
        
        # Theoretical PDF
        x_min, x_max = data.min(), data.max()
        x = np.linspace(x_min, x_max, 1000)
        y = theoretical_dist(x, *dist_params)
        ax.plot(x, y, 'r-', linewidth=2, label='Theoretical')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_box_plot(self, data: Union[np.ndarray, List[np.ndarray]],
                     labels: Optional[List[str]] = None,
                     title: str = 'Box Plot',
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot box plot for one or more datasets."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = [data]
        
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_correlation_matrix(self, data: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               title: str = 'Correlation Matrix',
                               ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: Data matrix (n_samples, n_features)
            feature_names: Feature names
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        corr_matrix = np.corrcoef(data.T)
        
        im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        
        if feature_names is not None:
            ax.set_xticks(range(len(feature_names)))
            ax.set_yticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_yticklabels(feature_names)
        
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title(title, fontsize=14)
        
        # Add correlation values
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center', fontsize=8)
        
        return ax
    
    def plot_learning_curve(self, train_scores: np.ndarray,
                           val_scores: Optional[np.ndarray] = None,
                           title: str = 'Learning Curve',
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot learning curve showing training/validation scores over epochs.
        
        Args:
            train_scores: Training scores per epoch
            val_scores: Validation scores per epoch (optional)
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(train_scores) + 1)
        
        ax.plot(epochs, train_scores, 'b-', linewidth=2, label='Train')
        
        if val_scores is not None:
            ax.plot(epochs, val_scores, 'r--', linewidth=2, label='Validation')
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None,
                             title: str = 'Confusion Matrix',
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot confusion matrix for classification results.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Class names
            title: Plot title
            ax: Existing axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        
        # Compute confusion matrix
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for true, pred in zip(y_true, y_pred):
            i = np.where(classes == true)[0][0]
            j = np.where(classes == pred)[0][0]
            cm[i, j] += 1
        
        # Plot
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)
        
        if class_names is None:
            class_names = [f'Class {i}' for i in classes]
        
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add values
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                       color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        ax.set_title(title, fontsize=14)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        return ax
    
    def save(self, filepath: str, dpi: int = 300):
        """Save the current figure."""
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")
    
    def show(self):
        """Display the plot."""
        plt.show()
