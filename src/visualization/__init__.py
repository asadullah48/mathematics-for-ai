"""
Visualization Module

Mathematical and ML visualizations for educational purposes.
Includes 2D/3D plots, animations, and interactive visualizations.
"""

from .plots_2d import Plot2D
from .plots_3d import Plot3D
from .animations import Animation

__version__ = "0.1.0"
__all__ = ["Plot2D", "Plot3D", "Animation"]
