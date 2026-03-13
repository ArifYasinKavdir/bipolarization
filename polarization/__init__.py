"""
polarization — a library for detecting and quantifying polarization.

Public API
----------
calculate_scores(df, first_variable, second_variable, **kwargs)
    Compute polarization or consensus scores with bootstrap CIs.

weight_matrix_generator(start_value, end_value, p, q, type, kernel)
    Build a raw weight matrix.

dashboard_pair(df, x, y, **kwargs)
    Full visual dashboard for a variable pair.

weight_matrix_visualization(start_value, end_value, p, q, type, kernel)
    Quick heatmap of the weight matrix.
"""

from .weights import weight_matrix_generator
from .scores import calculate_scores
from .visualization import dashboard_pair, weight_matrix_visualization

__all__ = [
    "weight_matrix_generator",
    "calculate_scores",
    "dashboard_pair",
    "weight_matrix_visualization",
]
