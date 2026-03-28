"""
stfa/__init__.py
================
Public API for the STFA (Spatial Transcriptomics Fused-Gromov-Wasserstein Alignment) package.
"""

from .align import pairwise_align_stfa
from .evaluate import (
    cell_type_matching_metric,
    get_perf_metrics,
    get_perf_metrics_enhanced,
)
from .utils import (
    stack_slices_pairwise,
    generalized_procrustes_analysis,
    visualize_alignment,
    visualize_alignment_unbalanced,
    neighborhood_distribution,
)

__all__ = [
    "pairwise_align_stfa",
    "cell_type_matching_metric",
    "get_perf_metrics",
    "get_perf_metrics_enhanced",
    "stack_slices_pairwise",
    "generalized_procrustes_analysis",
    "visualize_alignment",
    "visualize_alignment_unbalanced",
    "neighborhood_distribution",
]
