"""MI-directional reward pretraining with Dame-inspired temporally-aligned potential distillation.

Built on RLinf/LaWAM. This package provides:
  - Dame-style B-spline soft-histogram MI estimation
  - Monotonic temporal alignment of candidate and reference trajectories
  - Directional information-potential scoring
  - Pseudo-preference construction
  - State-potential reward model distillation
"""

__all__ = ["__version__"]

__version__ = "0.2.0"
