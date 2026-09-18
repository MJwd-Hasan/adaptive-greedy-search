"""
AGS -- Adaptive Greedy Search

Surrogate-guided hyperparameter search over a discrete grid: greedy
radius-1 hill-climbing with a TPE (or GP) surrogate, radius-based
plateau escape instead of a full-grid scan, and sequential fold-by-fold
cross-validation with bound-based pruning to skip clearly-uncompetitive
candidates early.

    from ags import AdaptiveGreedySearch

is all you need -- everything else (numpy, scikit-learn) is imported
internally by the package.
"""

from .core import AdaptiveGreedySearch
from .surrogates import TPESurrogate, build_gp_surrogate

__all__ = ["AdaptiveGreedySearch", "TPESurrogate", "build_gp_surrogate"]
__version__ = "2.0.0"
