"""
AGS -- Adaptive Greedy Search

Surrogate-assisted hyperparameter search over finite discrete spaces.
AGS combines typed geometry, RF/GP/KDE surrogate guidance, adaptive
trust-region climbers, confidence-bound multi-fidelity racing, and
diversity-aware batch selection and restart.

    from ags import AdaptiveGreedySearch

is all you need -- numpy, scikit-learn, and joblib are imported
internally by the package.
"""

from .core import AdaptiveGreedySearch, Climber
from .surrogates import (
    RandomForestSurrogate,
    KDEDensityRatioSurrogate,
    TPESurrogate,
    build_gp_surrogate,
)
from .search_space import NumericParameter, OrdinalParameter, CategoricalParameter

__all__ = [
    "AdaptiveGreedySearch",
    "Climber",
    "RandomForestSurrogate",
    "KDEDensityRatioSurrogate",
    "TPESurrogate",
    "build_gp_surrogate",
    "NumericParameter",
    "OrdinalParameter",
    "CategoricalParameter",
]
__version__ = "3.6.0"
