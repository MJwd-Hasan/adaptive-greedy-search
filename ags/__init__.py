"""
AGS -- Adaptive Greedy Search

Surrogate-assisted hyperparameter optimization over finite, typed, and
optionally conditional search spaces. AGS combines lazy candidate generation,
RF/GP/KDE surrogate guidance, adaptive trust-region climbers, confidence-bound
multi-fidelity racing, and diversity-aware batch selection and restart.

    from ags import AdaptiveGreedySearch

is all you need -- numpy, scikit-learn, and joblib are imported internally.
"""

from .core import AdaptiveGreedySearch, Climber
from .surrogates import (
    RandomForestSurrogate,
    KDEDensityRatioSurrogate,
    TPESurrogate,
    build_gp_surrogate,
)
from .search_space import (
    NumericParameter,
    FloatParameter,
    IntegerParameter,
    OrdinalParameter,
    CategoricalParameter,
    ConditionalParameter,
    StructuredSearchSpace,
)

__all__ = [
    "AdaptiveGreedySearch",
    "Climber",
    "RandomForestSurrogate",
    "KDEDensityRatioSurrogate",
    "TPESurrogate",
    "build_gp_surrogate",
    "NumericParameter",
    "FloatParameter",
    "IntegerParameter",
    "OrdinalParameter",
    "CategoricalParameter",
    "ConditionalParameter",
    "StructuredSearchSpace",
]
__version__ = "3.7.1"
