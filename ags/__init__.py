"""
AGS -- Adaptive Greedy Search

Surrogate-guided hyperparameter search over a discrete grid: a swarm of
independent greedy climbers (Max-Min seeded, dispatched in parallel via
joblib) sharing one TPE (or GP) surrogate, one global evaluation cache,
and one fold-level pruning engine. Boxed-in climbers escape local
plateaus by ring expansion before retiring; retired climbers are
replaced at the highest-uncertainty unvisited state while budget
remains.

    from ags import AdaptiveGreedySearch

is all you need -- numpy, scikit-learn, and joblib are imported
internally by the package.
"""

from .core import AdaptiveGreedySearch, Climber
from .surrogates import TPESurrogate, build_gp_surrogate
from .search_space import NumericParameter, OrdinalParameter, CategoricalParameter

__all__ = [
    "AdaptiveGreedySearch",
    "Climber",
    "TPESurrogate",
    "build_gp_surrogate",
    "NumericParameter",
    "OrdinalParameter",
    "CategoricalParameter",
]
__version__ = "3.2.0"
