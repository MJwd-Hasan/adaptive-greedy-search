"""Acquisition functions for model-guided hyperparameter search."""

import numpy as np
from scipy.stats import norm


def acquisition_scores(
    surrogate,
    features,
    acquisition="auto",
    exploration_weight=2.0,
    best_score=None,
):
    """
    Score candidate feature rows independently of the search strategy.

    Supported acquisitions:
    - auto: density ratio for KDE; UCB for predictive surrogates.
    - ucb: predictive mean + weight * predictive spread.
    - ei / expected_improvement: expected improvement under a normal
      approximation using the surrogate's mean and standard deviation.
    - density_ratio: only available for the KDE density-ratio surrogate.

    For the Random Forest, standard deviation is between-tree dispersion,
    not a calibrated posterior uncertainty. EI is consequently an approximate
    ranking rule in RF mode. With a GP, EI uses the GP predictive distribution.
    """
    features = np.asarray(features, dtype=float)
    is_density_ratio = hasattr(surrogate, "score_density_ratio")

    if acquisition == "auto":
        acquisition = "density_ratio" if is_density_ratio else "ucb"

    if acquisition == "density_ratio":
        if not is_density_ratio:
            raise ValueError(
                "acquisition='density_ratio' requires surrogate_type='kde' "
                "(or the deprecated 'tpe' alias)."
            )
        return np.asarray(surrogate.score_density_ratio(features), dtype=float)

    if is_density_ratio:
        raise ValueError(
            "The KDE density-ratio surrogate does not expose predictive "
            "mean/std. Use acquisition='auto' or 'density_ratio', or choose "
            "an RF/GP surrogate for UCB or EI."
        )

    if acquisition not in {"ucb", "ei", "expected_improvement"}:
        raise ValueError(
            "Unknown acquisition {!r}; choose 'auto', 'ucb', 'ei', "
            "'expected_improvement', or 'density_ratio'.".format(acquisition)
        )

    mean, std = surrogate.predict(features, return_std=True)
    mean = np.asarray(mean, dtype=float)
    std = np.maximum(np.asarray(std, dtype=float), 0.0)

    if acquisition == "ucb":
        return mean + float(exploration_weight) * std

    if best_score is None:
        raise ValueError("EI requires best_score from completed evaluations.")

    improvement = mean - float(best_score)
    scores = np.maximum(improvement, 0.0)
    positive_std = std > 1e-12

    if np.any(positive_std):
        z = improvement[positive_std] / std[positive_std]
        scores[positive_std] = (
            improvement[positive_std] * norm.cdf(z)
            + std[positive_std] * norm.pdf(z)
        )
    return scores
