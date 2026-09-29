"""
Surrogate models used by AdaptiveGreedySearch to score unevaluated
hyperparameter configurations.

Both surrogates expose the same interface:
    .fit(X, y)
    .predict(states, return_std=True) -> (mean, std)
so AdaptiveGreedySearch never needs to know which one it's using.
"""

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neighbors import KernelDensity


class TPESurrogate:
    """
    Minimal Tree-structured Parzen Estimator surrogate (same idea as
    Optuna's default sampler). Splits observed points into "good" (top
    `gamma` fraction by score) and "bad" (the rest), fits a KDE over
    each group in grid-coordinate space, and scores candidates by the
    good/bad log-density ratio.
    """

    def __init__(self, gamma=0.25, bandwidth=1.0):
        self.gamma = gamma
        self.bandwidth = bandwidth
        self.good_kde = None
        self.bad_kde = None
        self._y_min = 0.0
        self._y_max = 1.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        order = np.argsort(y)[::-1]
        n_good = max(1, int(np.ceil(self.gamma * len(y))))
        good_idx = order[:n_good]
        bad_idx = order[n_good:]
        if len(bad_idx) == 0:
            bad_idx = good_idx

        self._y_min, self._y_max = float(y.min()), float(y.max())

        self.good_kde = KernelDensity(bandwidth=self.bandwidth).fit(X[good_idx])
        self.bad_kde = KernelDensity(bandwidth=self.bandwidth).fit(X[bad_idx])
        return self

    def predict(self, states, return_std=True):
        states = np.asarray(states, dtype=float)

        log_l = self.good_kde.score_samples(states)
        log_g = self.bad_kde.score_samples(states)
        tpe_score = log_l - log_g

        squashed = 1.0 / (1.0 + np.exp(-tpe_score))
        mean = self._y_min + squashed * (self._y_max - self._y_min)

        if not return_std:
            return mean

        std = (1.0 / (1.0 + np.abs(tpe_score))) * (self._y_max - self._y_min)
        return mean, std


def build_gp_surrogate(random_state=None):
    """Construct the Matern-kernel GaussianProcessRegressor surrogate."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, nu=2.5)
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1))
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=random_state,
    )


def make_surrogate(surrogate_type, tpe_gamma=0.25, tpe_bandwidth=1.0, random_state=None):
    """Factory: build the requested surrogate ("tpe" or "gp")."""
    if surrogate_type == "tpe":
        return TPESurrogate(gamma=tpe_gamma, bandwidth=tpe_bandwidth)
    elif surrogate_type == "gp":
        return build_gp_surrogate(random_state=random_state)
    raise ValueError(f"Unknown surrogate_type '{surrogate_type}', expected 'tpe' or 'gp'")
