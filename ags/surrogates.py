"""Surrogate models used by Adaptive Greedy Search (AGS).

Predictive surrogates (Random Forest and Gaussian Process) expose fit(X, y)
and predict(X, return_std=True). For Random Forest, the reported spread is
empirical dispersion across trees, not a calibrated posterior standard deviation.

The KDE density-ratio model exposes score_density_ratio(X) instead.
It is deliberately not represented as a predictive mean/standard-deviation
model, because its log density ratio has different semantics.
"""

import warnings

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neighbors import KernelDensity


class RandomForestSurrogate:
    """Random Forest regression surrogate with between-tree dispersion."""

    def __init__(self, n_estimators=200, random_state=None, min_samples_leaf=1):
        if n_estimators < 2:
            raise ValueError("n_estimators must be >= 2 to estimate ensemble dispersion.")
        self.n_estimators = int(n_estimators)
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            n_jobs=1,
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.model.fit(X, y)
        return self

    def predict(self, X, return_std=True):
        X = np.asarray(X, dtype=float)
        if not return_std:
            return self.model.predict(X)

        tree_predictions = np.vstack(
            [tree.predict(X) for tree in self.model.estimators_]
        )
        mean = tree_predictions.mean(axis=0)
        std = tree_predictions.std(axis=0, ddof=1)
        return mean, std


class KDEDensityRatioSurrogate:
    """
    KDE density-ratio ranking model (legacy TPE-like implementation).

    It estimates the log density ratio between historically good and bad
    configurations. The resulting score is a candidate-ranking signal,
    not a predicted objective mean or a posterior uncertainty estimate.
    """

    def __init__(self, gamma=0.25, bandwidth=1.0):
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be between 0 and 1.")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be > 0.")
        self.gamma = float(gamma)
        self.bandwidth = float(bandwidth)
        self.good_kde = None
        self.bad_kde = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")
        if len(X) != len(y) or len(y) == 0:
            raise ValueError("X and y must contain the same non-zero number of rows.")

        order = np.argsort(y)[::-1]
        n_good = max(1, min(len(y) - 1, int(np.ceil(self.gamma * len(y)))))
        good_idx = order[:n_good]
        bad_idx = order[n_good:]

        if len(bad_idx) == 0:
            bad_idx = good_idx

        self.good_kde = KernelDensity(bandwidth=self.bandwidth).fit(X[good_idx])
        self.bad_kde = KernelDensity(bandwidth=self.bandwidth).fit(X[bad_idx])
        return self

    def score_density_ratio(self, X):
        """Return log p(X | good) - log p(X | bad), higher is better."""
        X = np.asarray(X, dtype=float)
        log_good = self.good_kde.score_samples(X)
        log_bad = self.bad_kde.score_samples(X)
        return log_good - log_bad

    def predict(self, X, return_std=False):
        """Compatibility scoring method; this model does not expose std."""
        if return_std:
            raise ValueError(
                "KDE density-ratio scores are not predictive means/stds. "
                "Use score_density_ratio(X) or AGS acquisition='auto'."
            )
        return self.score_density_ratio(X)


class TPESurrogate(KDEDensityRatioSurrogate):
    """
    Deprecated compatibility wrapper.

    This implementation is a KDE density-ratio scorer, not a full
    tree-structured Parzen Estimator.
    """

    def __init__(self, gamma=0.25, bandwidth=1.0):
        warnings.warn(
            "TPESurrogate is a legacy name for a KDE density-ratio model, "
            "not a full TPE implementation. Use KDEDensityRatioSurrogate.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(gamma=gamma, bandwidth=bandwidth)


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


def make_surrogate(
    surrogate_type,
    tpe_gamma=0.25,
    tpe_bandwidth=1.0,
    random_state=None,
    rf_n_estimators=200,
):
    """Create an RF, GP, or KDE density-ratio surrogate."""
    if surrogate_type == "rf":
        return RandomForestSurrogate(
            n_estimators=rf_n_estimators,
            random_state=random_state,
        )
    if surrogate_type == "gp":
        return build_gp_surrogate(random_state=random_state)
    if surrogate_type == "kde":
        return KDEDensityRatioSurrogate(
            gamma=tpe_gamma,
            bandwidth=tpe_bandwidth,
        )
    if surrogate_type == "tpe":
        warnings.warn(
            "surrogate_type='tpe' is deprecated because this is a KDE "
            "density-ratio model, not a full TPE. Use surrogate_type='kde'.",
            DeprecationWarning,
            stacklevel=2,
        )
        return KDEDensityRatioSurrogate(
            gamma=tpe_gamma,
            bandwidth=tpe_bandwidth,
        )
    raise ValueError(
        "Unknown surrogate_type {!r}; expected 'rf', 'gp', or 'kde' "
        "('tpe' is a deprecated alias).".format(surrogate_type)
    )
