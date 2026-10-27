import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from ags import (
    AdaptiveGreedySearch,
    KDEDensityRatioSurrogate,
    RandomForestSurrogate,
)
from ags.acquisition import acquisition_scores
from ags.surrogates import make_surrogate


class _FixedPredictiveSurrogate:
    def predict(self, X, return_std=True):
        mean = np.array([0.4, 0.8], dtype=float)
        std = np.array([0.1, 0.0], dtype=float)
        if return_std:
            return mean, std
        return mean


def _balanced_data():
    X = np.arange(60, dtype=float).reshape(-1, 1)
    y = np.array([0, 1] * 30)
    return X, y


def test_random_forest_surrogate_exposes_mean_and_tree_dispersion():
    X = np.array([[0.0], [0.1], [0.8], [1.0]])
    y = np.array([0.0, 0.1, 0.9, 1.0])
    model = RandomForestSurrogate(n_estimators=20, random_state=3).fit(X, y)

    mean, std = model.predict(np.array([[0.05], [0.95]]), return_std=True)

    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(std))
    assert np.all(std >= 0)


def test_ucb_acquisition_is_separate_from_model():
    model = _FixedPredictiveSurrogate()
    X = np.zeros((2, 1))

    scores = acquisition_scores(
        model,
        X,
        acquisition="ucb",
        exploration_weight=2.0,
        best_score=0.5,
    )

    assert np.allclose(scores, [0.6, 0.8])


def test_expected_improvement_uses_incumbent_and_handles_zero_spread():
    model = _FixedPredictiveSurrogate()
    scores = acquisition_scores(
        model,
        np.zeros((2, 1)),
        acquisition="ei",
        best_score=0.5,
    )

    # Candidate 2 has deterministic mean 0.8 and zero spread, so EI = 0.3.
    assert np.isfinite(scores).all()
    assert np.isclose(scores[1], 0.3)
    assert scores[0] >= 0


def test_density_ratio_surrogate_does_not_claim_predictive_std():
    X = np.array([[0.0], [0.2], [0.8], [1.0]])
    y = np.array([0.9, 0.8, 0.2, 0.1])
    model = KDEDensityRatioSurrogate(gamma=0.5, bandwidth=0.3).fit(X, y)

    score = model.score_density_ratio(np.array([[0.1], [0.9]]))
    assert score.shape == (2,)
    assert np.all(np.isfinite(score))

    with pytest.raises(ValueError, match="not predictive means/stds"):
        model.predict(np.array([[0.1]]), return_std=True)


def test_kde_density_ratio_is_default_acquisition_for_kde():
    model = KDEDensityRatioSurrogate(gamma=0.5, bandwidth=0.3).fit(
        np.array([[0.0], [0.2], [0.8], [1.0]]),
        np.array([0.9, 0.8, 0.2, 0.1]),
    )
    candidates = np.array([[0.1], [0.9]])
    automatic = acquisition_scores(model, candidates, acquisition="auto")
    explicit = acquisition_scores(model, candidates, acquisition="density_ratio")

    assert np.allclose(automatic, explicit)


def test_ei_is_rejected_for_density_ratio_surrogate():
    model = KDEDensityRatioSurrogate(gamma=0.5, bandwidth=0.3).fit(
        np.array([[0.0], [0.2], [0.8], [1.0]]),
        np.array([0.9, 0.8, 0.2, 0.1]),
    )
    with pytest.raises(ValueError, match="does not expose predictive"):
        acquisition_scores(
            model, np.array([[0.3]]), acquisition="ei", best_score=0.9
        )


def test_legacy_tpe_factory_alias_warns_and_returns_density_ratio_model():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = make_surrogate("tpe")

    assert isinstance(model, KDEDensityRatioSurrogate)
    assert any("not a full TPE" in str(item.message) for item in caught)


def test_default_surrogate_and_acquisition_are_rf_and_ucb():
    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {"strategy": ["most_frequent", "prior", "stratified"]},
        cv=3,
        n_climbers=1,
        n_jobs=1,
        max_evaluations=3,
    )

    assert search.surrogate_type == "rf"
    assert search.acquisition == "auto"
    assert isinstance(search.surrogate, RandomForestSurrogate)


def test_warmup_default_is_based_on_parameter_count_and_grid_size():
    small = AdaptiveGreedySearch(
        DummyClassifier(),
        {
            "strategy": ["most_frequent", "prior"],
            "random_state": [0, 1, 2],
        },
        cv=2,
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
    )

    assert small.min_observations == 6  # min(10, 2 * 3)

    larger = AdaptiveGreedySearch(
        DummyClassifier(),
        {
            "strategy": ["most_frequent", "prior", "stratified", "uniform"],
            "random_state": list(range(20)),
        },
        cv=2,
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
    )

    assert larger.min_observations == 10


def test_warmup_can_be_configured_and_surrogate_is_not_ready_early():
    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {"strategy": ["most_frequent", "prior", "stratified", "uniform"]},
        cv=2,
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
        min_observations=3,
    )

    search.completed_evaluations.update({
        (0,): 0.5,
        (1,): 0.6,
    })
    assert search._surrogate_ready() is False

    search.completed_evaluations[(2,)] = 0.7
    search.fit_surrogate()
    assert search._surrogate_ready() is True


@pytest.mark.parametrize("acquisition", ["ucb", "ei"])
def test_search_executes_with_random_forest_acquisitions(acquisition):
    X, y = _balanced_data()
    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {
            "strategy": ["most_frequent", "prior", "stratified", "uniform"],
            "random_state": [0, 1],
        },
        cv=3,
        scoring="accuracy",
        n_climbers=2,
        n_jobs=1,
        max_evaluations=6,
        pruning_strategy="none",
        surrogate_type="rf",
        acquisition=acquisition,
        min_observations=3,
        rf_n_estimators=20,
        random_state=5,
    )
    search.fit(X, y)

    assert search._surrogate_fitted
    assert search.best_state in search.completed_evaluations
    assert search.n_completed_evaluations >= 3


def test_search_executes_with_kde_density_ratio_mode():
    X, y = _balanced_data()
    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {"strategy": ["most_frequent", "prior", "stratified", "uniform"]},
        cv=3,
        scoring="accuracy",
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
        pruning_strategy="none",
        surrogate_type="kde",
        acquisition="auto",
        min_observations=2,
        random_state=4,
    )
    search.fit(X, y)

    assert search._surrogate_fitted
    assert search.best_state in search.completed_evaluations


def test_incompatible_acquisition_is_rejected_at_construction():
    with pytest.raises(ValueError, match="does not expose predictive"):
        AdaptiveGreedySearch(
            DummyClassifier(),
            {"strategy": ["most_frequent", "prior"]},
            surrogate_type="kde",
            acquisition="ucb",
            n_jobs=1,
        )
