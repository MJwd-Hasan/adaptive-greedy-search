import warnings

import numpy as np
from joblib import Parallel
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from ags import AdaptiveGreedySearch


def _balanced_binary_data():
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.array([0, 1] * 20)
    return X, y


def test_optimistic_pruning_requires_explicit_upper_bound():
    X, y = _balanced_binary_data()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        search = AdaptiveGreedySearch(
            DummyClassifier(),
            {"strategy": ["most_frequent"]},
            cv=4,
            scoring="accuracy",
            n_climbers=1,
            n_jobs=1,
            max_evaluations=1,
            enable_pruning=True,
            pruning_strategy="optimistic",
            score_upper_bound=None,
        )

    assert search.enable_pruning is False
    assert any("explicit valid score_upper_bound" in str(w.message) for w in caught)


def test_pruned_result_is_partial_not_completed():
    X, y = _balanced_binary_data()

    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {"strategy": ["most_frequent"]},
        cv=4,
        scoring="accuracy",
        n_climbers=1,
        n_jobs=1,
        max_evaluations=1,
        enable_pruning=True,
        pruning_strategy="optimistic",
        min_folds_before_pruning=1,
        score_upper_bound=1.0,
    )

    search._cv_splits = list(
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(X, y)
    )

    # A confirmed incumbent may drive pruning; a partial result may not.
    search.completed_evaluations[(99,)] = 0.99

    with Parallel(n_jobs=1) as parallel:
        results = search.evaluate_batch(
            {"climber_0": (0,)},
            X,
            y,
            parallel,
        )

    state, score, completed = results["climber_0"]

    assert state == (0,)
    assert completed is False
    assert score is None
    assert state in search.visited_states
    assert state in search.partial_evaluations
    assert state not in search.completed_evaluations
    assert state not in search.evaluated

    obs = search.partial_evaluations[state]
    assert obs["completed"] is False
    assert obs["pruned"] is True
    assert 0.0 < obs["fidelity"] < 1.0
    assert obs["score"] is None
    assert obs["partial_mean"] is not None
    assert len(obs["fold_scores"]) == obs["n_folds_used"]


def test_fit_reports_only_full_cv_winner():
    X, y = _balanced_binary_data()

    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {
            "strategy": ["most_frequent", "prior", "stratified"],
            "random_state": [0, 1],
        },
        cv=4,
        scoring="accuracy",
        n_climbers=2,
        n_jobs=1,
        max_evaluations=4,
        enable_pruning=False,
        random_state=7,
    )

    search.fit(X, y)

    assert search.best_state in search.completed_evaluations
    assert search.best_state not in search.partial_evaluations
    assert search.best_score == search.completed_evaluations[search.best_state]
    assert search.n_evaluations == len(search.visited_states)
    assert search.n_completed_evaluations == len(search.completed_evaluations)
    assert search.n_partial_evaluations == len(search.partial_evaluations)
    assert all(h["completed"] for h in search.history)


def test_surrogate_training_excludes_partial_observations():
    X, y = _balanced_binary_data()

    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {"strategy": ["most_frequent", "prior", "stratified", "uniform"]},
        cv=4,
        scoring="accuracy",
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
        enable_pruning=False,
        surrogate_type="gp",
    )

    search.completed_evaluations.update({
        (0,): 0.50,
        (1,): 0.55,
        (2,): 0.60,
    })
    search.partial_evaluations[(3,)] = {
        "state": (3,),
        "partial_mean": 0.99,
        "fidelity": 0.25,
        "completed": False,
        "pruned": True,
    }

    captured = {}

    class RecordingSurrogate:
        def fit(self, X_train, y_train):
            captured["X"] = np.array(X_train)
            captured["y"] = np.array(y_train)
            return self

    search.surrogate = RecordingSurrogate()
    search.fit_surrogate()

    assert captured["X"].shape[0] == 3
    assert set(map(tuple, captured["X"].astype(int))) == {(0,), (1,), (2,)}
    assert 0.99 not in captured["y"]
