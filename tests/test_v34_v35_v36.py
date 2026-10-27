import numpy as np
from joblib import Parallel
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from ags import AdaptiveGreedySearch
from ags.core import Climber


def _data():
    X = np.arange(80, dtype=float).reshape(-1, 1)
    y = np.array([0, 1] * 40)
    return X, y


def _search(**kwargs):
    options = dict(
        estimator=DummyClassifier(),
        param_grid={
            "strategy": ["most_frequent", "prior", "stratified", "uniform"],
            "random_state": [0, 1, 2],
        },
        cv=4,
        scoring="accuracy",
        n_climbers=2,
        n_jobs=1,
        max_evaluations=8,
        enable_pruning=False,
        min_observations=3,
        rf_n_estimators=20,
        random_state=7,
    )
    options.update(kwargs)
    return AdaptiveGreedySearch(**options)


def test_v34_climber_tracks_current_and_personal_best_separately():
    search = _search(
        initial_trust_radius=1.0,
        max_neighbor_radius=4,
        trust_expand=2.0,
        trust_shrink=0.5,
    )
    climber = Climber(id="c", current_state=(0, 0), trust_radius=1.0)
    search.climbers = [climber]
    search._climber_by_id = {"c": climber}

    search._update_climbers({"c": ((1, 0), 0.70, True)})
    assert climber.current_state == (1, 0)
    assert climber.current_score == 0.70
    assert climber.personal_best_state == (1, 0)
    assert climber.personal_best_score == 0.70
    assert climber.trust_radius == 2.0
    assert search.trust_expansions == 1

    search._update_climbers({"c": ((2, 0), 0.60, True)})
    assert climber.current_state == (2, 0)
    assert climber.current_score == 0.60
    assert climber.personal_best_state == (1, 0)
    assert climber.personal_best_score == 0.70
    assert climber.trust_radius == 1.0
    assert search.trust_contractions == 1


def test_v34_trust_region_reaches_farther_states_when_expanded():
    search = _search()
    search.visited_states.add((0, 0))

    radius_one = set(search._states_within_hops((0, 0), 1))
    radius_two = set(search._states_within_hops((0, 0), 2))

    assert radius_one
    assert radius_one.issubset(radius_two)
    assert len(radius_two) > len(radius_one)


def test_v35_racing_prunes_statistically_weak_candidate():
    X, y = _data()
    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {"strategy": ["most_frequent"]},
        cv=4,
        scoring="accuracy",
        n_climbers=1,
        n_jobs=1,
        max_evaluations=1,
        enable_pruning=True,
        pruning_strategy="racing",
        min_folds_before_pruning=2,
        racing_confidence=0.95,
    )
    search._cv_splits = list(
        StratifiedKFold(n_splits=4, shuffle=True, random_state=42).split(X, y)
    )
    search.completed_evaluations[(99,)] = 0.99

    with Parallel(n_jobs=1) as parallel:
        results = search.evaluate_batch({"climber_0": (0,)}, X, y, parallel)

    _, score, completed = results["climber_0"]
    obs = search.partial_evaluations[(0,)]

    assert completed is False
    assert score is None
    assert obs["prune_reason"] == "confidence_racing"
    assert 0 < obs["fidelity"] < 1


def test_v36_diversity_bonus_prefers_farther_candidate_when_quality_ties():
    search = _search(diversity_weight=1.0)
    search.completed_evaluations.update({
        (0, 0): 0.5,
        (3, 2): 0.6,
        (1, 1): 0.55,
    })
    search.fit_surrogate()

    # Isolate the diversity term by forcing equal acquisition scores.
    search.acquisition_scores = lambda states: np.zeros(len(states), dtype=float)
    candidates = [(0, 1), (3, 2)]
    chosen = search._select_model_candidate(
        candidates,
        references=[(0, 0)],
        weight=1.0,
    )
    assert chosen == (3, 2)


def test_v36_end_to_end_reports_adaptive_metrics():
    X, y = _data()
    search = _search(
        max_evaluations=7,
        initial_trust_radius=1.0,
        diversity_weight=0.2,
        restart_diversity_weight=0.4,
    )
    search.fit(X, y)

    assert search.best_state in search.completed_evaluations
    assert 0.0 < search.mean_fidelity <= 1.0
    assert search.trust_expansions >= 0
    assert search.trust_contractions >= 0
    assert len(search.visited_states) == search.n_evaluations
