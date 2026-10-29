import numpy as np
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from ags import (
    AdaptiveGreedySearch,
    CategoricalParameter,
    ConditionalParameter,
    IntegerParameter,
    NumericParameter,
)


def _conditional_grid():
    return {
        "kernel": CategoricalParameter(["linear", "rbf", "poly"]),
        "gamma": ConditionalParameter(
            NumericParameter([0.01, 0.1]),
            parent="kernel",
            active_values=["rbf", "poly"],
        ),
        "degree": ConditionalParameter(
            IntegerParameter([2, 3]),
            parent="kernel",
            active_values=["poly"],
        ),
    }


def test_v37_conditional_state_omits_inactive_estimator_params():
    search = AdaptiveGreedySearch(
        SVC(),
        _conditional_grid(),
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
        enable_pruning=False,
    )

    assert search.state_to_params((0, -1, -1)) == {"kernel": "linear"}
    assert search.state_to_params((1, 1, -1)) == {
        "kernel": "rbf",
        "gamma": 0.1,
    }
    assert search.state_to_params((2, 0, 1)) == {
        "kernel": "poly",
        "gamma": 0.01,
        "degree": 3,
    }


def test_v37_conditional_neighbors_are_always_valid():
    search = AdaptiveGreedySearch(
        SVC(),
        _conditional_grid(),
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
        enable_pruning=False,
    )

    linear = (0, -1, -1)
    neighbors = search.get_neighbors(linear)

    assert neighbors
    assert all(search.search_space.is_valid(state) for state in neighbors)

    params = [search.state_to_params(state) for state in neighbors]
    assert any(item["kernel"] == "rbf" and "gamma" in item for item in params)
    assert any(
        item["kernel"] == "poly"
        and "gamma" in item
        and "degree" in item
        for item in params
    )


def test_v37_conditional_encoding_has_stable_width_and_active_masks():
    search = AdaptiveGreedySearch(
        SVC(),
        _conditional_grid(),
        n_climbers=1,
        n_jobs=1,
        max_evaluations=4,
        enable_pruning=False,
    )

    linear = search.state_to_features((0, -1, -1))
    rbf = search.state_to_features((1, 0, -1))
    poly = search.state_to_features((2, 0, 0))

    assert len(linear) == len(rbf) == len(poly)
    assert linear != rbf
    assert rbf != poly


def test_v37_lazy_space_does_not_materialize_cartesian_product():
    huge_grid = {
        f"p{i}": list(range(100))
        for i in range(6)
    }
    search = AdaptiveGreedySearch(
        DummyClassifier(),
        huge_grid,
        n_climbers=2,
        n_jobs=1,
        max_evaluations=10,
        enable_pruning=False,
        candidate_pool_size=16,
    )

    assert search.search_space_size_upper_bound == 100 ** 6
    assert search.lazy_search is True
    assert not hasattr(search, "all_states")

    pool = search._candidate_pool(size=16)
    assert 1 <= len(pool) <= 16
    assert len(set(pool)) == len(pool)


def test_v37_lazy_iterator_generates_only_valid_conditional_states():
    search = AdaptiveGreedySearch(
        SVC(),
        _conditional_grid(),
        n_climbers=1,
        n_jobs=1,
        max_evaluations=7,
        enable_pruning=False,
    )

    states = list(search.search_space.iter_states())
    assert len(states) == 7
    assert len(set(states)) == 7
    assert all(search.search_space.is_valid(state) for state in states)


def test_v37_end_to_end_conditional_search():
    X, y = make_classification(
        n_samples=80,
        n_features=6,
        n_informative=4,
        random_state=4,
    )
    search = AdaptiveGreedySearch(
        SVC(),
        _conditional_grid(),
        cv=3,
        scoring="accuracy",
        n_climbers=2,
        n_jobs=1,
        max_evaluations=6,
        enable_pruning=False,
        min_observations=3,
        candidate_pool_size=32,
        random_state=9,
    )
    search.fit(X, y)

    assert search.best_state in search.completed_evaluations
    assert search.has_conditional_parameters is True
    assert search.n_evaluations <= 6
    assert search.search_space.is_valid(search.best_state)

    best = search.best_params
    if best["kernel"] == "linear":
        assert "gamma" not in best
        assert "degree" not in best
    elif best["kernel"] == "rbf":
        assert "gamma" in best
        assert "degree" not in best
    else:
        assert "gamma" in best
        assert "degree" in best


def test_v371_profiling_report_exposes_internal_timings():
    X, y = make_classification(
        n_samples=60,
        n_features=5,
        n_informative=3,
        random_state=17,
    )
    search = AdaptiveGreedySearch(
        DummyClassifier(strategy="most_frequent"),
        {"strategy": ["most_frequent", "prior", "stratified"]},
        cv=3,
        scoring="accuracy",
        n_climbers=1,
        n_jobs=1,
        max_evaluations=3,
        enable_pruning=False,
        min_observations=2,
        candidate_pool_size=8,
        random_state=3,
        profiling=True,
    )
    search.fit(X, y)
    report = search.profile_report(print_report=False)

    assert report["enabled"] is True
    assert report["total_wall_time"] > 0
    assert report["n_evaluations"] > 0
    assert report["evaluation_batches"] > 0
    assert report["worker_compute_sum"] >= 0
    sections = {row["section"] for row in report["sections"]}
    assert "proposal_generation" in sections
    assert "evaluation_batch_wall" in sections


def test_v371_profiling_disabled_keeps_timing_sections_empty():
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=8,
    )
    search = AdaptiveGreedySearch(
        DummyClassifier(),
        {"strategy": ["most_frequent", "prior"]},
        cv=2,
        n_climbers=1,
        n_jobs=1,
        max_evaluations=2,
        enable_pruning=False,
        candidate_pool_size=8,
        profiling=False,
    )
    search.fit(X, y)
    report = search.profile_report(print_report=False)

    assert report["enabled"] is False
    assert report["sections"] == []
