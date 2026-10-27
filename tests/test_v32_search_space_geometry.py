import numpy as np
from sklearn.dummy import DummyClassifier

from ags import (
    AdaptiveGreedySearch,
    CategoricalParameter,
    NumericParameter,
    OrdinalParameter,
)


def _search(param_grid, random_state=7):
    return AdaptiveGreedySearch(
        DummyClassifier(),
        param_grid,
        cv=2,
        scoring="accuracy",
        n_climbers=2,
        n_jobs=1,
        max_evaluations=4,
        enable_pruning=False,
        random_state=random_state,
    )


def test_legacy_grids_infer_numeric_and_categorical_types():
    search = _search({
        "constant": [0.0, 1.0, 10.0],
        "strategy": ["most_frequent", "prior"],
    })

    assert isinstance(search.parameters[0], NumericParameter)
    assert isinstance(search.parameters[1], CategoricalParameter)


def test_categorical_distance_is_invariant_to_list_order():
    p1 = CategoricalParameter(["gini", "entropy", "log_loss"])
    p2 = CategoricalParameter(["log_loss", "gini", "entropy"])

    gini_1 = p1.values.index("gini")
    entropy_1 = p1.values.index("entropy")
    gini_2 = p2.values.index("gini")
    entropy_2 = p2.values.index("entropy")

    assert p1.distance(gini_1, entropy_1) == 1.0
    assert p2.distance(gini_2, entropy_2) == 1.0


def test_uneven_numeric_grid_uses_value_space_not_index_space():
    p = NumericParameter([0.001, 0.01, 1.0, 100.0])

    small_gap = p.distance(0, 1)
    large_gap = p.distance(2, 3)

    assert small_gap > 0
    assert large_gap > small_gap
    assert not np.isclose(small_gap, large_gap)


def test_log_numeric_geometry_matches_orders_of_magnitude():
    p = NumericParameter([0.001, 0.01, 0.1, 1.0], scale="log")

    d01 = p.distance(0, 1)
    d12 = p.distance(1, 2)
    d23 = p.distance(2, 3)

    assert np.isclose(d01, d12)
    assert np.isclose(d12, d23)


def test_ordinal_uses_rank_geometry():
    p = OrdinalParameter(["low", "medium", "high", "very_high"])

    assert np.isclose(p.distance(0, 1), 1 / 3)
    assert np.isclose(p.distance(0, 3), 1.0)


def test_semantic_neighbors_include_all_categorical_alternatives():
    search = _search({
        "strategy": CategoricalParameter(
            ["most_frequent", "prior", "stratified"]
        ),
        "random_state": NumericParameter([0, 1, 2]),
    })

    state = (0, 1)
    neighbors = set(search.get_neighbors(state))

    # categorical dimension: all other categories
    assert (1, 1) in neighbors
    assert (2, 1) in neighbors

    # numeric dimension: adjacent ordered values only
    assert (0, 0) in neighbors
    assert (0, 2) in neighbors

    assert len(neighbors) == 4


def test_visited_states_are_filtered_from_semantic_neighbors():
    search = _search({
        "strategy": CategoricalParameter(
            ["most_frequent", "prior", "stratified"]
        )
    })

    search.visited_states.add((1,))
    neighbors = set(search.get_neighbors((0,)))

    assert (1,) not in neighbors
    assert (2,) in neighbors


def test_state_to_features_uses_numeric_rank_and_one_hot_semantics():
    search = _search({
        "constant": NumericParameter([0.0, 10.0, 100.0]),
        "quality": OrdinalParameter(["low", "medium", "high"]),
        "strategy": CategoricalParameter(
            ["most_frequent", "prior", "stratified"]
        ),
    })

    features = search.state_to_features((1, 1, 2))

    # numeric = 10 / 100, ordinal rank = 1 / 2, categorical = one-hot third
    assert np.allclose(features, [0.1, 0.5, 0.0, 0.0, 1.0])


def test_mixed_distance_uses_parameter_semantics():
    search = _search({
        "constant": NumericParameter([0.0, 10.0, 100.0]),
        "strategy": CategoricalParameter(["most_frequent", "prior"]),
    })

    # Numeric contribution: 10/100 = 0.1. Categorical contribution: 1.
    assert np.isclose(search.distance((0, 0), (1, 1)), 1.1)


def test_max_min_seeding_uses_typed_distance():
    search = _search({
        "constant": NumericParameter([0.0, 1.0, 100.0]),
        "strategy": CategoricalParameter(["most_frequent", "prior"]),
    })

    seeds = search._get_max_min_seeds(4)

    assert len(seeds) == 4
    assert len(set(seeds)) == 4

    pairwise = [
        search.distance(seeds[i], seeds[j])
        for i in range(len(seeds))
        for j in range(i + 1, len(seeds))
    ]
    assert all(d > 0 for d in pairwise)


def test_categorical_reordering_preserves_semantic_pair_distance_in_search():
    a = _search({
        "strategy": CategoricalParameter(
            ["most_frequent", "prior", "stratified"]
        )
    })
    b = _search({
        "strategy": CategoricalParameter(
            ["stratified", "most_frequent", "prior"]
        )
    })

    a_most = a.values[0].index("most_frequent")
    a_prior = a.values[0].index("prior")
    b_most = b.values[0].index("most_frequent")
    b_prior = b.values[0].index("prior")

    assert a.distance((a_most,), (a_prior,)) == b.distance(
        (b_most,), (b_prior,)
    )
