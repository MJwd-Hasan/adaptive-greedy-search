# Adaptive Greedy Search (AGS)

**Adaptive Greedy Search (AGS) v3.6.0** is a surrogate-assisted, multi-start adaptive local-search optimizer for finite discrete hyperparameter spaces.

It combines typed search-space geometry, a shared RF/GP/KDE surrogate, adaptive trust-region climbers, fold-level multi-fidelity racing, and diversity-aware batch/restart logic.

```python
from ags import AdaptiveGreedySearch
```

## What's new in 3.6

### v3.4 — Adaptive trust-region climbers

Each climber now keeps separate current and historical-best state:

- `current_state`, `current_score`
- `personal_best_state`, `personal_best_score`
- `trust_radius`

Confirmed improvement expands the local region; failed or pruned proposals contract it.

```python
search = AdaptiveGreedySearch(
    estimator,
    param_grid,
    initial_trust_radius=1.0,
    min_trust_radius=1.0,
    trust_expand=2.0,
    trust_shrink=0.5,
    max_neighbor_radius=4,
)
```

### v3.5 — Multi-fidelity confidence racing

CV folds are treated as fidelity. In addition to `percentile`, `optimistic`, and `none`, AGS now supports:

```python
pruning_strategy="racing"
racing_confidence=0.95
```

After enough folds, a statistically weak candidate can be stopped when its confidence upper bound falls below the confirmed incumbent. Racing is a statistical resource-allocation rule, **not a deterministic guarantee**. Partial observations still cannot become the winner or train the surrogate as full observations.

### v3.6 — Diversity-aware batch search and restart

Batch candidates are coordinated before evaluation. AGS adds a typed-distance diversity bonus to acquisition scores so parallel climbers are less likely to collapse into the same basin.

Retired climbers are respawned using both predicted quality and distance from active search regions.

```python
search = AdaptiveGreedySearch(
    estimator,
    param_grid,
    diversity_weight=0.25,
    restart_diversity_weight=0.5,
)
```

## Installation

```bash
pip install adaptive-greedy-search
```

## Quickstart

```python
from ags import AdaptiveGreedySearch
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=20, random_state=0)

param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 5, 7, 9, None],
    "min_samples_split": [2, 4, 6, 8],
}

search = AdaptiveGreedySearch(
    RandomForestClassifier(random_state=0),
    param_grid,
    cv=5,
    scoring="accuracy",
    max_evaluations=30,
    n_climbers=6,
    n_jobs=-1,
    surrogate_type="rf",
    acquisition="auto",
    pruning_strategy="racing",
    random_state=42,
)

search.fit(X, y)

print(search.best_params)
print(search.best_score)
print("evaluations:", search.n_evaluations)
print("folds saved:", search.folds_saved)
print("mean fidelity:", search.mean_fidelity)
print("trust expansions:", search.trust_expansions)
print("trust contractions:", search.trust_contractions)
```

## Typed search spaces

Plain lists still work. For explicit geometry:

```python
from ags import NumericParameter, OrdinalParameter, CategoricalParameter

param_grid = {
    "C": NumericParameter([0.01, 0.1, 1, 10], scale="log"),
    "depth": OrdinalParameter([2, 4, 6, 8]),
    "kernel": CategoricalParameter(["linear", "rbf", "poly"]),
}
```

Numeric distance is normalized in value space, ordinal distance uses rank, and nominal categories use Hamming distance plus one-hot surrogate features.

## Surrogates and acquisitions

| Surrogate | Meaning |
|---|---|
| `rf` | Default. Random Forest mean + between-tree dispersion proxy |
| `gp` | Gaussian Process predictive mean/std |
| `kde` | Good/bad KDE density-ratio ranking |
| `tpe` | Deprecated compatibility alias for `kde`; not a full TPE |

Acquisition options are `auto`, `ucb`, `ei` / `expected_improvement`, and `density_ratio` where compatible.

The surrogate is trained **only on complete CV evaluations** after the warm-up target is reached.

## Pruning / racing

| Strategy | Meaning |
|---|---|
| `percentile` | Heuristic comparison with historical partial means |
| `optimistic` | Deterministic bound only when a valid `score_upper_bound` is supplied |
| `racing` | Confidence-bound statistical early stopping |
| `none` | Full CV for every selected configuration |

## Important result attributes

- `best_params`, `best_score`, `best_state`
- `completed_evaluations`, `partial_evaluations`, `history`
- `n_evaluations`, `n_completed_evaluations`, `n_partial_evaluations`
- `n_pruned`, `folds_saved`, `mean_fidelity`
- `trust_expansions`, `trust_contractions`, `plateau_escapes`
- `climbers_spawned`, `climbers_merged`

## Limitations

AGS still materializes the finite Cartesian product. Conditional spaces and lazy candidate generation are planned for v4.0. AGS is not exhaustive and does not guarantee the global optimum.

## License

MIT © Mohammad Jawad Hasan
