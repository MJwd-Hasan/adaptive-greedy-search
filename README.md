# Adaptive Greedy Search (AGS)

**Adaptive Greedy Search (AGS) v3.7.0** is a surrogate-assisted, multi-start adaptive local-search optimizer for finite hyperparameter spaces.

AGS combines typed and conditional search-space geometry, RF/GP/KDE surrogate guidance, adaptive trust-region climbers, confidence-bound multi-fidelity racing, diversity-aware batch search, and **lazy candidate generation**.

```python
from ags import AdaptiveGreedySearch
```

## What's new in 3.7

### Conditional hyperparameters

AGS can now represent parameters that are active only for selected values of an earlier parent parameter.

```python
from ags import (
    AdaptiveGreedySearch,
    CategoricalParameter,
    ConditionalParameter,
    IntegerParameter,
    NumericParameter,
)
from sklearn.svm import SVC

param_grid = {
    "kernel": CategoricalParameter(["linear", "rbf", "poly"]),
    "gamma": ConditionalParameter(
        NumericParameter([0.001, 0.01, 0.1, 1.0], scale="log"),
        parent="kernel",
        active_values=["rbf", "poly"],
    ),
    "degree": ConditionalParameter(
        IntegerParameter([2, 3, 4, 5]),
        parent="kernel",
        active_values=["poly"],
    ),
}

search = AdaptiveGreedySearch(
    SVC(),
    param_grid,
    max_evaluations=30,
)
```

For a state with `kernel="linear"`, `gamma` and `degree` are inactive and are **not passed to the estimator**. For `kernel="rbf"`, `gamma` is active but `degree` is omitted. For `kernel="poly"`, both are active.

Conditional parents must currently appear before their children in `param_grid` insertion order.

### Lazy candidate generation

AGS no longer constructs:

```python
list(itertools.product(...))
```

for the entire search space.

Instead, global initialization, space-filling search, and restart logic operate on bounded candidate pools generated lazily:

```python
search = AdaptiveGreedySearch(
    estimator,
    param_grid,
    candidate_pool_size=256,
)
```

This changes memory growth from being directly tied to the full Cartesian-product size to being tied primarily to the candidate pool, visited observations, and local neighborhoods.

`search.search_space_size_upper_bound` reports the raw Cartesian upper bound, while `search.lazy_search` indicates that the complete grid is not materialized.

### Conditional-aware geometry and surrogate encoding

Inactive dimensions use an internal inactive state and an explicit activity mask in surrogate features.

Distance behaves as follows:

- inactive in both states → no contribution;
- active in one state but inactive in the other → activity mismatch penalty;
- active in both → the wrapped parameter's normal numeric / ordinal / categorical distance.

Local neighbors are canonicalized after parent changes, so the search does not intentionally propose invalid conditional configurations.

## Earlier 3.x improvements

### v3.4 — Adaptive trust-region climbers

Each climber tracks its current state separately from its personal best and adapts `trust_radius` after success or failure.

### v3.5 — Multi-fidelity confidence racing

`pruning_strategy="racing"` treats CV folds as fidelity and can stop statistically weak candidates before full CV.

### v3.6 — Diversity-aware batch search and restart

Parallel proposals and respawns balance model quality with typed-distance diversity.

## Installation

```bash
pip install adaptive-greedy-search
```

## Quickstart

```python
from ags import AdaptiveGreedySearch
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(
    n_samples=500,
    n_features=20,
    random_state=0,
)

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
    candidate_pool_size=256,
    random_state=42,
)

search.fit(X, y)

print("Best parameters:", search.best_params)
print("Best CV score:", search.best_score)
print("Evaluations:", search.n_evaluations)
print("Folds saved:", search.folds_saved)
print("Mean fidelity:", search.mean_fidelity)
```

## Search-space API

Plain Python lists remain supported.

Explicit parameter types:

| Type | Purpose |
|---|---|
| `NumericParameter` | Numeric values with linear or log geometry |
| `FloatParameter` | Explicit real-valued finite grid |
| `IntegerParameter` | Explicit integer finite grid |
| `OrdinalParameter` | Ordered choices using rank distance |
| `CategoricalParameter` | Nominal values using Hamming geometry |
| `ConditionalParameter` | Wraps another parameter and activates it only for selected parent values |

All values are still explicitly enumerated. AGS 3.7 does **not** turn these classes into continuous-domain optimizers.

## Surrogates and acquisitions

| Surrogate | Meaning |
|---|---|
| `rf` | Default. Random Forest mean + between-tree dispersion proxy |
| `gp` | Gaussian Process predictive mean/std |
| `kde` | Good/bad KDE density-ratio ranking |
| `tpe` | Deprecated compatibility alias for `kde`; not a full TPE |

Acquisition options are `auto`, `ucb`, `ei` / `expected_improvement`, and `density_ratio` where compatible.

Only **completed full-CV evaluations** are used as complete surrogate training targets.

## Pruning / racing

| Strategy | Meaning |
|---|---|
| `percentile` | Heuristic comparison with historical partial means |
| `optimistic` | Safe only with a valid explicit `score_upper_bound` |
| `racing` | Confidence-bound statistical early stopping |
| `none` | Full CV for every selected configuration |

Partial observations never become the reported winner.

## Important 3.7 parameters and attributes

- `candidate_pool_size=256`: maximum target size for lazily generated global candidate pools.
- `search_space_size_upper_bound`: raw product-size upper bound.
- `lazy_search`: `True` when using the structured lazy engine.
- `has_conditional_parameters`: whether the configured search space contains conditionals.
- `best_params`, `best_score`, `best_state`
- `completed_evaluations`, `partial_evaluations`, `history`
- `folds_saved`, `mean_fidelity`
- `trust_expansions`, `trust_contractions`
- `climbers_spawned`, `climbers_merged`, `plateau_escapes`

## Current limitations

- Search values are finite and explicitly listed.
- Conditional dependencies are currently parent → child and the parent must appear earlier in the mapping.
- Lazy candidate pools reduce full-grid memory pressure, but they do not guarantee exhaustive coverage of huge spaces.
- AGS remains a heuristic optimizer and does not guarantee the global optimum.

## License

MIT © Mohammad Jawad Hasan
