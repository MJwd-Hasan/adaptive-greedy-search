# Adaptive Greedy Search (AGS)

**Adaptive Greedy Search (AGS) v3.3.0** is a surrogate-guided hyperparameter optimizer for finite, discrete search spaces. It combines a shared surrogate model, a swarm of local-search climbers, space-filling initialization, and optional fold-level cross-validation pruning.

[View the v3.3.0 release](https://github.com/MJwd-Hasan/adaptive-greedy-search/releases/tag/v3.3.0) · [Changelog](CHANGELOG.md)

```python
from ags import AdaptiveGreedySearch
```

## How AGS works

AGS is intended for finite hyperparameter grids that are too expensive to exhaustively evaluate, but where nearby configurations may have related performance.

- **Typed search-space geometry.** Numerical dimensions use normalized value-space distance (or log-distance when requested), ordinal dimensions use rank distance, and nominal categories use Hamming distance and one-hot features. Existing list-based grids remain supported.
- **A shared surrogate.** Climbers share one model trained on **completed, full-CV evaluations**. Random Forest is the default for mixed and discrete spaces; Gaussian Process (GP) and KDE density-ratio alternatives are also available.
- **Space-filling warm-up.** AGS starts with Max-Min / farthest-point seeding and keeps using space-filling proposals until the configured number of complete observations is available.
- **Multiple local-search climbers.** Each active climber proposes an unevaluated neighbor using the acquisition scores. Proposals are deduplicated before a batch is evaluated in parallel with `joblib`.
- **Plateau escape and restart.** Climbers can search outward through nearby rings when boxed in, and retired climbers can be respawned at promising unexplored states.
- **Optional fold-level pruning.** A candidate can stop before all CV folds are run when it is out of contention under the selected pruning rule. A pruned partial mean is never used as the official best score or as a complete surrogate observation.

Unlike exhaustive grid search, AGS does not guarantee that it will find the global best configuration. Its advantage depends on the evaluation budget, the structure of the performance landscape, the estimator's cost, and the selected pruning strategy.

**AGS may be a poor fit** when performance has little relationship to search-space proximity, the grid is very small, the budget is too small to support useful warm-up and local search, or you require an exhaustive optimum guarantee. AGS currently operates on a finite, materialized grid; conditional search spaces and lazy candidate generation are future roadmap work.

## Installation

```bash
pip install adaptive-greedy-search
```

Or install from a local clone:

```bash
git clone https://github.com/MJwd-Hasan/adaptive-greedy-search.git
cd adaptive-greedy-search
pip install -e .
```

Requires Python 3.8+ and installs `numpy`, `scikit-learn`, and `joblib` automatically.

## Quickstart

This example tunes a scikit-learn classifier on a finite grid:

```python
from ags import AdaptiveGreedySearch
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

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
    estimator=RandomForestClassifier(random_state=0),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    max_evaluations=30,
    n_climbers=6,
    n_jobs=-1,
    random_state=42,
)

search.fit(X, y)

print("Best parameters:", search.best_params)
print("Best CV score:", search.best_score)
print(
    "Folds saved:",
    f"{search.folds_saved}/{search.total_folds_possible}",
)
print(
    "Climbers spawned:", search.climbers_spawned,
    "| merged:", search.climbers_merged,
    "| plateau escapes:", search.plateau_escapes,
)
```

**Tip:** Choose `n_climbers` with the evaluation budget in mind. When left as `None`, AGS creates about 75% of the available CPU cores as climbers (at least one); this can be too many for a very small budget.

## Defining the search space

Plain Python lists are still supported:

```python
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "kernel": ["linear", "rbf"],
}
```

AGS automatically infers pure numeric lists as numerical parameters and other lists as categorical parameters. You can specify a type explicitly when its geometry matters:

```python
from ags import (
    AdaptiveGreedySearch,
    CategoricalParameter,
    NumericParameter,
    OrdinalParameter,
)

param_grid = {
    "C": NumericParameter([0.01, 0.1, 1, 10], scale="log"),
    "kernel": CategoricalParameter(["linear", "rbf"]),
    "degree": OrdinalParameter([2, 3, 4, 5]),
}
```

The example above uses an SVC-style parameter space. The parameter types behave as follows:

| Type | Geometry | Use it for |
|---|---|---|
| `NumericParameter(values)` | Normalized distance between numerical values | Integer/float grids where numeric gaps matter |
| `NumericParameter(values, scale="log")` | Normalized distance in log space; all values must be positive | Log-spaced quantities such as `C` or `learning_rate` |
| `OrdinalParameter(values)` | Normalized rank distance | Ordered choices where rank matters more than numeric gap |
| `CategoricalParameter(values)` | Different categories have distance 1; one-hot surrogate features | Nominal choices such as kernels or split criteria |

All parameter values must be finite and explicitly listed. A parameter type does not create a continuous search interval. Conditional parameters are not yet supported, so AGS forms the Cartesian product of the provided values.

## Surrogates and acquisition functions

Version 3.3 separates the surrogate model from the acquisition rule that scores candidate configurations.

### Surrogate choices

| `surrogate_type` | Meaning | Notes |
|---|---|---|
| `"rf"` (default) | Random Forest regression surrogate | Suitable default for mixed/discrete grids. Between-tree dispersion is an exploration proxy, **not** calibrated posterior uncertainty. |
| `"gp"` | Gaussian Process surrogate | Provides predictive mean and standard deviation; often useful for smaller numerical spaces. |
| `"kde"` | KDE density-ratio scoring | Ranks candidates by log-density ratio of good versus bad observations; it is not a predictive mean/std model. |
| `"tpe"` | Deprecated compatibility alias for `"kde"` | Kept for compatibility; this is **not** a full Tree-structured Parzen Estimator implementation. |

### Acquisition choices

| `acquisition` | Behavior | Compatible surrogates |
|---|---|---|
| `"auto"` (default) | Uses UCB for RF/GP and density-ratio scoring for KDE | RF, GP, KDE |
| `"ucb"` | Upper Confidence Bound-style score: predicted mean + `exploration_weight × spread` | RF, GP |
| `"ei"` or `"expected_improvement"` | Expected Improvement ranking | RF, GP |
| `"density_ratio"` | Uses the KDE good/bad log-density ratio | KDE only |

For Random Forest, both UCB's spread and the EI calculation rely on between-tree dispersion, so these rankings are approximations rather than calibrated Bayesian uncertainty calculations. With GP, Expected Improvement uses the GP predictive distribution.

Example:

```python
search = AdaptiveGreedySearch(
    estimator,
    param_grid,
    surrogate_type="rf",
    acquisition="ei",
    min_observations=12,
)
```

### Surrogate warm-up

AGS does not use surrogate-guided proposals immediately. By default, it waits for:

```python
max(2 * number_of_parameters, 10)
```

complete CV evaluations, capped by the number of states in the finite grid. Before then, it uses space-filling proposals. Set `min_observations` to override the target. Warm-up counts **completed** evaluations, not pruned partial runs.

## Cross-validation pruning

AGS can stop evaluating a candidate before running all folds. This can save estimator fits, but pruning behavior depends on the selected rule.

| `pruning_strategy` | Behavior | Important caveat |
|---|---|---|
| `"percentile"` (default) | Compares a candidate's partial CV mean with historical partial means at the same fold index | Heuristic; not a guarantee of preserving the best configuration |
| `"optimistic"` | Bounds the best possible final mean using a score upper bound | Supply a valid `score_upper_bound`; otherwise AGS disables optimistic pruning and warns |
| `"none"` | Runs every CV fold for each selected configuration | Useful for baselines or debugging |

For example, accuracy has a valid upper bound of 1.0:

```python
search = AdaptiveGreedySearch(
    estimator,
    param_grid,
    scoring="accuracy",
    pruning_strategy="optimistic",
    score_upper_bound=1.0,
)
```

Only use `"optimistic"` when the bound you supply is genuinely valid for the scorer. Do not assume that 1.0 is a valid upper bound for every scikit-learn scoring metric. For a full-CV baseline without fold pruning, select `pruning_strategy="none"`.

### Parallelism notes

Candidate configurations in a batch are evaluated concurrently via `joblib`. Each worker sees the partial-history snapshot taken before that batch was dispatched, so candidates in the same batch cannot use one another's newly produced partial folds for percentile pruning. This slightly softens percentile pruning within a batch.

Parallel execution is workload-dependent: for cheap folds or a single-core machine, process overhead can make `n_jobs > 1` slower than `n_jobs=1`. Also consider limiting the estimator's own threads; AGS sets the estimator's `n_jobs` to 1 when that parameter exists, to reduce nested parallelism.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `cv` | `5` | Number of CV folds. |
| `scoring` | `"accuracy"` | Scikit-learn scorer name, such as `"accuracy"`, `"roc_auc"`, or `"neg_mean_squared_error"`. |
| `n_climbers` | `None` → about 75% of CPU cores, minimum 1 | Number of search climbers. Consider setting this explicitly for small evaluation budgets. |
| `climber_patience` | `3` | Consecutive non-improving steps before a climber is retired as stagnated. |
| `n_jobs` | `-1` | Number of parallel workers for candidate evaluation. |
| `max_evaluations` | `25` | Hard cap on attempted grid states, bounded by grid size. |
| `surrogate_type` | `"rf"` | `"rf"`, `"gp"`, `"kde"`, or deprecated alias `"tpe"`. |
| `acquisition` | `"auto"` | `"auto"`, `"ucb"`, `"ei"` / `"expected_improvement"`, or `"density_ratio"` when compatible with the selected surrogate. |
| `min_observations` | `None` | Warm-up target; defaults to `max(2 * parameter_count, 10)`, capped by grid size. |
| `rf_n_estimators` | `200` | Number of trees for the Random Forest surrogate (minimum 2). |
| `exploration_weight` | `2.0` | Weight applied to predictive spread by UCB. |
| `pruning_strategy` | `"percentile"` | `"percentile"`, `"optimistic"`, or `"none"`. |
| `pruning_percentile` | `25` | Percentile threshold used by heuristic pruning; higher values generally prune more aggressively. |
| `max_neighbor_radius` | `4` | Maximum ring radius used when a climber escapes a plateau. |
| `early_stopping_patience` | `None` | Optional global stop after this many evaluations without improving the best score. |

## Results and evaluation semantics

After `search.fit(X, y)`, the main result attributes include:

- `best_params`, `best_score`, `best_state`: the best configuration with a **complete CV evaluation**.
- `history`: one record per attempted candidate, including `n_folds_used`, `n_folds_total`, `climber_id`, and pruning status/reason.
- `n_evaluations`, `n_completed_evaluations`, `n_partial_evaluations`, `total_time`, `stopped_early`: run summary.
- `n_pruned`, `total_folds_run`, `total_folds_possible`, `folds_saved`: cross-validation work and pruning summary.
- `climbers`, `climbers_spawned`, `climbers_merged`, `plateau_escapes`: information about swarm behavior.
- `completed_evaluations` and `partial_evaluations`: separate stores for fully evaluated and incomplete/pruned results.

AGS trains its surrogate using completed CV observations only. Partial CV results can appear in history for diagnostics but cannot become the reported winner.

## License

MIT © Mohammad Jawad Hasan
