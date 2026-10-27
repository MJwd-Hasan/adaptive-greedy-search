# Adaptive Greedy Search (AGS)

Surrogate-guided hyperparameter search over a discrete grid, using a
**swarm of independent greedy climbers** that share one surrogate
model, one evaluation cache, and one fold-level pruning engine.

```python
from ags import AdaptiveGreedySearch
```

## Why

`GridSearchCV` evaluates every combination — correct, but wasteful once
the grid gets large. `RandomizedSearchCV` is cheap but has no memory:
it never uses what it already learned from earlier trials to pick the
next one. Bayesian approaches such as TPE model which regions of a space look
promising, but a general-purpose sampler may still spend full
cross-validation budget on every trial and does not use AGS's swarm
of local search agents.

AGS combines these ideas for the specific case of a **discrete**
hyperparameter grid:

- **A shared surrogate model** (Random Forest by default, or a Gaussian Process)
  learns from complete CV evaluations across the whole swarm. The Random
  Forest uses between-tree dispersion as an exploration signal; this is
  not a calibrated posterior standard deviation.
- **A swarm of climbers, not one.** `n_climbers` climbers are seeded at
  maximally distant starting points on the grid (Max-Min / farthest-point
  sampling) and search concurrently. Each step, every active climber
  proposes its best surrogate-scored neighbor; duplicate proposals are
  resolved immediately (the lagging climber is retired, not evaluated
  twice), and the whole batch is evaluated in parallel via `joblib`.
- **Plateau escape.** A climber boxed in by already-evaluated neighbors
  doesn't just give up — it expands outward ring by ring (bounded by
  `max_neighbor_radius`) and jumps to the best unvisited state it finds,
  rather than falling back to scanning the whole grid.
- **Respawn toward the unknown.** A climber that's genuinely stuck (no
  ring within the radius has anything left) or has stagnated for
  `climber_patience` steps is retired and replaced — while budget
  remains — at the unvisited state with the *highest surrogate
  uncertainty*, actively pulling the swarm toward unexplored territory.
- **Pruned cross-validation.** Each candidate's CV folds run
  sequentially, and a candidate is stopped early once it's
  mathematically (`"optimistic"`) or statistically (`"percentile"`)
  out of contention against the best result seen so far.

The result: a fixed evaluation budget is spent exploring several basins
of the hyperparameter space at once instead of committing early to one,
while still cutting the *inside* of each evaluation (the CV folds), not
just the *number* of evaluations.

**Where it's not the right tool:** landscapes where quality is scattered
with no local structure (grid distance doesn't correlate with
performance — the surrogate and the climbers' local steps both lose
their signal), very small evaluation budgets (the surrogate needs a
handful of points before it's useful, and a swarm needs enough budget
to give each climber room to walk), or cases where you need the
completeness guarantee of exhaustive grid search.

## Installation

```bash
pip install adaptive-greedy-search
```

Or, from a cloned copy of this repository:

```bash
pip install -e .
```

Requires Python 3.8+, `numpy`, `scikit-learn`, and `joblib` (installed
automatically).

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
    cv=5,                  # pruning gives little benefit below cv=5; use 5-10
    scoring="accuracy",
    max_evaluations=30,
    # n_climbers defaults to 75% of your CPU cores -- pass an int to override
    # n_jobs defaults to -1 (use all cores for the parallel batch dispatch)
)
search.fit(X, y)

print(search.best_params)
print(search.best_score)
print(f"folds saved by pruning: {search.folds_saved}/{search.total_folds_possible}")
print(f"climbers spawned: {search.climbers_spawned}, merged: {search.climbers_merged}, "
      f"plateau escapes: {search.plateau_escapes}")
```

## Key parameters

| Parameter | Default | What it does |
|---|---|---|
| `cv` | `5` | Number of CV folds. Pruning is far more effective at `5` or `10` -- with `3` there's barely a checkpoint before the last fold. |
| `scoring` | `"accuracy"` | Any scikit-learn scorer string (`"accuracy"`, `"neg_mean_squared_error"`, `"roc_auc"`, ...). |
| `n_climbers` | `None` -> 75% of CPU cores (min 1) | Swarm size. Each climber is an independent greedy search sharing the same surrogate. Pass an int to override. |
| `climber_patience` | `3` | Consecutive non-improving steps before a climber is retired as `"stagnated"`. |
| `n_jobs` | `-1` | Passed to `joblib.Parallel` for the per-round batched evaluation dispatch. |
| `max_evaluations` | `25` | Hard cap on the number of grid points evaluated, across the whole swarm. |
| `surrogate_type` | `"rf"` | `"rf"` (Random Forest; mixed-space default; tree-to-tree dispersion is an uncertainty proxy), `"gp"` (Gaussian Process; posterior mean/std), or `"kde"` (KDE density-ratio ranking). `"tpe"` remains a deprecated alias for `"kde"`, not a full TPE implementation. |
| `acquisition` | `"auto"` | `"auto"` chooses UCB for RF/GP and density-ratio ranking for KDE. Also supports `"ucb"`, `"ei"` / `"expected_improvement`, or `"density_ratio"` when compatible with the surrogate. |\n| `min_observations` | `None` | Warm-up target before model-guided search; defaults to `max(2 * parameter_count, 10)`, capped by grid size. |\n| `rf_n_estimators` | `200` | Number of trees in the Random Forest surrogate (minimum 2). |
| `pruning_strategy` | `"percentile"` | `"percentile"` (heuristic), `"optimistic"` (only safe when `score_upper_bound` is a valid upper bound for the scorer), or `"none"` (full CV, no pruning). |
| `pruning_percentile` | `25` | Higher percentiles generally make percentile pruning more aggressive (fewer candidates survive). |
| `max_neighbor_radius` | `4` | How many rings outward a boxed-in climber's plateau escape is allowed to search before it's retired. `0` disables the escape entirely. |
| `early_stopping_patience` | `None` | Optional *global* stop if the swarm's best score hasn't improved for this many evaluations. Per-climber `climber_patience` is the main stagnation mechanism; this is an extra, off-by-default cap on top. |

## Surrogate and acquisition selection

AGS v3.3 defaults to a Random Forest surrogate and a UCB acquisition.
Choose `acquisition="ei"` or `acquisition="expected_improvement"` for
Expected Improvement. With GP, EI uses the GP predictive distribution;
with Random Forest, it is an approximation based on between-tree
dispersion rather than calibrated posterior uncertainty.

The legacy KDE density-ratio model can be used with
`surrogate_type="kde"` and `acquisition="auto"` (or
`acquisition="density_ratio"`). It is deliberately not exposed as a
predictive mean/std model. The old `surrogate_type="tpe"` name is a
deprecated compatibility alias.

By default the surrogate is not activated until at least
`max(2 * number_of_parameters, 10)` complete evaluations have been
collected, capped by the finite grid size. Override this using
`min_observations`. Before that threshold, AGS uses a space-filling
warm-up rather than relying on a tiny, unstable surrogate fit.

```python
search = AdaptiveGreedySearch(
    estimator,
    param_grid,
    surrogate_type="rf",
    acquisition="ei",
    min_observations=12,
)
```

## Choosing a pruning strategy

- Want speed and can tolerate heuristic pruning? Use the default
  `"percentile"`.
- Use `"optimistic"` only when supplying a valid `score_upper_bound`
  for the scorer, e.g. `score_upper_bound=1.0` for accuracy.
- Establishing a baseline, or debugging unexpected results? Use
  `"none"` to run every fold for each selected configuration.

## A note on parallelism

Fold-level percentile pruning compares a candidate's partial CV mean
against the historical distribution of other candidates at the same
fold index. Candidates evaluated in the same parallel batch run
concurrently in separate processes, so each one only sees a snapshot of
that history taken when the batch was dispatched -- it can't see its
batch-mates' partial folds as they happen. This doesn't affect the
`"optimistic"` rule's safety guarantee (it only compares against the
already-completed incumbent), but it very slightly softens
`"percentile"`'s aggressiveness within a single batch.

Parallel speedup depends on your machine and workload: on a
single-core machine, or when each candidate is cheap to evaluate,
`n_jobs > 1` can be *slower* than `n_jobs=1` due to process-spawn and
IPC overhead. It pays off once each fold's training is expensive enough
to amortize that cost. Worth sweeping `n_jobs` on your own hardware
rather than assuming more cores automatically wins.

## What you get back after `.fit(X, y)`

- `best_params`, `best_score`, `best_state` -- the winning configuration.
- `history` -- a list of dicts, one per evaluated candidate, including
  `n_folds_used`, `n_folds_total`, `climber_id`, and whether/why it was pruned.
- `climbers` -- the final list of `Climber` objects (active, stagnated,
  and merged), each with `id`, `current_state`, `current_score`,
  `steps_taken`, `stagnation_count`, `status`.
- `n_evaluations`, `total_time`, `stopped_early`.
- `n_pruned`, `total_folds_run`, `total_folds_possible`, `folds_saved`
  -- how much cross-validation work was actually skipped.
- `climbers_spawned`, `climbers_merged`, `plateau_escapes` -- swarm-specific
  counters for how the search actually behaved.

## License

MIT (c) Mohammad Jawad Hasan
