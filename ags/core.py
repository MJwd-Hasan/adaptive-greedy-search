"""
Core AdaptiveGreedySearch implementation (AGS v2).
"""

import itertools
import time
import numpy as np

from sklearn.base import clone, is_classifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import get_scorer

from .surrogates import make_surrogate
from .utils import safe_index


class AdaptiveGreedySearch:
    """
    Surrogate-guided greedy search over a discrete hyperparameter grid.

    Search strategy:
        1. Random initial exploration.
        2. Surrogate model fitting (TPE by default, or GP).
        3. Greedy radius-1 neighbor selection (batched surrogate UCB).
        4. Radius-based plateau escape when the immediate neighborhood
           is exhausted (expands outward, stops at the first ring with
           unevaluated candidates, scores them in one batched call).
        5. Batched surrogate fallback over any remaining states, for the
           rare case where even the expanded neighborhood is exhausted.
        6. Optional early stopping if the best score hasn't improved for
           `early_stopping_patience` consecutive evaluations.

    Evaluation:
        Each candidate is scored via cross-validation, run SEQUENTIALLY
        fold by fold (not parallel `cross_val_score`), so that a
        bound-based pruning rule can stop a clearly-uncompetitive
        candidate early instead of finishing out every fold:
          - "optimistic": safe/conservative. Prunes only when even the
            best possible outcome for the remaining folds (assuming
            they all hit the scorer's theoretical max) can't beat the
            current best.
          - "percentile": more aggressive. Prunes when a candidate's
            partial mean falls below a percentile of how other
            candidates were doing at the same fold count.

    Acquisition:
        UCB = predicted_mean + exploration_weight * predicted_std
    """

    def __init__(
        self,
        estimator,
        param_grid,
        cv=5,
        scoring="accuracy",
        initial_points=8,
        max_evaluations=25,
        exploration_weight=2.0,
        random_state=42,
        max_neighbor_radius=4,
        early_stopping_patience=5,
        surrogate_type="tpe",
        tpe_gamma=0.25,
        tpe_bandwidth=1.0,
        enable_pruning=True,
        pruning_strategy="percentile",   # "optimistic" | "percentile" | "none"
        pruning_margin=0.0,
        pruning_percentile=25,
        min_folds_before_pruning=2,
        min_history_for_percentile=10,
        score_upper_bound=None,
    ):

        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

        self.initial_points = initial_points
        self.max_evaluations = max_evaluations
        self.exploration_weight = exploration_weight
        self.random_state = random_state
        self.max_neighbor_radius = max_neighbor_radius
        self.early_stopping_patience = early_stopping_patience
        self.surrogate_type = surrogate_type
        self.tpe_gamma = tpe_gamma
        self.tpe_bandwidth = tpe_bandwidth

        self.enable_pruning = enable_pruning
        self.pruning_strategy = pruning_strategy
        self.pruning_margin = pruning_margin
        self.pruning_percentile = pruning_percentile
        self.min_folds_before_pruning = min_folds_before_pruning
        self.min_history_for_percentile = min_history_for_percentile

        self.rng = np.random.default_rng(random_state)

        self.parameter_names = list(param_grid.keys())
        self.values = [list(param_grid[p]) for p in self.parameter_names]
        self.all_states = list(
            itertools.product(*[range(len(v)) for v in self.values])
        )

        self.evaluated = {}
        self.history = []

        # --- pruning bookkeeping ---
        self._partial_mean_history = {}       # {fold_index: [partial_means...]}
        self._observed_max_fold_score = -np.inf
        self.score_upper_bound = (
            score_upper_bound
            if score_upper_bound is not None
            else self._infer_score_upper_bound(scoring)
        )

        if is_classifier(estimator):
            self._cv_splitter = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )
        else:
            self._cv_splitter = KFold(
                n_splits=cv, shuffle=True, random_state=random_state
            )

        self._scorer = get_scorer(scoring)

        self.surrogate = make_surrogate(
            surrogate_type,
            tpe_gamma=tpe_gamma,
            tpe_bandwidth=tpe_bandwidth,
            random_state=random_state,
        )

    # ========================================================
    # Upper bound inference for the "optimistic" pruning rule.
    # ========================================================

    @staticmethod
    def _infer_score_upper_bound(scoring):
        if not isinstance(scoring, str):
            return None
        if scoring.startswith("neg_"):
            return 0.0  # e.g. neg_mean_squared_error: perfect = 0
        return 1.0  # accuracy, f1*, roc_auc*, r2, precision*, recall*, etc.

    # ========================================================
    # STATE -> REAL HYPERPARAMETERS
    # ========================================================

    def state_to_params(self, state):
        return {
            p: self.values[j][state[j]]
            for j, p in enumerate(self.parameter_names)
        }

    def distance(self, a, b):
        return sum(abs(x - y) for x, y in zip(a, b))

    def get_neighbors(self, state):
        neighbors = []
        for i in range(len(state)):
            for step in (-1, 1):
                new_state = list(state)
                new_state[i] += step
                if 0 <= new_state[i] < len(self.values[i]):
                    new_state = tuple(new_state)
                    if new_state not in self.evaluated:
                        neighbors.append(new_state)
        return neighbors

    # ========================================================
    # EVALUATE -- sequential, fold by fold, with pruning
    # ========================================================

    def evaluate(self, state, X, y):
        params = self.state_to_params(state)

        base_model = clone(self.estimator)
        base_model.set_params(**params)
        if "n_jobs" in base_model.get_params():
            base_model.set_params(n_jobs=1)

        K = self._cv_splitter.get_n_splits(X, y)

        incumbent = max(self.evaluated.values()) if self.evaluated else None

        fold_scores = []
        pruned = False
        prune_reason = None

        start = time.perf_counter()

        for k, (train_idx, test_idx) in enumerate(
            self._cv_splitter.split(X, y), start=1
        ):
            X_train, X_test = safe_index(X, train_idx), safe_index(X, test_idx)
            y_train, y_test = safe_index(y, train_idx), safe_index(y, test_idx)

            fold_model = clone(base_model)
            fold_model.fit(X_train, y_train)
            fold_score = float(self._scorer(fold_model, X_test, y_test))
            fold_scores.append(fold_score)

            if fold_score > self._observed_max_fold_score:
                self._observed_max_fold_score = fold_score

            partial_mean = float(np.mean(fold_scores))
            self._partial_mean_history.setdefault(k, []).append(partial_mean)

            is_last_fold = (k == K)
            can_prune = (
                self.enable_pruning
                and incumbent is not None
                and not is_last_fold
                and k >= self.min_folds_before_pruning
            )

            if can_prune and self.pruning_strategy == "optimistic":
                upper_bound = (
                    self.score_upper_bound
                    if self.score_upper_bound is not None
                    else self._observed_max_fold_score
                )
                best_possible_mean = (
                    sum(fold_scores) + (K - k) * upper_bound
                ) / K
                if best_possible_mean < incumbent - self.pruning_margin:
                    pruned = True
                    prune_reason = "optimistic_bound"
                    break

            elif can_prune and self.pruning_strategy == "percentile":
                hist = self._partial_mean_history.get(k, [])
                if len(hist) >= self.min_history_for_percentile:
                    threshold = np.percentile(hist, self.pruning_percentile)
                    if partial_mean < threshold:
                        pruned = True
                        prune_reason = "percentile_bound"
                        break

        runtime = time.perf_counter() - start
        score = float(np.mean(fold_scores))

        self.evaluated[state] = score
        self.history.append({
            "state": state,
            "params": params,
            "score": score,
            "runtime": runtime,
            "n_folds_used": len(fold_scores),
            "n_folds_total": K,
            "pruned": pruned,
            "prune_reason": prune_reason,
        })

        return score

    # ========================================================
    # SURROGATE
    # ========================================================

    def fit_surrogate(self):
        X_train = np.array(list(self.evaluated.keys()), dtype=float)
        y_train = np.array(list(self.evaluated.values()), dtype=float)
        self.surrogate.fit(X_train, y_train)

    def predict_uncertainty(self, states):
        states_array = np.array(states, dtype=float)
        mean, std = self.surrogate.predict(states_array, return_std=True)
        return mean, std

    def calculate_f(self, state, start_state):
        """Diagnostic blended distance/quality score. Not in the hot path."""
        mean, std = self.predict_uncertainty([state])
        mean = mean[0]
        std = std[0]

        upper_score = mean + (self.exploration_weight * std)

        observed = np.array(list(self.evaluated.values()))
        lo, hi = observed.min(), observed.max()
        span = hi - lo
        norm_score = np.clip((upper_score - lo) / span, 0.0, 1.0) if span > 0 else 0.5

        g_raw = self.distance(start_state, state)
        g = np.clip(g_raw / max(self.max_neighbor_radius, 1), 0.0, 1.0)

        h = 1 - norm_score
        return g + h

    def greedy_select(self):
        best_state = max(self.evaluated, key=self.evaluated.get)
        neighbors = self.get_neighbors(best_state)
        if not neighbors:
            return None
        mean, std = self.predict_uncertainty(neighbors)
        upper_score = mean + (self.exploration_weight * std)
        return neighbors[int(np.argmax(upper_score))]

    def _resolve_plateau(self, start_state):
        visited = {start_state}
        frontier = [start_state]
        candidates = []

        for _ in range(self.max_neighbor_radius):
            next_frontier = []
            for state in frontier:
                for i in range(len(state)):
                    for step in (-1, 1):
                        new_state = list(state)
                        new_state[i] += step
                        if 0 <= new_state[i] < len(self.values[i]):
                            new_state = tuple(new_state)
                            if new_state in visited:
                                continue
                            visited.add(new_state)
                            next_frontier.append(new_state)
                            if new_state not in self.evaluated:
                                candidates.append(new_state)
            frontier = next_frontier
            if candidates or not frontier:
                break

        if not candidates:
            return None

        mean, std = self.predict_uncertainty(candidates)
        upper = mean + self.exploration_weight * std
        return candidates[int(np.argmax(upper))]

    # ========================================================
    # MAIN FIT
    # ========================================================

    def fit(self, X, y):
        total_start = time.perf_counter()

        initial_size = min(
            self.initial_points, len(self.all_states), self.max_evaluations
        )
        initial_states = self.rng.choice(
            len(self.all_states), size=initial_size, replace=False
        )
        for index in initial_states:
            self.evaluate(self.all_states[index], X, y)

        best_so_far = max(self.evaluated.values())
        evaluations_since_improvement = 0
        self.stopped_early = False

        while len(self.evaluated) < min(self.max_evaluations, len(self.all_states)):

            if (
                self.early_stopping_patience is not None
                and evaluations_since_improvement >= self.early_stopping_patience
            ):
                self.stopped_early = True
                break

            if len(self.evaluated) >= 3:
                self.fit_surrogate()

                next_state = self.greedy_select()

                if next_state is None:
                    current_best = max(self.evaluated, key=self.evaluated.get)
                    next_state = self._resolve_plateau(current_best)

                if next_state is None:
                    remaining = [s for s in self.all_states if s not in self.evaluated]
                    if not remaining:
                        break
                    mean, std = self.predict_uncertainty(remaining)
                    upper = mean + self.exploration_weight * std
                    next_state = remaining[int(np.argmax(upper))]

                new_score = self.evaluate(next_state, X, y)

                if new_score > best_so_far:
                    best_so_far = new_score
                    evaluations_since_improvement = 0
                else:
                    evaluations_since_improvement += 1

            else:
                remaining = [s for s in self.all_states if s not in self.evaluated]
                if not remaining:
                    break
                random_state = remaining[self.rng.integers(len(remaining))]
                self.evaluate(random_state, X, y)

        best_state = max(self.evaluated, key=self.evaluated.get)
        self.best_state = best_state
        self.best_score = self.evaluated[best_state]
        self.best_params = self.state_to_params(best_state)
        self.n_evaluations = len(self.evaluated)
        self.total_time = time.perf_counter() - total_start

        self.n_pruned = sum(1 for h in self.history if h["pruned"])
        self.total_folds_run = sum(h["n_folds_used"] for h in self.history)
        self.total_folds_possible = sum(h["n_folds_total"] for h in self.history)
        self.folds_saved = self.total_folds_possible - self.total_folds_run

        return self
