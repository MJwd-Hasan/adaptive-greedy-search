"""
Core AdaptiveGreedySearch implementation (AGS v2 + Multi-Start Parallel Swarm).

The single-climber search of AGS v2 is replaced by a swarm of
`n_climbers` independent greedy climbers that share ONE surrogate model,
ONE global evaluation cache, and ONE fold-level pruning engine.
"""

import itertools
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed, cpu_count

from sklearn.base import clone, is_classifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import get_scorer

from .surrogates import make_surrogate
from .utils import safe_index


NEG_INF = float("-inf")


# ============================================================
# CLIMBER
# ============================================================

@dataclass
class Climber:
    """
    One search frontier in the swarm.

    `current_score` is the best score this climber has achieved along
    its whole path (a personal best), not necessarily the score at
    `current_state`.
    """
    id: str
    current_state: Tuple[int, ...]
    current_score: float = NEG_INF
    steps_taken: int = 0
    stagnation_count: int = 0
    status: str = "active"   # "active" | "stagnated" | "merged"


# ============================================================
# PARALLEL WORKER
#
# Module-level (not a method) so joblib can pickle it. It touches NO
# shared state: everything it needs is passed in, everything it
# produces is returned, and the main process merges results into the
# global dictionaries after the batch finishes.
#
# The pruning math below is unchanged from AGS v2's evaluate().
# ============================================================

def _evaluate_worker(
    state,
    params,
    estimator,
    X,
    y,
    cv_splits,
    scorer_name,
    enable_pruning,
    pruning_strategy,
    pruning_margin,
    pruning_percentile,
    min_folds_before_pruning,
    min_history_for_percentile,
    incumbent,
    score_upper_bound,
    observed_max_fold_score,
    partial_mean_history_snapshot,
):
    base_model = clone(estimator)
    base_model.set_params(**params)
    if "n_jobs" in base_model.get_params():
        base_model.set_params(n_jobs=1)

    scorer = get_scorer(scorer_name)
    K = len(cv_splits)

    fold_scores = []
    pruned = False
    prune_reason = None

    local_observed_max = observed_max_fold_score
    new_partial_means = {}   # {fold_index: partial_mean} produced by THIS candidate

    start = time.perf_counter()

    for k, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        X_train, X_test = safe_index(X, train_idx), safe_index(X, test_idx)
        y_train, y_test = safe_index(y, train_idx), safe_index(y, test_idx)

        fold_model = clone(base_model)
        fold_model.fit(X_train, y_train)
        fold_score = float(scorer(fold_model, X_test, y_test))
        fold_scores.append(fold_score)

        if fold_score > local_observed_max:
            local_observed_max = fold_score

        partial_mean = float(np.mean(fold_scores))
        new_partial_means[k] = partial_mean

        is_last_fold = (k == K)
        can_prune = (
            enable_pruning
            and incumbent is not None
            and not is_last_fold
            and k >= min_folds_before_pruning
        )

        if can_prune and pruning_strategy == "optimistic":
            upper_bound = (
                score_upper_bound
                if score_upper_bound is not None
                else local_observed_max
            )
            best_possible_mean = (
                sum(fold_scores) + (K - k) * upper_bound
            ) / K
            if best_possible_mean < incumbent - pruning_margin:
                pruned = True
                prune_reason = "optimistic_bound"
                break

        elif can_prune and pruning_strategy == "percentile":
            # history = snapshot from BEFORE this batch (other candidates in
            # the same batch run concurrently and cannot be seen), plus
            # nothing from this candidate's own earlier folds at index k.
            hist = partial_mean_history_snapshot.get(k, [])
            if len(hist) >= min_history_for_percentile:
                threshold = np.percentile(hist, pruning_percentile)
                if partial_mean < threshold:
                    pruned = True
                    prune_reason = "percentile_bound"
                    break

    runtime = time.perf_counter() - start
    score = float(np.mean(fold_scores))

    return {
        "state": state,
        "params": params,
        "score": score,
        "runtime": runtime,
        "n_folds_used": len(fold_scores),
        "n_folds_total": K,
        "pruned": pruned,
        "prune_reason": prune_reason,
        "max_fold_score": local_observed_max,
        "partial_means": new_partial_means,
    }


# ============================================================
# MAIN CLASS
# ============================================================

class AdaptiveGreedySearch:
    """
    Surrogate-guided search over a discrete hyperparameter grid using a
    Simultaneous Multi-Start Parallel Swarm of greedy climbers.

    Search strategy:
        1. Max-Min seeding: `n_climbers` start states are chosen to be
           maximally far apart on the grid (farthest-point sampling).
        2. Each round, every active climber proposes its best radius-1
           neighbor according to the shared surrogate (UCB). A climber
           that has not been scored yet proposes its own start state.
        3. Deduplication: if a proposal is already evaluated, occupied
           by another climber, or already claimed this round by an
           earlier climber, the lagging climber is retired as "merged"
           (no duplicate compute).
        4. All surviving proposals are evaluated in parallel
           (`joblib.Parallel`), each via sequential fold-by-fold
           cross-validation with the Optimistic / Percentile pruning
           rules unchanged from AGS v2.
        5. Results are merged into the global caches by the main
           process; each climber's personal best and stagnation counter
           are updated; the shared surrogate is refit once per round.
        5b. Plateau escape: a climber whose radius-1 neighbors are all
           evaluated does not die immediately; it expands outward ring by
           ring (up to `max_neighbor_radius`) and jumps to the best
           surrogate-scored unvisited state in the nearest non-empty
           ring. Only if the whole radius budget is exhausted is it
           retired as "stagnated".
        6. A climber that fails to improve for `climber_patience`
           consecutive steps (or has no legal neighbor left) is retired
           as "stagnated". Every retired climber (merged or stagnated)
           is replaced, while budget remains, by a new climber spawned
           at the unvisited state with the highest surrogate
           uncertainty.

    Note on percentile pruning inside a parallel batch: candidates in
    the same batch run concurrently in separate processes, so each one
    sees a snapshot of the percentile history taken when the batch was
    dispatched, not partial results from its batch-mates. The optimistic
    rule is unaffected (it only compares against the already-completed
    incumbent).

    `n_climbers` defaults to 75% of the available CPU cores
    (`max(1, floor(0.75 * joblib.cpu_count()))`), so one batch keeps most
    cores busy while leaving headroom for the main process (surrogate
    fitting, result merging) and the rest of the system.

    Acquisition:
        UCB = predicted_mean + exploration_weight * predicted_std
    """

    def __init__(
        self,
        estimator,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_climbers=None,              # None -> 75% of CPU cores (floor, minimum 1)
        climber_patience=3,
        n_jobs=-1,
        initial_points=None,          # deprecated: swarm seeds `n_climbers` states via Max-Min
        max_evaluations=25,
        exploration_weight=2.0,
        random_state=42,
        max_neighbor_radius=4,        # kept for calculate_f() diagnostics
        early_stopping_patience=None, # optional GLOBAL stop; per-climber patience is the main mechanism
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

        if n_climbers is None:
            n_climbers = max(1, int(cpu_count() * 0.75))
        if n_climbers < 1:
            raise ValueError("n_climbers must be >= 1.")
        if climber_patience < 1:
            raise ValueError("climber_patience must be >= 1.")

        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

        self.n_climbers = n_climbers
        self.climber_patience = climber_patience
        self.n_jobs = n_jobs

        if initial_points is not None:
            warnings.warn(
                "`initial_points` is deprecated and ignored: the swarm is seeded "
                "with `n_climbers` Max-Min states instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.initial_points = initial_points

        self.max_evaluations = max_evaluations
        self.exploration_weight = exploration_weight
        self.random_state = random_state
        self.max_neighbor_radius = max_neighbor_radius
        self.early_stopping_patience = early_stopping_patience
        self.stopped_early = False
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

        # --- swarm bookkeeping ---
        self.climbers: List[Climber] = []
        self._climber_by_id: Dict[str, Climber] = {}
        self._next_climber_idx = 0
        self.climbers_spawned = 0
        self.climbers_merged = 0
        self.plateau_escapes = 0

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
    # MAX-MIN (FARTHEST-POINT) SEEDING
    # ========================================================

    def _get_max_min_seeds(self, n_seeds):
        """
        Greedy farthest-point sampling on the grid: start from one random
        state, then repeatedly add the state whose minimum Manhattan
        distance to all already-chosen seeds is largest. Seeds therefore
        land in maximally different regions of the hyperparameter space.
        """
        states = self.all_states
        n = len(states)
        n_seeds = min(n_seeds, n)

        first_idx = int(self.rng.integers(n))
        chosen_idx = [first_idx]

        min_dist = np.array(
            [self.distance(states[first_idx], s) for s in states], dtype=float
        )

        for _ in range(n_seeds - 1):
            min_dist[chosen_idx] = -1.0     # never re-pick a chosen state
            next_idx = int(np.argmax(min_dist))
            chosen_idx.append(next_idx)
            new_dist = np.array(
                [self.distance(states[next_idx], s) for s in states], dtype=float
            )
            min_dist = np.minimum(min_dist, new_dist)

        return [states[i] for i in chosen_idx]

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

    # ========================================================
    # SWARM MECHANICS
    # ========================================================

    def _retire(self, climber, status):
        climber.status = status
        if status == "merged":
            self.climbers_merged += 1

    def _spawn_climber(self, exclude):
        """Spawn a climber at the unvisited state with highest surrogate uncertainty."""
        remaining = [
            s for s in self.all_states
            if s not in self.evaluated and s not in exclude
        ]
        if not remaining:
            return None

        if len(self.evaluated) >= 3:
            _, std = self.predict_uncertainty(remaining)
            state = remaining[int(np.argmax(std))]
        else:
            state = remaining[int(self.rng.integers(len(remaining)))]

        climber = Climber(id=f"climber_{self._next_climber_idx}", current_state=state)
        self._next_climber_idx += 1
        self.climbers_spawned += 1
        self._climber_by_id[climber.id] = climber
        return climber

    def _resolve_plateau(self, start_state, excluded=None):
        """
        Local plateau escape for a boxed-in climber.

        Breadth-first search outward from `start_state` ring by ring
        (Manhattan distance 2, 3, ... up to `max_neighbor_radius`),
        walking THROUGH already-evaluated states. Stops at the first ring
        that contains an unevaluated, unclaimed state and returns the
        candidate in that ring with the best surrogate UCB (random if the
        surrogate has too little data). Returns None if no such state
        exists within the radius budget; set `max_neighbor_radius=0` to
        disable escapes entirely.
        """
        excluded = excluded or set()
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
                            if new_state not in self.evaluated and new_state not in excluded:
                                candidates.append(new_state)
            frontier = next_frontier
            if candidates or not frontier:
                break

        if not candidates:
            return None

        if len(self.evaluated) >= 3:
            mean, std = self.predict_uncertainty(candidates)   # one batched call
            return candidates[int(np.argmax(mean + self.exploration_weight * std))]
        return candidates[int(self.rng.integers(len(candidates)))]

    def _propose(self):
        """
        Every active climber proposes one candidate state.

        * A climber that has not been scored yet (fresh seed / respawn)
          proposes its own start state.
        * A scored climber proposes its best-UCB unevaluated radius-1
          neighbor from the shared surrogate. If it has no unevaluated
          neighbor left, it is boxed in and retired as "stagnated".

        Deduplication: if two climbers propose the exact same state, the
        LAGGING one (lower personal-best score; ties go to the later
        climber) is retired as "merged" and its proposal is dropped, so
        no state is ever evaluated twice.

        Returns (proposals {climber_id: state}, retired_here [Climber]).
        """
        proposals = {}     # climber_id -> state
        owner = {}         # state -> climber_id that currently holds the claim
        retired_here = []

        have_surrogate = len(self.evaluated) >= 3

        def claim(climber, candidate):
            rival_id = owner.get(candidate)
            if rival_id is None:
                proposals[climber.id] = candidate
                owner[candidate] = climber.id
                return

            rival = self._climber_by_id[rival_id]
            if climber.current_score > rival.current_score:
                # the newcomer leads: the earlier claimant is the laggard
                proposals.pop(rival_id, None)
                proposals[climber.id] = candidate
                owner[candidate] = climber.id
                lagging = rival
            else:
                lagging = climber
            self._retire(lagging, "merged")
            retired_here.append(lagging)

        active = [c for c in self.climbers if c.status == "active"]
        fresh = [c for c in active if c.current_score == NEG_INF]
        scored = [c for c in active if c.current_score != NEG_INF]

        for climber in fresh:
            if climber.current_state in self.evaluated:
                self._retire(climber, "merged")     # start state already visited
                retired_here.append(climber)
                continue
            claim(climber, climber.current_state)

        for climber in scored:
            neighbors = self.get_neighbors(climber.current_state)
            if not neighbors:
                # boxed in: try a bounded ring-expansion jump before giving up
                jump = self._resolve_plateau(climber.current_state, excluded=set(owner))
                if jump is None:
                    self._retire(climber, "stagnated")
                    retired_here.append(climber)
                    continue
                self.plateau_escapes += 1
                claim(climber, jump)
                continue

            if have_surrogate:
                mean, std = self.predict_uncertainty(neighbors)
                ucb = mean + self.exploration_weight * std
                candidate = neighbors[int(np.argmax(ucb))]
            else:
                candidate = neighbors[int(self.rng.integers(len(neighbors)))]

            claim(climber, candidate)

        return proposals, retired_here

    def evaluate_batch(self, proposals, X, y, parallel):
        """
        Evaluate all proposed candidates in parallel. Workers return result
        dictionaries; this (main-process) method merges them into the
        global caches, so no shared state is mutated inside a worker.
        """
        incumbent = max(self.evaluated.values()) if self.evaluated else None
        history_snapshot = {
            k: list(v) for k, v in self._partial_mean_history.items()
        }

        jobs = list(proposals.items())   # [(climber_id, state), ...]

        outputs = parallel(
            delayed(_evaluate_worker)(
                state,
                self.state_to_params(state),
                self.estimator,
                X,
                y,
                self._cv_splits,
                self.scoring,
                self.enable_pruning,
                self.pruning_strategy,
                self.pruning_margin,
                self.pruning_percentile,
                self.min_folds_before_pruning,
                self.min_history_for_percentile,
                incumbent,
                self.score_upper_bound,
                self._observed_max_fold_score,
                history_snapshot,
            )
            for _, state in jobs
        )

        results = {}
        for (climber_id, _), res in zip(jobs, outputs):
            state = res["state"]

            # ---- safe merge into global state (main process only) ----
            self.evaluated[state] = res["score"]
            self.history.append({
                "state": state,
                "params": res["params"],
                "score": res["score"],
                "runtime": res["runtime"],
                "n_folds_used": res["n_folds_used"],
                "n_folds_total": res["n_folds_total"],
                "pruned": res["pruned"],
                "prune_reason": res["prune_reason"],
                "climber_id": climber_id,
            })
            if res["max_fold_score"] > self._observed_max_fold_score:
                self._observed_max_fold_score = res["max_fold_score"]
            for k, pm in res["partial_means"].items():
                self._partial_mean_history.setdefault(k, []).append(pm)

            results[climber_id] = (state, res["score"])

        return results

    def _update_climbers(self, results):
        """Update each climber from its new score; return climbers that stagnated."""
        stagnated = []
        for climber_id, (state, score) in results.items():
            climber = self._climber_by_id[climber_id]
            climber.steps_taken += 1
            climber.current_state = state

            if score > climber.current_score:
                climber.current_score = score
                climber.stagnation_count = 0
            else:
                climber.stagnation_count += 1

            if climber.stagnation_count >= self.climber_patience:
                self._retire(climber, "stagnated")
                stagnated.append(climber)
        return stagnated

    def _respawn(self, retired):
        """Replace each retired climber while evaluation budget remains."""
        if not retired:
            return

        exclude = {c.current_state for c in self.climbers}
        for _ in retired:
            if len(self.evaluated) >= self.max_evaluations:
                break
            new_climber = self._spawn_climber(exclude)
            if new_climber is None:
                break   # grid exhausted
            exclude.add(new_climber.current_state)
            self.climbers.append(new_climber)

    # ========================================================
    # MAIN FIT
    # ========================================================

    def fit(self, X, y):
        total_start = time.perf_counter()

        # ---- reset (fit is safe to call more than once, and deterministic) ----
        self.rng = np.random.default_rng(self.random_state)
        self.surrogate = make_surrogate(
            self.surrogate_type,
            tpe_gamma=self.tpe_gamma,
            tpe_bandwidth=self.tpe_bandwidth,
            random_state=self.random_state,
        )
        self.evaluated = {}
        self.history = []
        self._partial_mean_history = {}
        self._observed_max_fold_score = -np.inf
        self.climbers_spawned = 0
        self.climbers_merged = 0
        self.plateau_escapes = 0

        # cv splits computed once; workers receive the index arrays
        self._cv_splits = list(self._cv_splitter.split(X, y))

        # ---- Max-Min initialization of the swarm ----
        n_seeds = min(self.n_climbers, self.max_evaluations, len(self.all_states))
        seeds = self._get_max_min_seeds(n_seeds)
        self.climbers = [
            Climber(id=f"climber_{i}", current_state=s)
            for i, s in enumerate(seeds)
        ]
        self._climber_by_id = {c.id: c for c in self.climbers}
        self._next_climber_idx = len(self.climbers)

        budget = min(self.max_evaluations, len(self.all_states))
        best_so_far = -np.inf
        evals_since_improvement = 0
        self.stopped_early = False

        # one persistent worker pool for the whole run
        with Parallel(n_jobs=self.n_jobs) as parallel:

            while (
                len(self.evaluated) < budget
                and any(c.status == "active" for c in self.climbers)
            ):
                # 1 + 2. propose and deduplicate
                proposals, retired_in_proposal = self._propose()

                # never exceed the evaluation budget within a batch
                remaining_budget = budget - len(self.evaluated)
                if len(proposals) > remaining_budget:
                    proposals = dict(list(proposals.items())[:remaining_budget])

                # 3. parallel evaluation, main-process merge
                retired_after_eval = []
                if proposals:
                    results = self.evaluate_batch(proposals, X, y, parallel)

                    # optional global early stopping (off by default)
                    for _, (_, score) in results.items():
                        if score > best_so_far:
                            best_so_far = score
                            evals_since_improvement = 0
                        else:
                            evals_since_improvement += 1

                    # 4. update climbers
                    retired_after_eval = self._update_climbers(results)

                if len(self.evaluated) >= 3:
                    self.fit_surrogate()

                # 4. respawn replacements for everyone retired this round
                self._respawn(retired_in_proposal + retired_after_eval)

                if (
                    self.early_stopping_patience is not None
                    and evals_since_improvement >= self.early_stopping_patience
                ):
                    self.stopped_early = True
                    break

                if not proposals and not any(c.status == "active" for c in self.climbers):
                    break

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
