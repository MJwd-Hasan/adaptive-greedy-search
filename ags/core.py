"""
Core AdaptiveGreedySearch implementation.

AGS uses a multi-start parallel swarm over a typed, optionally conditional
discrete search space. Candidate pools are generated lazily, so the optimizer
does not need to materialize the full Cartesian product.
"""


import math
import time
import warnings
from statistics import NormalDist
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed, cpu_count

from sklearn.base import clone, is_classifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import get_scorer

from .acquisition import acquisition_scores
from .surrogates import make_surrogate
from .search_space import StructuredSearchSpace
from .utils import safe_index


NEG_INF = float("-inf")


# ============================================================
# CLIMBER
# ============================================================

@dataclass
class Climber:
    """
    One search frontier in the swarm.

    v3.4 keeps the score at the current position separate from the
    climber's historical personal best and maintains an adaptive trust radius.
    """
    id: str
    current_state: Tuple[int, ...]
    current_score: float = NEG_INF
    personal_best_state: Optional[Tuple[int, ...]] = None
    personal_best_score: float = NEG_INF
    trust_radius: float = 1.0
    steps_taken: int = 0
    stagnation_count: int = 0
    successes: int = 0
    failures: int = 0
    status: str = "active"   # "active" | "stagnated" | "merged"

    def __post_init__(self):
        if self.personal_best_state is None:
            self.personal_best_state = self.current_state


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
    racing_confidence,
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

    racing_z = NormalDist().inv_cdf(0.5 + racing_confidence / 2.0)
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
            hist = partial_mean_history_snapshot.get(k, [])
            if len(hist) >= min_history_for_percentile:
                threshold = np.percentile(hist, pruning_percentile)
                if partial_mean < threshold:
                    pruned = True
                    prune_reason = "percentile_bound"
                    break

        elif can_prune and pruning_strategy == "racing" and k >= 2:
            # Statistical multi-fidelity racing. This is intentionally not
            # described as a deterministic guarantee: repeated ordinary
            # confidence bounds are a resource-allocation rule.
            sample_std = float(np.std(fold_scores, ddof=1))
            standard_error = sample_std / math.sqrt(k)
            candidate_ucb = partial_mean + racing_z * standard_error
            if candidate_ucb < incumbent - pruning_margin:
                pruned = True
                prune_reason = "confidence_racing"
                break

    runtime = time.perf_counter() - start
    partial_mean = float(np.mean(fold_scores))
    completed = (not pruned) and (len(fold_scores) == K)
    score = partial_mean if completed else None

    return {
        "state": state,
        "params": params,
        "score": score,
        "partial_mean": partial_mean,
        "fold_scores": list(fold_scores),
        "fidelity": len(fold_scores) / K,
        "completed": completed,
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

    Surrogate choices:
        "rf"  -> Random Forest; between-tree spread is an uncertainty proxy.
        "gp"  -> Gaussian Process; posterior predictive mean/std.
        "kde" -> KDE density-ratio ranking, without predictive std.

    Acquisition choices:
        "auto" -> UCB for RF/GP, density ratio for KDE.
        "ucb"  -> predicted mean + exploration_weight * predicted spread.
        "ei"   -> Expected Improvement (approximate when using RF).
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
        surrogate_type="rf",
        tpe_gamma=0.25,
        tpe_bandwidth=1.0,
        enable_pruning=True,
        pruning_strategy="percentile",   # "optimistic" | "percentile" | "none"
        pruning_margin=0.0,
        pruning_percentile=25,
        min_folds_before_pruning=2,
        min_history_for_percentile=10,
        score_upper_bound=None,
        acquisition="auto",
        min_observations=None,
        rf_n_estimators=200,
        initial_trust_radius=1.0,
        min_trust_radius=1.0,
        trust_expand=2.0,
        trust_shrink=0.5,
        racing_confidence=0.95,
        diversity_weight=0.25,
        restart_diversity_weight=0.5,
        candidate_pool_size=256,
    ):

        if n_climbers is None:
            n_climbers = max(1, int(cpu_count() * 0.75))
        if n_climbers < 1:
            raise ValueError("n_climbers must be >= 1.")
        if climber_patience < 1:
            raise ValueError("climber_patience must be >= 1.")
        if max_neighbor_radius < 1:
            raise ValueError("max_neighbor_radius must be >= 1.")
        if initial_trust_radius <= 0 or min_trust_radius <= 0:
            raise ValueError("Trust radii must be positive.")
        if min_trust_radius > max_neighbor_radius:
            raise ValueError("min_trust_radius cannot exceed max_neighbor_radius.")
        if not (0 < trust_shrink <= 1):
            raise ValueError("trust_shrink must be in (0, 1].")
        if trust_expand < 1:
            raise ValueError("trust_expand must be >= 1.")
        if not (0 < racing_confidence < 1):
            raise ValueError("racing_confidence must be between 0 and 1.")
        if diversity_weight < 0 or restart_diversity_weight < 0:
            raise ValueError("Diversity weights must be non-negative.")
        if candidate_pool_size < 8:
            raise ValueError("candidate_pool_size must be >= 8.")

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
        self.acquisition = acquisition
        self.rf_n_estimators = rf_n_estimators
        self.initial_trust_radius = float(
            np.clip(initial_trust_radius, min_trust_radius, max_neighbor_radius)
        )
        self.min_trust_radius = float(min_trust_radius)
        self.trust_expand = float(trust_expand)
        self.trust_shrink = float(trust_shrink)
        self.racing_confidence = float(racing_confidence)
        self.diversity_weight = float(diversity_weight)
        self.restart_diversity_weight = float(restart_diversity_weight)
        self.candidate_pool_size = int(candidate_pool_size)

        valid_acquisitions = {
            "auto", "ucb", "ei", "expected_improvement", "density_ratio"
        }
        if acquisition not in valid_acquisitions:
            raise ValueError(
                "acquisition must be one of: {}".format(
                    ", ".join(sorted(valid_acquisitions))
                )
            )
        is_kde = surrogate_type in {"kde", "tpe"}
        if acquisition == "density_ratio" and not is_kde:
            raise ValueError(
                "acquisition='density_ratio' requires surrogate_type='kde'."
            )
        if acquisition in {"ucb", "ei", "expected_improvement"} and is_kde:
            raise ValueError(
                "The KDE density-ratio surrogate does not expose predictive "
                "mean/std; use acquisition='auto' or 'density_ratio', or use "
                "surrogate_type='rf'/'gp'."
            )

        valid_pruning = {"optimistic", "percentile", "racing", "none"}
        if pruning_strategy not in valid_pruning:
            raise ValueError(
                "pruning_strategy must be one of: optimistic, percentile, racing, none."
            )
        self.enable_pruning = bool(enable_pruning and pruning_strategy != "none")
        self.pruning_strategy = pruning_strategy
        self.pruning_margin = pruning_margin
        self.pruning_percentile = pruning_percentile
        self.min_folds_before_pruning = min_folds_before_pruning
        self.min_history_for_percentile = min_history_for_percentile

        self.rng = np.random.default_rng(random_state)

        self.search_space = StructuredSearchSpace(param_grid)
        self.parameter_names = self.search_space.parameter_names
        self.parameters = self.search_space.parameters
        self.values = self.search_space.values
        self.search_space_size_upper_bound = self.search_space.size_upper_bound
        self.lazy_search = True

        if min_observations is not None and min_observations < 2:
            raise ValueError("min_observations must be >= 2 when specified.")
        requested_warmup = (
            int(min_observations)
            if min_observations is not None
            else max(2 * len(self.parameters), 10)
        )
        self.min_observations = min(
            requested_warmup,
            max(self.search_space_size_upper_bound, 1),
        )
        self._surrogate_fitted = False

        # v3.1 evaluation semantics:
        # - visited_states tracks every attempted configuration.
        # - partial_evaluations stores pruned / incomplete CV observations.
        # - completed_evaluations stores only full-CV objective values.
        # evaluated remains a backward-compatible alias for completed results.
        self.visited_states = set()
        self.partial_evaluations = {}
        self.completed_evaluations = {}
        self.evaluated = self.completed_evaluations
        self.history = []

        # --- swarm bookkeeping ---
        self.climbers: List[Climber] = []
        self._climber_by_id: Dict[str, Climber] = {}
        self._next_climber_idx = 0
        self.climbers_spawned = 0
        self.climbers_merged = 0
        self.plateau_escapes = 0
        self.trust_expansions = 0
        self.trust_contractions = 0
        self.diversity_resolutions = 0

        # --- pruning bookkeeping ---
        self._partial_mean_history = {}       # {fold_index: [partial_means...]}
        self._observed_max_fold_score = -np.inf
        self.score_upper_bound = score_upper_bound
        if (
            self.enable_pruning
            and self.pruning_strategy == "optimistic"
            and self.score_upper_bound is None
        ):
            warnings.warn(
                "Optimistic pruning requires an explicit valid score_upper_bound "
                "in AGS v3.1. Optimistic pruning has been disabled for this run "
                "because no safe bound was supplied.",
                UserWarning,
                stacklevel=2,
            )
            self.enable_pruning = False

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
            rf_n_estimators=rf_n_estimators,
        )

    # ========================================================
    # Upper bound inference for the "optimistic" pruning rule.
    # ========================================================

    @staticmethod
    def _infer_score_upper_bound(scoring):
        """Deprecated in v3.1: safe pruning bounds must be supplied explicitly."""
        return None

    # ========================================================
    # STATE -> REAL HYPERPARAMETERS
    # ========================================================

    def state_to_params(self, state):
        """Convert a structured state to estimator params, omitting inactive keys."""
        return self.search_space.state_to_params(state)

    def state_to_features(self, state):
        """Encode values plus conditional active masks for the surrogate."""
        return self.search_space.encode(state)

    def distance(self, a, b):
        """Conditional-aware typed distance."""
        return self.search_space.distance(a, b)

    def _semantic_neighbors(self, state, unvisited_only=True):
        """Generate only valid semantic neighbors in the structured space."""
        neighbors = self.search_space.neighbors(state)
        if unvisited_only:
            neighbors = [
                neighbor for neighbor in neighbors
                if neighbor not in self.visited_states
            ]
        return neighbors

    def get_neighbors(self, state):
        return self._semantic_neighbors(state, unvisited_only=True)

    def _states_within_hops(self, start_state, radius, excluded=None):
        """Return unvisited states inside a climber's semantic trust region."""
        excluded = set(excluded or ())
        hops = max(1, min(self.max_neighbor_radius, int(math.ceil(radius))))
        seen = {start_state}
        frontier = [start_state]
        candidates = []
        for _ in range(hops):
            next_frontier = []
            for state in frontier:
                for neighbor in self._semantic_neighbors(state, unvisited_only=False):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
                    if neighbor not in self.visited_states and neighbor not in excluded:
                        candidates.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return candidates

    # ========================================================
    # MAX-MIN (FARTHEST-POINT) SEEDING
    # ========================================================

    def _candidate_pool(self, excluded=None, size=None):
        """Build a bounded lazy candidate pool instead of materializing the grid."""
        excluded = set(excluded or ())
        excluded.update(self.visited_states)
        size = int(size or self.candidate_pool_size)
        return self.search_space.sample_states(
            self.rng,
            size,
            exclude=excluded,
        )

    def _get_max_min_seeds(self, n_seeds):
        """Lazy farthest-point seeding from a bounded valid candidate pool."""
        n_seeds = max(0, int(n_seeds))
        if n_seeds == 0:
            return []

        pool_size = max(self.candidate_pool_size, n_seeds * 8)
        candidates = self.search_space.sample_states(
            self.rng,
            pool_size,
            exclude=(),
        )
        if not candidates:
            return []

        n_seeds = min(n_seeds, len(candidates))
        first_idx = int(self.rng.integers(len(candidates)))
        chosen = [candidates[first_idx]]

        while len(chosen) < n_seeds:
            remaining = [state for state in candidates if state not in chosen]
            if not remaining:
                break
            nearest = np.array([
                min(self.distance(state, seed) for seed in chosen)
                for state in remaining
            ], dtype=float)
            best = np.flatnonzero(np.isclose(nearest, nearest.max()))
            chosen.append(remaining[int(self.rng.choice(best))])

        return chosen

    # ========================================================
    # SURROGATE
    # ========================================================

    def fit_surrogate(self):
        """Fit the surrogate only on complete CV observations."""
        if not self.completed_evaluations:
            raise RuntimeError("At least one complete evaluation is required to fit a surrogate.")
        X_train = np.array(
            [self.state_to_features(state) for state in self.completed_evaluations],
            dtype=float,
        )
        y_train = np.array(list(self.completed_evaluations.values()), dtype=float)
        self.surrogate.fit(X_train, y_train)
        self._surrogate_fitted = True

    def _surrogate_ready(self):
        """Whether the configured warm-up target has been met."""
        return (
            self._surrogate_fitted
            and len(self.completed_evaluations) >= self.min_observations
        )

    def acquisition_scores(self, states):
        """Score states with the configured acquisition function."""
        features = np.array(
            [self.state_to_features(state) for state in states],
            dtype=float,
        )
        best_score = (
            max(self.completed_evaluations.values())
            if self.completed_evaluations
            else None
        )
        return acquisition_scores(
            self.surrogate,
            features,
            acquisition=self.acquisition,
            exploration_weight=self.exploration_weight,
            best_score=best_score,
        )

    def predict_uncertainty(self, states):
        """Return predictive mean and spread for RF/GP surrogates only."""
        if hasattr(self.surrogate, "score_density_ratio"):
            raise ValueError(
                "The KDE density-ratio surrogate has no predictive uncertainty."
            )
        states_array = np.array(
            [self.state_to_features(state) for state in states],
            dtype=float,
        )
        mean, std = self.surrogate.predict(states_array, return_std=True)
        return mean, std

    def calculate_f(self, state, start_state):
        """Diagnostic blended distance/quality score. Not in the hot path."""
        upper_score = float(self.acquisition_scores([state])[0])
        observed_states = list(self.completed_evaluations)
        if observed_states:
            observed_scores = self.acquisition_scores(observed_states)
            lo, hi = float(np.min(observed_scores)), float(np.max(observed_scores))
            span = hi - lo
            norm_score = (
                np.clip((upper_score - lo) / span, 0.0, 1.0)
                if span > 0 else 0.5
            )
        else:
            norm_score = 0.5

        g_raw = self.distance(start_state, state)
        g = np.clip(g_raw / max(self.max_neighbor_radius, 1), 0.0, 1.0)
        return g + (1 - norm_score)

    # ========================================================
    # SWARM MECHANICS
    # ========================================================

    def _retire(self, climber, status):
        climber.status = status
        if status == "merged":
            self.climbers_merged += 1

    def _diversity_bonus(self, states, references):
        if not references:
            return np.ones(len(states), dtype=float)
        raw = np.array([
            min(self.distance(state, ref) for ref in references)
            for state in states
        ], dtype=float)
        scale = max(float(len(self.parameters)), 1.0)
        return raw / scale

    def _select_model_candidate(self, candidates, references=None, weight=None):
        scores = np.asarray(self.acquisition_scores(candidates), dtype=float)
        references = list(references or ())
        diversity_weight = self.diversity_weight if weight is None else float(weight)
        if references and diversity_weight > 0:
            scores = scores + diversity_weight * self._diversity_bonus(
                candidates, references
            )
        best = np.flatnonzero(np.isclose(scores, np.max(scores)))
        return candidates[int(self.rng.choice(best))]

    def _spawn_climber(self, exclude):
        """Respawn using acquisition quality plus distance from active basins."""
        remaining = self._candidate_pool(
            excluded=exclude,
            size=self.candidate_pool_size,
        )
        if not remaining:
            return None

        if self._surrogate_ready():
            active_positions = [
                c.current_state for c in self.climbers if c.status == "active"
            ]
            state = self._select_model_candidate(
                remaining,
                references=active_positions + list(exclude),
                weight=self.restart_diversity_weight,
            )
        else:
            state = self._get_space_filling_candidate(excluded=exclude)

        climber = Climber(
            id=f"climber_{self._next_climber_idx}",
            current_state=state,
            trust_radius=self.initial_trust_radius,
        )
        self._next_climber_idx += 1
        self.climbers_spawned += 1
        self._climber_by_id[climber.id] = climber
        return climber

    def _get_space_filling_candidate(self, excluded=None):
        """Pick a diverse candidate from a bounded lazy pool."""
        excluded = set(excluded or ())
        remaining = self._candidate_pool(
            excluded=excluded,
            size=self.candidate_pool_size,
        )
        if not remaining:
            return None

        references = list(self.visited_states | excluded)
        if not references:
            return remaining[int(self.rng.integers(len(remaining)))]

        nearest_distances = np.array([
            min(self.distance(candidate, reference) for reference in references)
            for candidate in remaining
        ], dtype=float)
        best_distance = nearest_distances.max()
        tied = np.flatnonzero(np.isclose(nearest_distances, best_distance))
        return remaining[int(self.rng.choice(tied))]

    def _resolve_plateau(self, start_state, excluded=None):
        """
        Local plateau escape for a boxed-in climber.

        Breadth-first search outward from `start_state` ring by ring
        (semantic graph distance 2, 3, ... up to `max_neighbor_radius`),
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
                for new_state in self._semantic_neighbors(state, unvisited_only=False):
                    if new_state in visited:
                        continue
                    visited.add(new_state)
                    next_frontier.append(new_state)
                    if new_state not in self.visited_states and new_state not in excluded:
                        candidates.append(new_state)
            frontier = next_frontier
            if candidates or not frontier:
                break

        if not candidates:
            return None

        if self._surrogate_ready():
            return self._select_model_candidate(
                candidates, references=list(excluded or ())
            )
        return candidates[int(self.rng.integers(len(candidates)))]

    def _propose(self):
        """Create a coordinated diversity-aware batch inside adaptive trust regions."""
        proposals = {}
        retired_here = []
        selected = []

        active = [c for c in self.climbers if c.status == "active"]
        fresh = [c for c in active if c.current_score == NEG_INF]
        scored = [c for c in active if c.current_score != NEG_INF]

        for climber in fresh:
            if (
                climber.current_state in self.visited_states
                or climber.current_state in selected
            ):
                self._retire(climber, "merged")
                retired_here.append(climber)
                continue
            proposals[climber.id] = climber.current_state
            selected.append(climber.current_state)

        anchors = [c.current_state for c in active]
        for climber in scored:
            candidates = self._states_within_hops(
                climber.current_state,
                climber.trust_radius,
                excluded=set(selected),
            )
            if not candidates:
                jump = self._resolve_plateau(
                    climber.current_state, excluded=set(selected)
                )
                if jump is None:
                    self._retire(climber, "stagnated")
                    retired_here.append(climber)
                    continue
                self.plateau_escapes += 1
                candidates = [jump]

            if self._surrogate_ready():
                candidate = self._select_model_candidate(
                    candidates, references=anchors + selected
                )
            else:
                candidate = self._get_space_filling_candidate(
                    excluded=set(selected)
                )
                if candidate is None:
                    self._retire(climber, "stagnated")
                    retired_here.append(climber)
                    continue

            if candidate in selected:
                self.diversity_resolutions += 1
                self._retire(climber, "merged")
                retired_here.append(climber)
                continue

            proposals[climber.id] = candidate
            selected.append(candidate)

        return proposals, retired_here

    def evaluate_batch(self, proposals, X, y, parallel):
        """
        Evaluate all proposed candidates in parallel. Workers return result
        dictionaries; this (main-process) method merges them into the
        global caches, so no shared state is mutated inside a worker.
        """
        incumbent = (
            max(self.completed_evaluations.values())
            if self.completed_evaluations
            else None
        )
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
                self.racing_confidence,
            )
            for _, state in jobs
        )

        results = {}
        for (climber_id, _), res in zip(jobs, outputs):
            state = res["state"]

            # ---- safe merge into global state (main process only) ----
            self.visited_states.add(state)

            observation = {
                "state": state,
                "params": res["params"],
                "score": res["score"],
                "partial_mean": res["partial_mean"],
                "fold_scores": res["fold_scores"],
                "fidelity": res["fidelity"],
                "runtime": res["runtime"],
                "n_folds_used": res["n_folds_used"],
                "n_folds_total": res["n_folds_total"],
                "completed": res["completed"],
                "pruned": res["pruned"],
                "prune_reason": res["prune_reason"],
                "climber_id": climber_id,
            }

            if res["completed"]:
                self.completed_evaluations[state] = res["score"]
            else:
                self.partial_evaluations[state] = observation

            self.history.append(observation)
            if res["max_fold_score"] > self._observed_max_fold_score:
                self._observed_max_fold_score = res["max_fold_score"]
            for k, pm in res["partial_means"].items():
                self._partial_mean_history.setdefault(k, []).append(pm)

            results[climber_id] = (state, res["score"], res["completed"])

        return results

    def _update_climbers(self, results):
        """Update positions, personal bests, and v3.4 adaptive trust radii."""
        stagnated = []
        for climber_id, (state, score, completed) in results.items():
            climber = self._climber_by_id[climber_id]
            climber.steps_taken += 1
            improved = completed and score > climber.personal_best_score

            if completed:
                climber.current_state = state
                climber.current_score = score
                if improved:
                    climber.personal_best_state = state
                    climber.personal_best_score = score
                    climber.stagnation_count = 0
                    climber.successes += 1
                    old_radius = climber.trust_radius
                    climber.trust_radius = min(
                        float(self.max_neighbor_radius),
                        max(
                            self.min_trust_radius,
                            climber.trust_radius * self.trust_expand,
                        ),
                    )
                    if climber.trust_radius > old_radius:
                        self.trust_expansions += 1
                else:
                    climber.stagnation_count += 1
                    climber.failures += 1
                    old_radius = climber.trust_radius
                    climber.trust_radius = max(
                        self.min_trust_radius,
                        climber.trust_radius * self.trust_shrink,
                    )
                    if climber.trust_radius < old_radius:
                        self.trust_contractions += 1
            else:
                climber.stagnation_count += 1
                climber.failures += 1
                old_radius = climber.trust_radius
                climber.trust_radius = max(
                    self.min_trust_radius,
                    climber.trust_radius * self.trust_shrink,
                )
                if climber.trust_radius < old_radius:
                    self.trust_contractions += 1

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
            if len(self.visited_states) >= self.max_evaluations:
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
            rf_n_estimators=self.rf_n_estimators,
        )
        self._surrogate_fitted = False
        self.visited_states = set()
        self.partial_evaluations = {}
        self.completed_evaluations = {}
        self.evaluated = self.completed_evaluations
        self.history = []
        self._partial_mean_history = {}
        self._observed_max_fold_score = -np.inf
        self.climbers_spawned = 0
        self.climbers_merged = 0
        self.plateau_escapes = 0
        self.trust_expansions = 0
        self.trust_contractions = 0
        self.diversity_resolutions = 0

        # cv splits computed once; workers receive the index arrays
        self._cv_splits = list(self._cv_splitter.split(X, y))

        # ---- Max-Min initialization of the swarm ----
        n_seeds = min(self.n_climbers, self.max_evaluations)
        seeds = self._get_max_min_seeds(n_seeds)
        self.climbers = [
            Climber(
                id=f"climber_{i}",
                current_state=s,
                trust_radius=self.initial_trust_radius,
            )
            for i, s in enumerate(seeds)
        ]
        self._climber_by_id = {c.id: c for c in self.climbers}
        self._next_climber_idx = len(self.climbers)

        budget = min(
            self.max_evaluations,
            self.search_space_size_upper_bound,
        )
        best_so_far = -np.inf
        evals_since_improvement = 0
        self.stopped_early = False

        # one persistent worker pool for the whole run
        with Parallel(n_jobs=self.n_jobs) as parallel:

            while (
                len(self.visited_states) < budget
                and any(c.status == "active" for c in self.climbers)
            ):
                # 1 + 2. propose and deduplicate
                proposals, retired_in_proposal = self._propose()

                # never exceed the evaluation budget within a batch
                remaining_budget = budget - len(self.visited_states)
                if len(proposals) > remaining_budget:
                    proposals = dict(list(proposals.items())[:remaining_budget])

                # 3. parallel evaluation, main-process merge
                retired_after_eval = []
                if proposals:
                    results = self.evaluate_batch(proposals, X, y, parallel)

                    # optional global early stopping (off by default)
                    for _, (_, score, completed) in results.items():
                        if not completed:
                            continue
                        if score > best_so_far:
                            best_so_far = score
                            evals_since_improvement = 0
                        else:
                            evals_since_improvement += 1

                    # 4. update climbers
                    retired_after_eval = self._update_climbers(results)

                if (
                    len(self.completed_evaluations) >= self.min_observations
                    and not self._surrogate_fitted
                ):
                    self.fit_surrogate()
                elif self._surrogate_ready() and proposals:
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

        if not self.completed_evaluations:
            raise RuntimeError(
                "AGS finished without any full-CV evaluation. "
                "A final result cannot be selected from partial observations."
            )

        best_state = max(
            self.completed_evaluations,
            key=self.completed_evaluations.get,
        )
        self.best_state = best_state
        self.best_score = self.completed_evaluations[best_state]
        self.best_params = self.state_to_params(best_state)
        self.n_evaluations = len(self.visited_states)
        self.n_completed_evaluations = len(self.completed_evaluations)
        self.n_partial_evaluations = len(self.partial_evaluations)
        self.total_time = time.perf_counter() - total_start

        self.n_pruned = sum(1 for h in self.history if h["pruned"])
        self.total_folds_run = sum(h["n_folds_used"] for h in self.history)
        self.total_folds_possible = sum(h["n_folds_total"] for h in self.history)
        self.folds_saved = self.total_folds_possible - self.total_folds_run
        self.mean_fidelity = float(np.mean([
            h["fidelity"] for h in self.history
        ])) if self.history else 0.0
        self.candidate_pool_size_used = self.candidate_pool_size
        self.has_conditional_parameters = self.search_space.has_conditionals

        return self
