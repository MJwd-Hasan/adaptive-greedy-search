# Adaptive Greedy Search — Gaps, Fixes, and Version Roadmap

This document records the main theoretical and implementation gaps identified in the current **Adaptive Greedy Search (AGS) v3.0** design and defines a staged roadmap for addressing them.

The purpose of this roadmap is **not** to implement the newly proposed architecture immediately. Instead, AGS should evolve incrementally, with each version fixing one major conceptual weakness and being benchmarked before the next mechanism is added.

The long-term direction is a theoretically stronger optimizer built around:

- typed and structured hyperparameter spaces,
- multiple adaptive local search processes,
- a shared probabilistic surrogate,
- principled acquisition functions,
- multi-fidelity/racing evaluation,
- diversity-aware exploration and restart,
- and lazy search-space generation.

---

# 1. Current Gaps and How to Fill Them

## Gap 1 — Partial CV evaluations are treated like complete evaluations

### Current problem

When a configuration is pruned after only a subset of cross-validation folds, its partial mean is stored in the same evaluation structure as a complete CV result.

For a configuration evaluated on only (k < K) folds,

[
ar{s}_k(x)=rac{1}{k}sum_{i=1}^{k}s_i(x)
]

is not equivalent to the complete objective

[
f(x)=rac{1}{K}sum_{i=1}^{K}s_i(x).
]

This creates several risks:

- a partially evaluated configuration can influence the surrogate as if it were fully evaluated;
- a partial score can become the global incumbent;
- future pruning thresholds can be based on an unreliable incumbent;
- a pruned configuration can theoretically become the reported best configuration;
- the final reported score may therefore not correspond to a full CV evaluation.

### Better fix

Separate evaluation states explicitly:

```text
visited_states
partial_evaluations
completed_evaluations
```

Only configurations in `completed_evaluations` should be allowed to:

- become the official incumbent,
- become the final best configuration,
- define the final reported score,
- update confirmed climber fitness.

Partial observations should retain fidelity metadata, for example:

[
(x,r,y)
]

where

[
r=rac{k}{K}.
]

A 2-fold observation from a 5-fold CV evaluation therefore has

[
r=0.4.
]

This creates a clean foundation for later multi-fidelity optimization.

---

## Gap 2 — CV pruning is heuristic rather than a principled resource-allocation mechanism

### Current problem

The current pruning system uses optimistic or percentile-based fold-level decisions.

Percentile pruning is useful operationally, but it has no strong guarantee that the chosen percentile threshold is an optimal or statistically justified allocation strategy.

There is also a documentation issue: a lower percentile produces a lower threshold and therefore generally prunes **fewer**, not more, candidates.

### Better fixes

Three increasingly rigorous alternatives are available.

#### Option A — Successive Halving

Evaluate many configurations at low fidelity, retain the strongest fraction, and increase resources for survivors.

[
n_{j+1}=leftlfloorrac{n_j}{eta}ightfloor
]

Example:

```text
16 configurations
      ↓
1 fold
      ↓
8 survivors
      ↓
2 folds
      ↓
4 survivors
      ↓
3 folds
      ↓
2 survivors
      ↓
full CV
```

#### Option B — Hyperband-style allocation

Run multiple Successive Halving schedules with different exploration/resource trade-offs.

This reduces dependence on a single initial resource choice.

#### Option C — Sequential confidence-bound racing

For configuration (x),

[
ar{s}_k(x)=rac{1}{k}sum_{i=1}^{k}s_i(x)
]

with uncertainty estimate

[
SE_k(x)=rac{hat{sigma}_k(x)}{sqrt{k}}.
]

Construct lower and upper bounds:

[
LCB_k(x)=ar{s}_k(x)-cSE_k(x)
]

[
UCB_k(x)=ar{s}_k(x)+cSE_k(x).
]

A candidate may be stopped when its plausible upper performance is below the plausible lower performance of a stronger candidate.

For a genuinely rigorous sequential guarantee, confidence sequences or another sequential-testing framework should eventually replace ordinary repeated confidence intervals.

---

## Gap 3 — The current TPE uncertainty is not a true predictive uncertainty

### Current problem

The current TPE-like surrogate converts a KDE density-ratio heuristic into values interpreted as predictive mean and standard deviation.

The resulting quantity called `std` is not a posterior predictive standard deviation.

Therefore an acquisition rule such as

[
mu(x)+etasigma(x)
]

does not have the theoretical interpretation of standard UCB.

### Better fixes

#### Option A — Proper TPE acquisition

Keep density models for good and bad observations:

[
l(x)=p(xmid yin	ext{good})
]

[
g(x)=p(xmid yin	ext{bad})
]

and select configurations using the appropriate density-ratio / expected-improvement relationship.

Do not reinterpret density-ratio ambiguity as posterior uncertainty.

#### Option B — Random Forest surrogate

Use an ensemble model.

For tree predictions (f_t(x)),

[
mu(x)=rac{1}{T}sum_{t=1}^{T}f_t(x)
]

and an empirical ensemble dispersion can be used as an uncertainty proxy:

[
sigma(x)=
sqrt{
rac{1}{T-1}
sum_{t=1}^{T}
(f_t(x)-mu(x))^2
}.
]

This is attractive for mixed discrete HPO spaces.

#### Option C — Gaussian Process surrogate

For sufficiently small and structured spaces,

[
f(x)sim GP(m(x),k(x,x')).
]

Then posterior mean and variance are mathematically defined, allowing standard acquisition functions such as:

[
UCB(x)=mu(x)+sqrt{eta_t}sigma(x)
]

or Expected Improvement.

### Recommended direction

Use a **Random Forest surrogate as the robust default** for mixed discrete spaces, with GP as an optional model for smaller primarily numerical spaces.

A proper TPE implementation can remain as an alternative surrogate rather than pretending to expose GP-like uncertainty.

---

## Gap 4 — Search-space geometry is artificial

### Current problem

AGS currently represents every parameter through its index in the supplied value list and uses Manhattan distance in index space.

This incorrectly assumes that all parameters are ordered and equally spaced.

For example,

```python
criterion = ["gini", "entropy", "log_loss"]
```

creates artificial distances:

[
d(	ext{gini},	ext{entropy})=1
]

[
d(	ext{gini},	ext{log-loss})=2.
]

There is no theoretical reason for this ordering.

Uneven numerical grids are also distorted. For

```python
C = [0.001, 0.01, 1, 100]
```

every adjacent pair currently has equal index distance even though the numerical scales differ greatly.

### Better fix — typed parameter geometry

Represent the search space as

[
mathcal{X}
=
mathcal{X}_{continuous}
	imes
mathcal{X}_{ordinal}
	imes
mathcal{X}_{categorical}.
]

Use type-specific metrics.

#### Continuous

[
d_j(x,x')
=
rac{|x_j-x'_j|}{U_j-L_j}.
]

#### Log-scale continuous

[
d_j(x,x')
=
rac{|log x_j-log x'_j|}
{log U_j-log L_j}.
]

#### Ordinal

[
d_j(x,x')
=
rac{|i-j|}{n_j-1}.
]

#### Nominal categorical

[
d_j(x,x')
=
mathbf{1}(x_j
eq x'_j).
]

Combine them with

[
d(x,x')=sum_j w_jd_j(x_j,x'_j).
]

This geometry should be shared by:

- Max-Min initialization,
- diversity calculations,
- restart logic,
- local-neighborhood construction,
- and any distance-aware surrogate mechanism.

---

## Gap 5 — The definition of a neighbor is not correct for mixed hyperparameters

### Current problem

A current AGS neighbor changes one parameter index by (+1) or (-1).

That is reasonable for ordered grids but not for nominal categorical variables.

### Better fix

Define neighborhoods by parameter type.

For ordered numerical or ordinal parameters,

[
N_j(x)={x_{j-1},x_{j+1}}
]

subject to boundaries.

For a categorical variable,

[
N_j(x)=mathcal{C}_jsetminus{x_j}.
]

For continuous parameters, later versions may use perturbation neighborhoods:

[
x'_j=x_j+epsilon.
]

This turns the local search graph into a representation of the actual parameter semantics rather than arbitrary index ordering.

---

## Gap 6 — Fixed neighborhood-radius plateau escape is heuristic

### Current problem

AGS expands outward ring by ring until reaching `max_neighbor_radius`.

The same radius policy is used regardless of whether the local landscape is improving rapidly or has become unproductive.

### Better fix — adaptive trust regions

Give each climber (c) its own search radius

[
Delta_c.
]

Local proposals are restricted to

[
xinmathcal{B}(x_c,Delta_c).
]

If the climber repeatedly finds confirmed improvements,

[
Delta_c
leftarrow
min(gamma_{expand}Delta_c,Delta_{max}).
]

If attempts fail,

[
Delta_c
leftarrow
gamma_{shrink}Delta_c.
]

This converts fixed ring expansion into an adaptive local-search mechanism.

---

## Gap 7 — Stagnation and restart rules are arbitrary

### Current problem

A climber is retired after a fixed number of non-improving steps.

This does not measure whether the surrounding region still contains potentially useful configurations.

### Better fix

Base stagnation on acquisition opportunity.

For example, restart if

[
max_{xin N(c)} EI(x)<epsilon.
]

Alternatively,

[
max_{xin N(c)} UCB(x)<f^*+epsilon.
]

Then the climber is retired because the model predicts low remaining opportunity, rather than merely because a counter was reached.

A patience counter may remain as a failsafe, but it should not be the primary theoretical rule.

---

## Gap 8 — Respawning uses pure uncertainty

### Current problem

Selecting

[
x_{spawn}=argmax_xsigma(x)
]

can send a climber toward a highly uncertain but obviously poor region.

### Better fix — quality/diversity-aware restart

Use an acquisition function containing both quality and diversity.

For active climbers (C),

[
A_{spawn}(x)
=
EI(x)
+
lambda
min_{cin C}d(x,x_c).
]

or use a UCB-based equivalent.

This rewards configurations that are both:

1. promising according to the surrogate;
2. sufficiently different from currently explored basins.

---

## Gap 9 — Duplicate proposals are resolved after they happen

### Current problem

Multiple climbers may independently propose the same configuration.

Duplicates are then merged or discarded.

This wastes proposal capacity and reduces effective parallelism.

### Better fix — batch-aware selection

Choose the batch jointly.

First,

[
x_1=argmax_x A(x).
]

Then penalize candidates near (x_1):

[
A_2(x)=A(x)-lambda K(x,x_1).
]

Choose

[
x_2=argmax_x A_2(x)
]

and repeat until the batch is complete.

Alternative batch BO mechanisms such as local penalization, q-EI, or Thompson-style batch sampling can later be explored.

---

## Gap 10 — AGS materializes the entire Cartesian product

### Current problem

The search space size is

[
|mathcal{X}|=prod_{j=1}^{d}n_j.
]

Materializing every state becomes infeasible when the grid reaches millions or billions of combinations.

The optimizer is intended to avoid exhaustive evaluation, so requiring exhaustive state construction creates a scalability contradiction.

### Better fix — lazy candidate generation

Maintain only a candidate subset

[
C_tsubsetmathcal{X}
]

where

[
|C_t|ll|mathcal{X}|.
]

Build candidates from sources such as:

[
C_t=
C_{local}
cup
C_{global}
cup
C_{uncertain}
cup
C_{diverse}.
]

The acquisition function is optimized over this temporary pool rather than every possible state.

Later, specialized acquisition optimizers may replace candidate pools entirely.

---

## Gap 11 — Default swarm size is coupled too strongly to CPU count

### Current problem

If

[
B=25
]

but the machine creates roughly 20–24 climbers, almost the entire evaluation budget can be consumed by initialization.

The algorithm then behaves more like space-filling random search than iterative local search.

### Better fix

Let swarm size depend on both parallel hardware and optimization budget.

For example,

[
n_c=
min
left(
n_{CPU},
lflooralpha Bfloor
ight)
]

with an experimentally validated range such as

[
alphain[0.15,0.30].
]

A later adaptive policy may begin with more climbers and gradually concentrate resources into fewer promising basins.

---

## Gap 12 — The surrogate begins learning from too few observations

### Current problem

A surrogate fitted after only a handful of evaluations is unstable, especially in moderate- or high-dimensional spaces.

### Better fix — explicit warm-up phase

Define

[
N_{warmup}=max(cd,N_{min})
]

where (d) is the active search-space dimension.

Before (N_{warmup}):

- use Max-Min,
- random sampling,
- Latin-hypercube-like sampling,
- or another space-filling strategy.

After warm-up:

- fit the surrogate,
- activate acquisition-guided local search,
- activate model-guided restarts.

---

## Gap 13 — Conditional hyperparameters are not represented correctly

### Current problem

Many HPO spaces are not Cartesian products.

Example:

```text
kernel
├── linear
├── rbf
│   └── gamma
└── poly
    ├── gamma
    └── degree
```

When `kernel="linear"`, `gamma` and `degree` are inactive.

Treating all combinations as ordinary vectors creates invalid or meaningless configurations.

### Better fix — structured parameter definitions

Introduce explicit parameter classes such as:

```text
FloatParameter
IntegerParameter
OrdinalParameter
CategoricalParameter
ConditionalParameter
```

Each state should have an active-mask describing which dimensions are relevant.

Distance, neighborhoods, surrogate encoding, and lazy candidate generation must respect that mask.

---

## Gap 14 — Automatic score upper bounds make the pruning guarantee too broad

### Current problem

Assuming every non-negative sklearn scorer has upper bound (1) is unsafe.

A deterministic pruning guarantee is only valid when the bound is actually valid.

### Better fix

Expose explicit modes.

#### Safe bounded pruning

```python
pruning="safe"
score_bounds=(0.0, 1.0)
```

The user or scorer metadata supplies a valid objective range.

#### Statistical pruning

```python
pruning="statistical"
```

Use racing/confidence logic without claiming deterministic safety.

#### Successive Halving

```python
pruning="successive_halving"
```

Treat fold count or another resource as fidelity.

If no valid bound is known, AGS must not claim that pruning is guaranteed to preserve the exhaustive-search winner.

---

## Gap 15 — Climber state mixes current position and historical best

### Current problem

The current `current_score` behaves more like a personal-best score than the score of `current_state`.

This makes the local-search state semantically ambiguous.

### Better fix

Represent each climber explicitly as

[
C_c=
(x_c,f(x_c),x_c^*,f_c^*,Delta_c).
]

Suggested fields:

```text
current_state
current_score

personal_best_state
personal_best_score

trust_radius
status
```

Then

[
current_score=f(current_state)
]

and

[
personal_best_score=
max_{xin path_c} f(x).
]

---

## Gap 16 — A* terminology does not match the actual optimization problem

### Current problem

A* is based on

[
f(n)=g(n)+h(n)
]

for shortest-path search.

Hyperparameter optimization instead seeks

[
x^*=argmax_{xinmathcal{X}}F(x).
]

AGS does not have a meaningful accumulated path cost comparable to A*.

The existing diagnostic `calculate_f()` is not part of the actual search policy.

### Better fix

Do not describe AGS as an A*-based optimizer unless a genuine admissible path-search formulation is later derived.

A more accurate description is:

> **Surrogate-assisted multi-start adaptive local search for hyperparameter optimization.**

The strongest research identity is likely to come from the interaction between:

- multiple local search agents,
- shared global knowledge,
- adaptive local regions,
- global diversity-aware restart,
- and multi-fidelity resource allocation.

---

# 2. Version Roadmap

The following seven steps should be implemented sequentially.

Every version should be benchmarked and tested before proceeding.

---

# Version 3.1 — Correct Evaluation Semantics

## Objective

Make evaluation bookkeeping mathematically correct before changing the search strategy.

## Changes

### 1. Separate complete and partial evaluations

Introduce structures such as:

```python
visited_states
partial_evaluations
completed_evaluations
```

### 2. Incumbent must use only completed evaluations

[
x^*=
argmax_{xin D_{full}}f(x).
]

A partial result must never become the official incumbent.

### 3. Final result must be fully evaluated

`best_state` and `best_score` must come only from complete CV observations.

### 4. Store fidelity explicitly

A partial observation should include:

```text
state
folds_completed
total_folds
partial_mean
fold_scores
fidelity
pruned
```

with

[
r=k/K.
]

### 5. Surrogate behavior

Initially, train the surrogate only from complete evaluations.

Partial observations can later be incorporated through a proper multi-fidelity model.

### 6. Correct score-bound logic

Remove unsafe automatic assumptions where possible.

Require explicit valid bounds for any pruning mode that claims deterministic safety.

### Acceptance criteria

- no partial score can become the incumbent;
- no partial score can become the reported final result;
- completed and partial observations are distinguishable in history;
- pruning tests verify that reported winners always received full CV;
- existing non-pruned behavior remains reproducible.

---

# Version 3.2 — Typed Search-Space Geometry

## Objective

Replace artificial index geometry with semantics-aware representations.

## Changes

### 1. Add parameter types

At minimum:

```text
NumericParameter
OrdinalParameter
CategoricalParameter
```

### 2. Add normalized distance functions

Use appropriate metrics for numerical, ordinal, logarithmic, and nominal parameters.

### 3. Redesign neighborhood generation

- numerical/ordinal: adjacent values or controlled numerical moves;
- categorical: alternative category choices;
- mixed spaces: one-dimension-at-a-time semantic moves.

### 4. Update Max-Min initialization

Use the new mixed-space metric.

### 5. Update diversity and restart calculations

All distance-based logic must use the same typed representation.

### Acceptance criteria

- reordering a categorical value list does not materially change distances;
- uneven numerical grids no longer behave as equally spaced indices;
- Max-Min seeds remain diverse under the typed metric;
- neighbor generation produces semantically valid alternatives.

---

# Version 3.3 — Principled Surrogate and Acquisition

**Status: Implemented as v3.3.0.** The implementation is committed to `main`; the final CI run and GitHub release are tracked by the release workflow.

## Objective

Replace heuristic uncertainty interpretation with a surrogate/acquisition pair that has a defensible meaning.

## Changes

### 1. Add a Random Forest surrogate

Make RF the preferred mixed-space default.

Expose:

[
mu(x)
]

and an ensemble-based uncertainty proxy

[
sigma(x).
]

### 2. Retain GP as an optional surrogate

Use GP for smaller primarily numerical spaces.

### 3. Refactor TPE

Either:

- implement a proper TPE density-ratio acquisition; or
- rename the current model so it is not presented as posterior mean/std TPE.

### 4. Introduce explicit acquisition functions

Support at least:

```text
UCB
Expected Improvement
```

where mathematically appropriate.

### 5. Add a warm-up requirement

Do not activate surrogate-guided optimization before a minimum number of observations.

A starting rule may be:

[
N_{warmup}=max(2d,10).
]

This value should later be benchmarked.

### Acceptance criteria

- every surrogate exposes clearly documented semantics;
- TPE no longer reports heuristic ambiguity as posterior standard deviation;
- acquisition choice is separated from surrogate implementation;
- surrogate-guided search begins only after warm-up;
- regression tests cover RF, GP acquisition plumbing, KDE density-ratio behavior, warm-up, and end-to-end runs. Cross-optimizer benchmark results remain future validation work.

---

# Version 3.4 — Adaptive Trust-Region Climbers

## Objective

Replace fixed neighborhood-radius logic with adaptive local search regions.

## Changes

### 1. Give every climber a trust radius

[
Delta_c.
]

### 2. Search locally within

[
mathcal{B}(x_c,Delta_c).
]

### 3. Expand after successful local progress

[
Delta_c
leftarrow
min(
gamma_{expand}Delta_c,
Delta_{max}
).
]

### 4. Shrink after failed progress

[
Delta_c
leftarrow
gamma_{shrink}Delta_c.
]

### 5. Separate current state and personal best

Each climber should explicitly track:

```text
current_state
current_score
personal_best_state
personal_best_score
trust_radius
```

### 6. Replace fixed plateau-BFS behavior

The trust region becomes the main local exploration mechanism.

A bounded fallback search may remain only as a recovery mechanism.

### Acceptance criteria

- trust regions expand and contract based on confirmed outcomes;
- climber state has unambiguous semantics;
- local-search behavior is reproducible under fixed random seeds;
- fixed-radius plateau search is no longer the primary exploitation mechanism.

---

# Version 3.5 — Principled Multi-Fidelity Racing

## Objective

Replace ad-hoc pruning with explicit resource allocation.

## Changes

### 1. Treat CV folds as fidelity

[
r=k/K.
]

### 2. Implement Successive Halving

Candidates receive increasing resources only when they survive earlier stages.

### 3. Preserve evaluation semantics from v3.1

Partial racing results remain partial observations.

Only promoted full-fidelity candidates may become the official best result.

### 4. Optionally add confidence-based racing

Experimental support may include sequential confidence bounds.

Any statistical guarantee must state its assumptions explicitly.

### 5. Track computational resource usage

Record:

- number of folds evaluated,
- total estimator fits,
- wall-clock time,
- number of promoted candidates,
- number of pruned candidates,
- full-equivalent evaluation cost.

### Acceptance criteria

- resource accounting is explicit;
- early termination cannot corrupt the incumbent;
- AGS can be benchmarked fairly against non-pruned optimizers using model-fit cost;
- Successive Halving results are reproducible;
- unsafe pruning guarantee language is removed.

---

# Version 3.6 — Diversity-Aware Batch Search and Adaptive Restart

## Objective

Make the multi-climber swarm act as a coordinated search system rather than independent agents that collide after proposal.

## Changes

### 1. Replace pure-uncertainty respawning

Use a quality/diversity acquisition such as

[
A_{spawn}(x)
=
A_{quality}(x)
+
lambda D(x,C).
]

For example,

[
A_{spawn}(x)
=
EI(x)
+
lambda
min_{cin C}d(x,x_c).
]

### 2. Introduce batch-aware proposal selection

After selecting a candidate, penalize nearby candidates before selecting the next proposal.

### 3. Reduce duplicate proposals before evaluation

Collision resolution should become a fallback rather than the main duplicate-management mechanism.

### 4. Make restart logic acquisition-aware

A climber may be restarted when its local region has negligible predicted improvement.

Example:

[
max_{xin N(c)}EI(x)<epsilon.
]

### 5. Make swarm size budget-aware

Use both available parallelism and search budget.

Example:

[
n_c=
min(
n_{CPU},
lflooralpha Bfloor
).
]

### 6. Evaluate adaptive swarm size

Experiment with larger exploration swarms early and smaller exploitation swarms later.

### Acceptance criteria

- duplicate batch proposals are substantially reduced;
- respawns balance expected quality and diversity;
- climber restarts are tied to search opportunity rather than only patience;
- default swarm size does not consume most of small evaluation budgets;
- ablations quantify the contribution of diversity-aware coordination.

---

# Version 4.0 — Conditional Spaces and Lazy Candidate Generation

## Objective

Remove the full Cartesian-grid assumption and make AGS capable of operating on substantially larger and structured HPO spaces.

This is the seventh roadmap step and the point where the search-space architecture changes enough to justify a major version increment.

## Changes

### 1. Add structured parameter classes

Target API:

```text
FloatParameter
IntegerParameter
OrdinalParameter
CategoricalParameter
ConditionalParameter
```

### 2. Support active/inactive parameters

Configurations must encode which dimensions are currently active.

Example:

```text
kernel = linear
    gamma  -> inactive
    degree -> inactive

kernel = rbf
    gamma  -> active
    degree -> inactive

kernel = poly
    gamma  -> active
    degree -> active
```

### 3. Remove mandatory full Cartesian-product materialization

Do not create:

[
mathcal{X}
=
X_1	imes X_2	imescdots	imes X_d
]

as a complete in-memory list.

### 4. Introduce lazy candidate generation

At optimization iteration (t), construct only

[
C_tsubsetmathcal{X}.
]

Possible sources:

[
C_t=
C_{local}
cup
C_{global}
cup
C_{uncertain}
cup
C_{diverse}.
]

### 5. Optimize acquisitions over candidate pools

Instead of scoring the whole state space,

[
x_{t+1}
=
argmax_{xin C_t}A_t(x).
]

### 6. Make local neighborhoods conditional-space aware

Local moves must never create invalid configurations.

### 7. Make surrogates condition-aware

Inactive parameters must be encoded in a way that does not create artificial distances or predictions.

### Acceptance criteria

- AGS can operate without materializing the full search grid;
- conditional configurations are always valid;
- large combinatorial spaces no longer cause immediate memory explosion;
- local search respects active parameter structure;
- Max-Min/diversity logic works with conditional states;
- benchmarks include search spaces too large to exhaustively enumerate.

---

# 3. Version Summary

| Version | Main Goal | Core Change |
|---|---|---|
| **3.1** | Correct evaluation semantics | Separate partial and complete observations |
| **3.2** | Correct search geometry | Typed numerical, ordinal, and categorical spaces |
| **3.3** | Sound model-guided search | RF/GP/proper-TPE surrogate + explicit acquisition |
| **3.4** | Adaptive exploitation | Trust-region climbers |
| **3.5** | Sound early resource allocation | Successive Halving / racing |
| **3.6** | Coordinated swarm search | Diversity-aware batch proposals and restart |
| **4.0** | Scalable structured HPO | Conditional spaces + lazy candidate generation |

---

# 4. Benchmarking Rule Between Versions

No new roadmap stage should be accepted simply because the code works.

Each version should be compared against the previous version using the same:

- datasets,
- estimators,
- search spaces,
- CV splits,
- random seeds,
- evaluation budgets,
- and hardware where possible.

At minimum track:

[
	ext{best objective vs evaluations}
]

[
	ext{best objective vs estimator fits}
]

[
	ext{best objective vs wall-clock time}
]

and, whenever exhaustive search remains feasible,

[
R_N
=
f(x^*)
-
max_{ile N}f(x_i)
]

where (x^*) is the true best state in the finite grid.

The goal of every version should be measurable improvement in one or more dimensions **without silently degrading correctness**.

---

# 5. Required Ablation Strategy

As the algorithm becomes more complex, every major mechanism should be removable through configuration so its effect can be measured.

Recommended ablations include:

```text
AGS without surrogate
AGS without multiple climbers
AGS without Max-Min initialization
AGS without trust regions
AGS without multi-fidelity pruning
AGS without diversity-aware restart
AGS without batch penalization
```

The final algorithm should not rely solely on a comparison of full AGS against Random Search or Grid Search.

It should also demonstrate which AGS components actually produce the gain.

---

# 6. Deferred Architecture

A larger redesign has been proposed around the following conceptual pipeline:

```text
Typed / Structured Search Space
              ↓
      Space-Filling Warm-up
              ↓
        Shared Surrogate
              ↓
   Local + Global Acquisition
              ↓
 Diversity-Aware Batch Selection
              ↓
      Multi-Fidelity Racing
              ↓
      Confirmed Evaluations
              ↓
  Adaptive Trust-Region Climbers
              ↓
     Restart / Continue Search
```

This architecture is **deliberately deferred**.

The roadmap above should be implemented first so that every conceptual change can be isolated, tested, benchmarked, and understood.

Only after v4.0 is stable should AGS be reconsidered as a fully redesigned optimization framework.

---

# 7. Long-Term Research Identity

AGS should avoid presenting itself as an A*-based hyperparameter optimizer unless a true path-cost formulation is later established.

A stronger and more accurate research description is:

> **Adaptive Greedy Search is a surrogate-assisted, multi-start adaptive local-search framework for hyperparameter optimization, combining coordinated local search, shared global modeling, diversity-aware restart, and multi-fidelity resource allocation.**

The potentially distinctive component is not any single mechanism in isolation.

The research question is whether the combination

[
oxed{
	ext{multiple adaptive local regions}
+
	ext{shared surrogate}
+
	ext{diversity-aware global restart}
+
	ext{multi-fidelity racing}
}
]

can reach near-optimal hyperparameter configurations using substantially fewer estimator fits and less wall-clock time than standard competing optimizers across diverse search landscapes.
