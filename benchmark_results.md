# Protocol Benchmark — AGS v3.7 vs Optuna / SMAC3 / Hyperopt / Random

## Configuration

| Parameter | Value |
|---|---|
| Metric | f1_macro |
| CV | 5 |
| Seeds | 20 |
| Budgets | 30, 60 |
| Time cap | 60.0s |
| SMAC3 | True |
| Hyperopt | True |
| scipy | True |

---

## Per-Problem Results

### rf_breast_cancer
- Dataset: breast_cancer (569×30)
- Space: RF (360 configs)
- Conditional: False

**BUDGET 60:**

| Method | Mean | ±95%CI | Wall(s) | N |
|---|---|---|---|---|
| AGS | 0.9357 | 0.0021 | 13.8 | 20 |
| Optuna | 0.9366 | 0.0000 | 11.4 | 20 |
| SMAC3 | -inf | nan | 0.0 | 20 |
| Hyperopt | 0.9366 | 0.0000 | 13.7 | 20 |
| Random | 0.9366 | 0.0000 | 6.4 | 20 |

### rf_noisy
- Dataset: noisy (600×20)
- Space: RF (360 configs)
- Conditional: False

**BUDGET 30:**

| Method | Mean | ±95%CI | Wall(s) | N |
|---|---|---|---|---|
| AGS | 0.6884 | 0.0036 | 9.6 | 20 |
| Optuna | 0.6939 | 0.0016 | 9.1 | 20 |
| SMAC3 | -inf | nan | 0.0 | 20 |
| Hyperopt | 0.6930 | 0.0017 | 9.7 | 20 |
| Random | 0.6912 | 0.0017 | 4.0 | 20 |

**BUDGET 60:**

| Method | Mean | ±95%CI | Wall(s) | N |
|---|---|---|---|---|
| AGS | 0.6899 | 0.0038 | 17.3 | 20 |
| Optuna | 0.6953 | 0.0012 | 21.2 | 20 |
| SMAC3 | -inf | nan | 0.1 | 20 |
| Hyperopt | 0.6949 | 0.0014 | 22.8 | 20 |
| Random | 0.6946 | 0.0013 | 8.4 | 20 |

---

## Aggregate Across Problems

### BUDGET 30

| Method | Mean | ±95%CI | Med Δ vs AGS | Wall(s) |
|---|---|---|---|---|
| AGS | 0.9326 | 0.0834 | — | 6.2 |
| Optuna | 0.9241 | 0.0794 | +0.0010 | 5.2 |
| SMAC3 | -inf | nan | +0.0387 | 4.1 |
| Hyperopt | 0.9237 | 0.0796 | +0.0013 | 5.6 |
| Random | 0.9238 | 0.0803 | +0.0008 | 2.5 |

### BUDGET 60

| Method | Mean | ±95%CI | Med Δ vs AGS | Wall(s) |
|---|---|---|---|---|
| AGS | 0.9330 | 0.0830 | — | 10.4 |
| Optuna | 0.9248 | 0.0793 | +0.0006 | 10.5 |
| SMAC3 | -inf | nan | +0.0385 | 6.8 |
| Hyperopt | 0.9250 | 0.0795 | +0.0001 | 11.6 |
| Random | 0.9249 | 0.0796 | +0.0005 | 4.6 |

---

## Regret Curves (mean best-so-far across seeds)

### rf_digits (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.9337 | 0.9704 | 0.9744 | 0.9765 | 0.9769 |
| Optuna | 0.9093 | 0.9346 | 0.9364 | 0.9386 | 0.9403 |
| SMAC3 | 0.8990 | 0.9356 | 0.9374 | 0.9396 | 0.9403 |
| Hyperopt | 0.8956 | 0.9334 | 0.9369 | 0.9382 | 0.9396 |
| Random | 0.9102 | 0.9327 | 0.9366 | 0.9387 | 0.9397 |

### rf_digits (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.9337 | 0.9704 | 0.9744 | 0.9765 | 0.9770 | 0.9773 |
| Optuna | 0.9093 | 0.9346 | 0.9364 | 0.9386 | 0.9403 | 0.9407 |
| SMAC3 | 0.8990 | 0.9356 | 0.9368 | 0.9392 | 0.9402 | 0.9413 |
| Hyperopt | 0.8956 | 0.9334 | 0.9369 | 0.9382 | 0.9396 | 0.9408 |
| Random | 0.9102 | 0.9327 | 0.9366 | 0.9387 | 0.9397 | 0.9412 |

### rf_breast_cancer (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.9523 | 0.9578 | 0.9588 | 0.9600 | 0.9601 |
| Optuna | 0.9551 | 0.9609 | 0.9618 | 0.9641 | 0.9646 |
| SMAC3 | -inf | -inf | -inf | -inf | -inf |
| Hyperopt | 0.9550 | 0.9606 | 0.9618 | 0.9626 | 0.9642 |
| Random | 0.9560 | 0.9601 | 0.9618 | 0.9631 | 0.9638 |

### rf_breast_cancer (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.9523 | 0.9578 | 0.9588 | 0.9601 | 0.9605 | 0.9605 |
| Optuna | 0.9551 | 0.9609 | 0.9618 | 0.9641 | 0.9646 | 0.9654 |
| SMAC3 | -inf | -inf | -inf | -inf | -inf | -inf |
| Hyperopt | 0.9550 | 0.9606 | 0.9618 | 0.9626 | 0.9642 | 0.9653 |
| Random | 0.9560 | 0.9601 | 0.9618 | 0.9631 | 0.9638 | 0.9651 |

### mlp_digits (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.9600 | 0.9681 | 0.9692 | 0.9697 | 0.9697 |
| Optuna | 0.9134 | 0.9260 | 0.9278 | 0.9288 | 0.9288 |
| SMAC3 | 0.9132 | 0.9263 | 0.9285 | 0.9288 | 0.9288 |
| Hyperopt | 0.9144 | 0.9244 | 0.9272 | 0.9282 | 0.9285 |
| Random | 0.9093 | 0.9288 | 0.9288 | 0.9288 | 0.9288 |

### mlp_digits (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.9600 | 0.9681 | 0.9692 | 0.9697 | 0.9697 | 0.9697 |
| Optuna | 0.9134 | 0.9260 | 0.9278 | 0.9288 | 0.9288 | 0.9288 |
| SMAC3 | 0.9132 | 0.9263 | 0.9286 | 0.9287 | 0.9287 | 0.9287 |
| Hyperopt | 0.9144 | 0.9244 | 0.9272 | 0.9282 | 0.9285 | 0.9288 |
| Random | 0.9093 | 0.9288 | 0.9288 | 0.9288 | 0.9288 | 0.9288 |

### svm_conditional (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.8348 | 0.9701 | 0.9724 | 0.9768 | 0.9778 |
| Optuna | 0.7885 | 0.9692 | 0.9720 | 0.9739 | 0.9746 |
| SMAC3 | 0.7121 | 0.9701 | 0.9726 | 0.9746 | 0.9755 |
| Hyperopt | 0.8060 | 0.9698 | 0.9716 | 0.9736 | 0.9746 |
| Random | 0.8052 | 0.9692 | 0.9718 | 0.9745 | 0.9751 |

### svm_conditional (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.8348 | 0.9701 | 0.9724 | 0.9768 | 0.9779 | 0.9782 |
| Optuna | 0.7885 | 0.9692 | 0.9720 | 0.9739 | 0.9746 | 0.9760 |
| SMAC3 | 0.7121 | 0.9701 | 0.9732 | 0.9752 | 0.9761 | 0.9769 |
| Hyperopt | 0.8060 | 0.9698 | 0.9716 | 0.9736 | 0.9746 | 0.9771 |
| Random | 0.8052 | 0.9692 | 0.9718 | 0.9745 | 0.9751 | 0.9763 |

### svm_flat_wine (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.8348 | 0.9701 | 0.9724 | 0.9768 | 0.9778 |
| Optuna | 0.7840 | 0.9678 | 0.9717 | 0.9731 | 0.9739 |
| SMAC3 | 0.8332 | 0.9700 | 0.9716 | 0.9749 | 0.9753 |
| Hyperopt | 0.7788 | 0.9698 | 0.9721 | 0.9734 | 0.9742 |
| Random | 0.8052 | 0.9692 | 0.9718 | 0.9745 | 0.9751 |

### svm_flat_wine (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.8348 | 0.9701 | 0.9724 | 0.9768 | 0.9779 | 0.9782 |
| Optuna | 0.7840 | 0.9678 | 0.9717 | 0.9731 | 0.9739 | 0.9756 |
| SMAC3 | 0.8332 | 0.9700 | 0.9714 | 0.9744 | 0.9753 | 0.9767 |
| Hyperopt | 0.7788 | 0.9698 | 0.9721 | 0.9734 | 0.9742 | 0.9768 |
| Random | 0.8052 | 0.9692 | 0.9718 | 0.9745 | 0.9751 | 0.9763 |

### knn_iris (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.9612 | 0.9713 | 0.9733 | 0.9746 | 0.9746 |
| Optuna | 0.9665 | 0.9759 | 0.9766 | 0.9796 | 0.9799 |
| SMAC3 | 0.9671 | 0.9753 | 0.9773 | 0.9799 | 0.9799 |
| Hyperopt | 0.9678 | 0.9749 | 0.9773 | 0.9790 | 0.9793 |
| Random | 0.9598 | 0.9598 | 0.9665 | 0.9799 | 0.9799 |

### knn_iris (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.9612 | 0.9713 | 0.9733 | 0.9746 | 0.9746 | 0.9746 |
| Optuna | 0.9665 | 0.9759 | 0.9766 | 0.9796 | 0.9799 | 0.9799 |
| SMAC3 | 0.9671 | 0.9753 | 0.9786 | 0.9799 | 0.9799 | 0.9799 |
| Hyperopt | 0.9678 | 0.9749 | 0.9773 | 0.9790 | 0.9793 | 0.9799 |
| Random | 0.9598 | 0.9598 | 0.9665 | 0.9799 | 0.9799 | 0.9799 |

### rf_moons (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.9206 | 0.9334 | 0.9341 | 0.9352 | 0.9354 |
| Optuna | 0.9267 | 0.9345 | 0.9354 | 0.9365 | 0.9366 |
| SMAC3 | -inf | -inf | -inf | -inf | -inf |
| Hyperopt | 0.9235 | 0.9349 | 0.9356 | 0.9359 | 0.9361 |
| Random | 0.9272 | 0.9353 | 0.9357 | 0.9362 | 0.9364 |

### rf_moons (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.9206 | 0.9334 | 0.9341 | 0.9352 | 0.9355 | 0.9357 |
| Optuna | 0.9267 | 0.9345 | 0.9354 | 0.9365 | 0.9366 | 0.9366 |
| SMAC3 | -inf | -inf | -inf | -inf | -inf | -inf |
| Hyperopt | 0.9235 | 0.9349 | 0.9356 | 0.9359 | 0.9361 | 0.9366 |
| Random | 0.9272 | 0.9353 | 0.9357 | 0.9362 | 0.9364 | 0.9366 |

### rf_noisy (budget 30)

| Method | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| AGS | 0.6667 | 0.6816 | 0.6854 | 0.6881 | 0.6884 |
| Optuna | 0.6707 | 0.6864 | 0.6895 | 0.6932 | 0.6939 |
| SMAC3 | -inf | -inf | -inf | -inf | -inf |
| Hyperopt | 0.6644 | 0.6836 | 0.6866 | 0.6912 | 0.6930 |
| Random | 0.6715 | 0.6846 | 0.6868 | 0.6888 | 0.6912 |

### rf_noisy (budget 60)

| Method | 1 | 5 | 10 | 20 | 30 | 60 |
|---|---|---|---|---|---|---|
| AGS | 0.6667 | 0.6816 | 0.6854 | 0.6881 | 0.6892 | 0.6899 |
| Optuna | 0.6707 | 0.6864 | 0.6895 | 0.6932 | 0.6939 | 0.6953 |
| SMAC3 | -inf | -inf | -inf | -inf | -inf | -inf |
| Hyperopt | 0.6644 | 0.6836 | 0.6866 | 0.6912 | 0.6930 | 0.6949 |
| Random | 0.6715 | 0.6846 | 0.6868 | 0.6888 | 0.6912 | 0.6946 |

---

## Final Verdict

### Budget 30: AGS vs Optuna across 8 problems

| Metric | Value |
|---|---|
| Mean Δ | +0.0085 |
| Median Δ | +0.0010 |
| AGS better | 4/8 |
| Optuna better | 4/8 |
| Wilcoxon p | 0.8438 [not significant] |

### Budget 60: AGS vs Optuna across 8 problems

| Metric | Value |
|---|---|
| Mean Δ | +0.0082 |
| Median Δ | +0.0006 |
| AGS better | 4/8 |
| Optuna better | 3/8 |
| Wilcoxon p | 0.8438 [not significant] |

---

## Research Interpretation

The benchmark should be interpreted as evidence of **problem-dependent competitiveness**, not general superiority of one optimizer.

Across the eight reported problems, AGS has a positive mean difference relative to Optuna at both budgets (+0.0085 at 30 evaluations and +0.0082 at 60), but the median differences are much smaller (+0.0010 and +0.0006). The Wilcoxon signed-rank result is p=0.8438 at both budgets, so the aggregate comparison does **not** establish a statistically significant AGS advantage at the conventional 0.05 level.

### Where AGS appears strongest

The best-so-far trajectories show large early-budget advantages on `rf_digits` and `mlp_digits`. On `rf_digits`, AGS reaches 0.9704 by evaluation 5 and 0.9769 by evaluation 30, compared with 0.9346 and 0.9403 for Optuna. On `mlp_digits`, AGS reaches 0.9681 by evaluation 5 and 0.9697 by evaluation 30, compared with 0.9260 and 0.9288 for Optuna. These cases motivate the hypothesis that AGS can exploit useful local structure particularly efficiently on some finite landscapes.

The conditional SVM experiment is also encouraging: AGS remains competitive with all comparators and finishes above Optuna at both reported budgets. This is relevant to the structured-space design introduced in v3.7, although broader conditional-space experiments are needed before attributing the difference specifically to conditional geometry.

### Where AGS is weaker

AGS trails Optuna on `rf_breast_cancer`, `knn_iris`, `rf_moons`, and `rf_noisy` at the reported checkpoints. Random search is also competitive or better on several of these tasks. This suggests that AGS's local-search bias is not uniformly beneficial; weakly structured, noisy, simple, or broad-plateau landscapes may reward wider stochastic coverage instead.

The difference between the positive mean and near-zero median AGS advantage is important: the aggregate mean is influenced by a small number of comparatively large AGS wins. A neutral interpretation is therefore that AGS currently shows a **higher upside on particular landscapes**, rather than a consistent advantage across tasks.

### Runtime

At budget 60 the aggregate reported wall times are 10.4 s for AGS and 10.5 s for Optuna, with Hyperopt at 11.6 s and Random Search at 4.6 s. Within this benchmark, AGS is therefore approximately at wall-clock parity with Optuna on average, while Random Search remains substantially cheaper in optimizer overhead. Runtime conclusions should remain tied to this protocol because estimator cost, dataset size, parallelism, and hardware can materially change the relative overhead.

Starting with v3.7.1, AGS includes opt-in internal profiling. Future benchmark runs should record the timing breakdown for proposal generation, local-neighborhood search, candidate generation, surrogate fitting/prediction, acquisition/diversity scoring, plateau/restart logic, and evaluation-batch wall time. This will separate optimizer overhead from model-evaluation cost.

### Benchmark integrity notes

- SMAC3 entries reported as `-inf` / `nan` should be treated as **failed or unavailable runs**, not as valid losses to AGS or the other methods. They should be excluded from comparative statistical claims until the integration is corrected and rerun.
- The reported AGS trajectories for `svm_conditional` and `svm_flat_wine` are identical at every displayed checkpoint; the Random trajectories are also identical. This may be legitimate if the two protocol cases intentionally share the same underlying objective, but it should be audited before publication to rule out result reuse, caching, or benchmark wiring errors.
- Budget-30 and budget-60 results from the same problem are correlated observations and should not be treated as independent tasks in inferential statistics.
- More independent datasets and search landscapes are more informative for generalization than adding further random seeds to the same small set of tasks.

### Research direction suggested by the benchmark

A useful next hypothesis is that AGS performance depends on **local objective autocorrelation**. For neighboring configurations (x) and (x'), future work can estimate whether small search-space distance tends to imply similar validation performance and test whether that landscape statistic predicts

[
\Delta_{AGS}=F1_{AGS}-F1_{competitor}.
]

If AGS advantage increases systematically with measurable local structure, the result would explain *when* adaptive greedy search is useful rather than merely ranking optimizers on a fixed benchmark suite.


---

## v3.7.1 Profiling Benchmark — Four-Problem Comparison

**Protocol.** AGS v3.7.1 was run with `profiling=True` against Random Search, Optuna, and Hyperopt on four finite HPO problems. Each method used 5 random seeds, evaluation budgets of 30 and 60, 5-fold cross-validation, and macro F1 as the optimization metric. Hyperopt and XGBoost were available. The four search spaces ranged from 540 to 1,080 configurations.

| Problem | Configs | Budget | AGS | Optuna | Hyperopt | Random | AGS wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RF — Breast Cancer | 1,080 | 30 | 0.9622 | 0.9641 | 0.9649 | **0.9660** | 11.8 |
| RF — Breast Cancer | 1,080 | 60 | 0.9623 | 0.9661 | 0.9657 | **0.9664** | 16.8 |
| RF — Digits | 1,080 | 30 | **0.9783** | 0.9426 | 0.9431 | 0.9410 | 18.1 |
| RF — Digits | 1,080 | 60 | **0.9796** | 0.9441 | 0.9433 | 0.9425 | 31.7 |
| SVM — Breast Cancer | 540 | 30 | **0.9811** | 0.9789 | 0.9785 | 0.9781 | 3.5 |
| SVM — Breast Cancer | 540 | 60 | **0.9811** | 0.9796 | 0.9796 | 0.9800 | 6.2 |
| XGBoost — Wine | 972 | 30 | **0.9870** | 0.9725 | 0.9716 | 0.9726 | 8.8 |
| XGBoost — Wine | 972 | 60 | **0.9893** | 0.9737 | 0.9726 | 0.9737 | 16.1 |

Values are means over five seeds.

### Aggregate results

| Budget | AGS | Optuna | Hyperopt | Random | AGS wall | Optuna wall | Hyperopt wall | Random wall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | **0.9772** | 0.9645 | 0.9645 | 0.9644 | 10.5 s | 7.2 s | 6.5 s | **3.4 s** |
| 60 | **0.9781** | 0.9659 | 0.9653 | 0.9657 | 17.7 s | 13.1 s | 13.8 s | **6.5 s** |

At budget 30, AGS had mean pairwise advantages of +0.0126 over Optuna, +0.0126 over Hyperopt, and +0.0127 over Random Search. At budget 60 the corresponding differences were +0.0122, +0.0127, and +0.0124. AGS won 3 of the 4 problem-level comparisons against each competitor at both budgets.

### Neutral interpretation

These results strengthen the evidence that AGS can be highly effective on some finite structured search landscapes. The largest separation again occurs on the Digits Random Forest problem: AGS exceeds all three competitors by roughly 0.035–0.038 macro-F1. AGS also leads on the SVM Breast Cancer and XGBoost Wine problems at both budgets.

The result is nevertheless not uniform. On Random Forest Breast Cancer, AGS is behind Optuna, Hyperopt, and Random Search at both budgets. This is important because the same estimator family behaves very differently across Breast Cancer and Digits; the evidence therefore points more strongly toward **landscape-dependent performance** than toward an estimator-specific AGS advantage.

The four-problem aggregate favors AGS, and unlike an aggregate driven only by a single comparator, the direction is similar against Optuna, Hyperopt, and Random Search. However, four independent problems remain too few for a strong general-superiority claim. The large Digits advantage also contributes materially to the aggregate mean. A broader suite of independent datasets and search-space geometries is still required for inferential conclusions.

The runtime comparison shows a consistent cost for the additional search machinery. AGS averages 10.5 s versus 7.2 s for Optuna at budget 30, and 17.7 s versus 13.1 s at budget 60. Hyperopt is similarly faster on average, while Random Search is substantially cheaper. Thus, the present evidence supports a **search-quality versus optimizer-overhead trade-off** rather than a claim that AGS is generally faster.

### Profiling status

The benchmark confirms that the installed v3.7.1 accepts `profiling=True` and that profiling is enabled. This alone does **not** identify the dominant runtime component. For bottleneck analysis, the benchmark must also save or print `search.profile_report()` after each AGS fit and aggregate its `sections`, `worker_compute_sum`, and `total_wall_time` fields across seeds.

The v3.7.1 profiler can separate optimizer-side phases such as proposal generation, local-neighborhood search, candidate-pool generation, acquisition scoring, surrogate fitting, diversity scoring, plateau escape, respawn, climber updates, and evaluation-batch wall time. The current profiler reports aggregate worker model-evaluation runtime separately. Deep phase timings are inclusive and may overlap, so their percentages must not be summed as if they were mutually exclusive.

A finer split of worker time into estimator fitting/scoring versus pruning-decision overhead is **not yet directly instrumented** in v3.7.1. Consequently, the current profiler can answer whether model-evaluation batches dominate wall time and whether surrogate/search machinery is expensive, but it cannot yet attribute worker time precisely between model computation and pruning logic.
