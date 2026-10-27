# Changelog

## v3.7.0 — Conditional Spaces and Lazy Candidate Generation

### Added
- `ConditionalParameter` for parent-dependent hyperparameters.
- Explicit `FloatParameter` and `IntegerParameter` types.
- `StructuredSearchSpace` with conditional-aware state validation, encoding, distance, neighborhoods, iteration, and sampling.
- Lazy bounded candidate pools controlled by `candidate_pool_size`.
- Explicit conditional activity masks in surrogate feature encoding.
- `search_space_size_upper_bound`, `lazy_search`, and `has_conditional_parameters` diagnostics.
- Regression tests for conditional state validity, inactive-parameter omission, stable encoding, huge-space construction, lazy iteration, and end-to-end conditional search.

### Changed
- AGS no longer materializes the complete Cartesian product with `list(itertools.product(...))`.
- Max-Min initialization now works over a lazily generated bounded candidate pool.
- Global space-filling proposals and respawns draw from lazy candidate pools.
- Local neighborhoods canonicalize conditional children when a parent changes.
- Inactive conditional parameters are omitted before estimator `set_params`.
- Conditional dimensions contribute an activity mask to surrogate features and conditional-aware distance.

### Compatibility
- Existing plain-list grids remain supported.
- Existing Numeric / Ordinal / Categorical parameter APIs remain supported.
- v3.1-v3.6 evaluation semantics, surrogates, acquisitions, trust regions, racing, and diversity-aware search remain available.

## v3.6.0 — Diversity-Aware Batch Search and Adaptive Restart

- Coordinated diversity-aware batch proposals.
- Quality/diversity-aware respawning.
- Adaptive trust-region and fidelity diagnostics.

## v3.5.0 — Principled Multi-Fidelity Racing

- Confidence-bound fold racing and explicit fidelity accounting.

## v3.4.0 — Adaptive Trust-Region Climbers

- Per-climber adaptive trust regions and separate current/personal-best state.

## v3.3.0 — Principled Surrogate and Acquisition

- Random Forest default surrogate, explicit acquisitions, warm-up, and KDE density-ratio mode.
