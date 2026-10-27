# Changelog

## v3.3.0 — Principled Surrogate and Acquisition

### Added
- Random Forest regression surrogate as the default for mixed/discrete search spaces.
- Separate acquisition functions: UCB and Expected Improvement, alongside automatic selection.
- A configurable surrogate warm-up target before model-guided proposals are enabled.
- Space-filling candidate selection during warm-up.
- KDE density-ratio scoring as an explicitly named alternative surrogate.
- Regression tests for surrogate outputs, acquisition scores, warm-up behavior, and end-to-end searches.

### Changed
- Acquisition logic is now separated from surrogate implementations.
- Random Forest between-tree dispersion is documented as an uncertainty proxy, not a calibrated posterior standard deviation.
- GP remains available for posterior mean/std and GP-based Expected Improvement.
- The previous TPE-like implementation is no longer presented as a full Tree-structured Parzen Estimator. Use `surrogate_type="kde"`; `surrogate_type="tpe"` remains a deprecated compatibility alias.
- The default surrogate changed from the legacy KDE density-ratio model to Random Forest.

### Compatibility
- Existing plain-list parameter grids and the v3.1/v3.2 evaluation semantics remain supported.
- New constructor options: `acquisition`, `min_observations`, and `rf_n_estimators`.

### Validation
- CI runs the pytest suite on Python 3.10 and 3.12.
