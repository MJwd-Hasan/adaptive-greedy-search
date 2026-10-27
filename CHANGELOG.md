# Changelog

## v3.6.0 — Diversity-Aware Batch Search and Adaptive Restart

### Added
- Diversity-aware acquisition scoring for coordinated parallel proposals.
- Quality/diversity-aware climber respawning using typed search-space distance.
- `diversity_weight` and `restart_diversity_weight`.
- Run diagnostics for trust-region expansion/contraction and mean fidelity.
- Regression tests covering v3.4, v3.5, and v3.6 behavior.

### Changed
- Duplicate avoidance is now proactive during batch construction rather than relying primarily on post-proposal collision handling.
- Restart selection is no longer pure uncertainty; it balances model quality and distance from active basins.

## v3.5.0 — Principled Multi-Fidelity Racing

### Added
- `pruning_strategy="racing"` using confidence-bound fold racing.
- `racing_confidence` configuration.
- Explicit fidelity accounting through fold counts and `mean_fidelity`.

### Correctness
- Partial/raced observations remain separate from full-CV observations.
- Only completed evaluations can become the incumbent, winner, or surrogate training targets.
- Racing is documented as statistical resource allocation rather than a deterministic preservation guarantee.

## v3.4.0 — Adaptive Trust-Region Climbers

### Added
- Per-climber adaptive `trust_radius`.
- Separate current state/score and historical personal-best state/score.
- Trust-region expansion after confirmed improvement and contraction after failure.
- `initial_trust_radius`, `min_trust_radius`, `trust_expand`, and `trust_shrink`.

### Changed
- Fixed-radius plateau traversal is now a bounded fallback; adaptive trust regions are the primary local exploitation mechanism.

## v3.3.0 — Principled Surrogate and Acquisition

- Random Forest regression surrogate as the mixed/discrete default.
- Separate UCB / Expected Improvement acquisition logic.
- Explicit surrogate warm-up and space-filling proposals.
- KDE density-ratio alternative; legacy `tpe` retained only as a deprecated alias.
