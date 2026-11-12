# Screening review — 2026-11-12

## Answers grounded in existing runs

1. **MAD collapse is plausible, not established.** Historical logs contain normalized z and evidence but no raw cosine scores or MAD. They cannot identify which part of the denominator changed. The implementation already uses max(1.4826 MAD, 0.1 |median score|); this relative floor also shrinks near zero. Other contributors are persistently atypical honest updates and parent-specific training effects. Full-population scoring already failed to solve false isolation (Phase 3c), so population shrinkage alone is insufficient. New diagnostics log raw scores, median, MAD, effective denominator and reference size every round. An absolute cosine-scale floor is now configurable (`plus.mad_floor`, default zero to preserve old behavior). Floors trade off false isolation against catching mild backdoor attackers; test rather than assume improvement.

2. **Temporal smoothing already exists:** EMA z with decay 0.7. It can still isolate after one very extreme round, and temporary exclusion previously operated immediately. Added independent consecutive-round settings `plus.isolation_patience` and `plus.exclusion_patience`; permanent isolation requires both persistence and the existing EMA condition. Defaults preserve historical behavior. The proposed ablation uses three rounds. Gradual weighting is not implemented; SCCF already supplies continuous weights after screening.

3. **Honest outliers are not simply the most skewed clients.** Reconstructed true training-label distributions using the original seeded CIFAR subset and partitioner. In the three clean alpha=0.5 runs of the best method, isolated honest clients have mean R_skew 0.443 versus 0.530 for others (lower means more skewed). But at alpha=0.05 the direction reverses: 0.155 versus 0.137. These are descriptive pooled associations, with only three seeds, not causal evidence or a calibrated threshold. CSV files identify every client, its sample size, entropy, R_skew and first isolation. Reconstructed labels are offline ground truth; they are not a deployable server-side estimate. Update/head statistics are also attacker-controlled. Do not grant a larger threshold using self-reported R_skew. A deployable calibration would need independently trusted server validation data or another validated measurement; no skew-aware threshold has been enabled.

4. **No ALIE/Min-Max benchmark results were in the existing outputs.** Added coordinated FedHEP attack support: ALIE mean-minus-z-times-std, and Min-Max along coordinate std with a distance-constrained binary search. These act on parameter updates and preserve each client's BatchNorm buffers. Default knowledge is colluders' clean updates; `attack.knowledge: omniscient` uses honest updates as a stronger oracle stress test. `attack.alie_z` controls ALIE magnitude. They do not explicitly optimize this defense's cosine/EMA boundary and must not be described as guaranteed threshold evasion. A defense-aware search and multi-seed training remain future validation.

5. **The gain does not generalize uniformly.** Existing matched three-seed means:

| Setting | Defended FedHEP accuracy | FedHEP+ accuracy |
|---|---:|---:|
| alpha=0.5, clean | 48.39 | 47.92 |
| alpha=0.5, 3 sign flippers | 38.08 | 46.34 |
| alpha=0.05, clean | 65.71 | 65.18 |
| alpha=0.05, 3 sign flippers | 62.81 | 62.75 |
| alpha=0.5, 5 sign flippers | 31.41 | 44.01 |
| alpha=0.5, 5 backdoor attackers | 45.98 | 44.93 |

Five-attacker backdoor success averages 79.1% versus 56.3%; protection is weaker than the three-attacker result. Thus the sign-flip gain holds at 5/15 attackers at alpha=0.5, but disappears at alpha=0.05. No independently held-out setting was established by the development record. Added a newly reserved alpha=0.2, 20-client, seed-2026 matrix; it is not a new dataset or an already validated result. Freeze choices before evaluating it.

6. **Yes, accuracy includes wrongly isolated honest clients.** The evaluation exclusion set is the actual attackers, not isolated clients. Clients with no test examples cannot be scored. FedHEP predicts using each client's most recently locally trained three-head ensemble, so this is personalized accuracy, not pure global-model accuracy. Isolated clients continue local training. Precision/recall already existed; added per-round false-exclusion rate and separately permanent false-isolation rate, both divided by all honest clients. The distinction matters with readmission/caps: ledger isolation and actual aggregation exclusion can differ. Final metrics now explicitly record evaluation scope and true-label diagnostic statistics.

## Artifacts and execution status

- `scripts/diagnose_screening.py`: reproducible audit and MAD plotting. Legacy logs are never assigned fabricated MAD values.
- `docs/followup/best_legacy/`: audited 30 existing best-method runs, including alpha/rate checks; per-client and per-run JSON/CSV.
- `docs/followup/legacy/`: 45 readmission/cap variant audits.
- `docs/followup/cpu/mad_per_round.png`: actual new CIFAR-100 MAD plot, 30 rounds, seed 123, **600 train / 300 test images, one local epoch, CPU**. This reduced-budget clean run has 4.33% accuracy and zero isolated clients. It verifies instrumentation but does not reproduce the historical cascade or justify a floor. Full-scale instrumented reruns are still required.
- The reduced-budget absolute-floor=0.03 run also completed (4.67% accuracy). These low accuracies cannot establish a defense gain. The persistence run was interrupted after round 20 to honor the request to finish quickly; its folder has no metrics.json and will not be treated as finished.
- `configs/sweeps/followup_diagnostics.yaml`: 36 full-scale paired baseline/floor/patience runs.
- `configs/sweeps/followup_generalization.yaml`: 108 baseline/generalization runs including ALIE/Min-Max.
- `configs/sweeps/followup_holdout.yaml`: 10 newly reserved holdout-configuration runs.

Validation: 44 targeted tests passed (screening, benchmark harness and new regression tests), including six new tests for floor behavior, temporal streaks, detection denominator and coordinated-attack constraints. Ruff and configured mypy pass. Full suite and full-scale new sweeps were not run. ALIE/Min-Max constraints are unit-tested; no claim of benchmark efficacy is made.

I agree that defense is the more promising direction: weights-only grouping improved clean accuracy by only about 0.3 points over three seeds. Prioritize measuring the denominator and false-isolation tradeoff before adding more group-level logic. Existing results support a conditional robustness improvement, not a uniformly stronger method.

Attack references: [ALIE, Baruch et al.](https://arxiv.org/abs/1902.06156); [Min-Max, Shejwalkar and Houmansadr](https://people.cs.umass.edu/~amir/papers/NDSS21-model-poisoning.pdf).

## Commands (from RFL_Lab)

```bash
MPLCONFIGDIR=/tmp/rfl-mpl ../.venv/bin/python scripts/diagnose_screening.py outputs --filter p3__zmr__ --out docs/followup/best_legacy
../.venv/bin/python scripts/sweep.py configs/sweeps/followup_diagnostics.yaml --dry-run
../.venv/bin/python scripts/sweep.py configs/sweeps/followup_diagnostics.yaml
MPLCONFIGDIR=/tmp/rfl-mpl ../.venv/bin/python scripts/diagnose_screening.py outputs_followup_full --filter followup__ --out docs/followup/full
```
