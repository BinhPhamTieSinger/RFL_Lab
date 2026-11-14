# Roadmap

Each phase ends with a check that must pass before the next starts. Results are only compared under identical data splits, seeds and round budgets.

| Phase | Goal | Done when |
|---|---|---|
| **0 · Skeleton** | Layered package, interfaces, quality gates, code graph | `scripts/check.sh` passes (done 2026-11-06) |
| **1 · Faithful port** | FedHEP (both versions in the authors' code) and the 6 baselines rebuilt on the new structure | Same partitions as the authors' code for the same seed; each method's 3-seed mean within 2 standard errors of the authors' code on your Mac |
| **2 · Benchmark harness** | Attacks, attacker placement (spread vs concentrated in one parent), metrics | Accuracy, attack success rate, detection precision/recall/F1 at client *and* parent level, per round |
| **3 · Unified FedHEP** | One system with similarity grouping *and* the defense (fixes "no single configuration does both"). First candidate fix: group on weights only (see the batch-counter finding in EXPERIMENT_LOG.md) | Beats both original versions, or the reason it doesn't is understood |
| **4 · Drill-down defense** | Global scores parents; flagged parents get a per-client check against the trusted consensus | Ablations: number of parents, reference used for drill-down, attacker placement, honest rare-class groups. Run 2026-11-09: **not an improvement** (v1: D26, v2: D28); parent check works; finding attackers inside a parent works but also blames honest members, who are genuinely alike in FedHEP. Phase 4 closed |
| **5 · Wider search** | More mechanisms (FLAME, FLTrust-style reference, divide-and-conquer, median, trimmed mean, Bulyan) and other models. First step: each established rule in FedHEP's Global step (D29) | Each compared against Phase 3 and Phase 4 under the same harness |
| **6 · Multi-level and peer scoring** (your ideas, candidate) | (a) Parents score their own members against the parent's consensus; Global scores parents and clients; one formula fuses the levels. Phase 4b found that honest members of a parent are alike, which should make a lone attacker stand out *more* inside its parent. (b) "Among us": clients evaluate each other's updates (for example on their own data) and vote, combined with server-side screening so a voting majority isn't needed | Same harness; beats FedHEP+ under D23 |

## Drill-down defense: open design questions
These decide whether the idea works, so each is an experiment, not a guess.
1. **How many parents?** Spotting an outlier among 3 parents is fragile; test 3, 5 and 8.
2. **What does Global compare parents on?** Parents with rare classes look unusual even when honest. Compare only on the classes each parent holds (the subspace idea from SCCF).
3. **Drill-down reference.** Judge clients of a flagged parent against the consensus of *trusted* parents, not against their own (possibly attacker-majority) group.
4. **Attackers who spread out.** If attackers avoid forming a malicious parent, the parent-level check sees nothing. Keep the within-parent check for unflagged parents and test both placements.
5. **Grouping must be by similarity.** Random groups (the authors' Defended version) dilute attackers across parents and hide them from the Global check.

## Phase 1 progress
| Step | Status | Evidence |
|---|---|---|
| 1a · Data, ResNet-9, FedAvg end to end | Code done (2026-11-06) | 21 parity tests against the authors' functions: identical Dirichlet and IID partitions (3 seeds × 4 alphas), identical CIFAR subsets, identical normalization, identical ResNet-9 structure and initial weights, identical learning-rate schedule. A planted one-line deviation breaks 12 of them. |
| 1a · Real-data check | Passed (2026-11-06) | IID 39.04 ± 1.25 vs authors' 38.04 ± 0.94 (1.1 SE); α=0.05 29.44 ± 1.33 vs 29.07 ± 2.05 (0.3 SE). See EXPERIMENT_LOG.md |
| 1b · FedHEP (accuracy-table version) | Code done (2026-11-06) | multi-head model, joint three-head trainer, similarity grouping, parent momentum, test-time ensemble. 100 parity tests, incl. one full local round and the ensemble evaluation against the authors' functions |
| 1b · Real-data check | Passed (2026-11-06) | α=0.05: 65.51 ± 0.80 vs authors' 65.68 ± 0.57 (0.3 SE); bottom-10% 49.57 ± 1.91 vs 50.23 ± 1.50. Diagnostics confirm parents copy the global model every round |
| 1c · Defended FedHEP | Code done (2026-11-06) | four attacks and the authors' attacker rule, SCCF at both tiers, trust ledger with permanent isolation. 149 parity tests |
| 1c · Real-data check | Passed (2026-11-06) | α=0.5, 30 rounds: clean 48.39 ± 0.72 (paper 47.88 ± 0.46), sign flip 20% 40.75 ± 4.65 (40.22 ± 3.24), noise 30% 44.63 ± 2.07 (44.53 ± 1.86). Trust ledger isolated no one in any run |
| 1d · FedProx, SCAFFOLD, Ditto, FedRep, Multi-Krum | Code done (2026-11-06) | two federated rounds of each local rule (with and without label flipping), three rounds of two SCAFFOLD clients, Krum scores and Multi-Krum selection, all identical to the authors' functions (43 tests) |
| 1d · Real-data check | Deferred (your call, 2026-11-06) | to run when the comparison tables need them |

## Phase 2 progress
| Step | Status | Evidence |
|---|---|---|
| Attacks | Done (2026-11-06) | the four Byzantine attacks of the attack table, plus a backdoor (BadNets corner trigger) with optional model-replacement boost |
| Attacker choice and placement | Done | the authors' rule (each client with probability q) or an exact count; initial parents random, all attackers in parent 0 ("concentrated") or one per parent ("spread") |
| Metrics | Done | per eval round: accuracy, bottom-10%, attack success rate; **every** round: client-level precision / recall / F1 of the flagged set and AUROC of the trust scores, malicious parents (attacker majority) and, for methods that check parents, parent-level precision / recall / F1; model size |
| Sweeps | Done | `scripts/sweep.py` (resumable grids), `scripts/summarize.py` (mean ± sd tables, CSV) |
| First benchmark | Core done (2026-11-07) | `phase2_core.yaml`, 72 runs: results and charts in EXPERIMENT_LOG.md. `phase2_placement.yaml` (24 runs) not run yet |

## Phase 3 progress
| Step | Status | Evidence |
|---|---|---|
| Diagnose Defended FedHEP's fading defense | Done (2026-11-07) | fixed temperature on shrinking cosines, equal-weights fallback in the last rounds, isolation cut never reached (EXPERIMENT_LOG.md) |
| FedHEP+ (screening, median reference, weight signal, weight-similarity regrouping) | Code done | 13 new tests: z unchanged under shrinking cosines, sign flippers far below honest clients for every signal and reference, median reference resists a coordinated minority, sign flippers excluded (and no honest client) end to end |
| Screening runs, seed 42 | Done (2026-11-07) | best: median reference + regrouping, 47.2 under sign and label flipping (was 38.6 and 42.8), backdoor success 29% (was 80%); found a false-isolation cascade |
| Clean-accuracy ceiling | Done (seed 42) | weights-only grouping +1.2 for FedHEP at α=0.5; 50 rounds no better than 30 |
| 3b · Readmission and exclusion cap | Code done | 3 new tests |
| 3b · Three-seed confirmation | Done (2026-11-08) | **z + median + regroup confirmed better than Defended FedHEP on every seed**: sign flip +8.3, label flip +3.1, backdoor success −58 points; no attack −0.5. Readmission and the cap made things worse |
| 3c · Full-population yardstick (stop false isolations) | Done (2026-11-09) | no improvement: more false isolations, backdoor protection lost. **Phase 3 result: FedHEP+ = screening (median, kept-population z, permanent isolation) + regrouping** |

## Phase 4 progress
| Step | Status | Evidence |
|---|---|---|
| Drill-down: parent check (permutation test) + stricter cut inside flagged parents | Code done (2026-11-09) | 2 new tests: coordinated attackers that look normal one by one are exposed at parent level (synthetic); parent-level metrics recorded end to end |
| Concentrated vs spread attackers, three seeds | Done (2026-11-09) | **not an improvement (D26)**: sign/label flip +0.3 to +0.45 (inside seed spread; attackers already isolated in round 1), concentrated backdoor success 21.5% → 39.7%, false parent flags with no attack. Parent-level detection of the backdoor works (precision 0.85) but the member-level cut cannot separate attackers, and one-round exclusion delayed their isolation |
| Number of parents (3 / 5 / 8), honest rare-class parents | dropped | not worth running after the first result |
| 4b · Drill-down v2: member alignment as evidence, no one-round exclusion, parents ≥ 3 members | Code done (2026-11-09) | 3 new tests (attackers line up with their parent's shared deviation, honest members don't; one honest outlier blames nobody; evidence mode never leaves anyone out for a round) |
| 4b · Three seeds | Done (2026-11-09) | **not an improvement (D28)**: concentrated backdoor success 21.5% → 1.9% (attackers isolated in rounds 2–3), but −1.1 to −1.7 accuracy and twice the false isolations with no attack; honest members of a parent line up with each other as much as attackers do |

## Phase 5 progress
| Step | Status | Notes |
|---|---|---|
| Established rules in FedHEP's Global step: median, trimmed mean, Bulyan, FLAME, FLTrust (`fedhep_robust`) | Code done (2026-11-09) | 13 tests: each rule's defining property, each rule end to end, FLTrust's server training leaves client randomness untouched |
| Screen, seed 42 | Done (2026-11-10) | `phase5_screen.yaml` (30 runs); results in EXPERIMENT_LOG.md |
| Confirm the promising rules on seeds 123 and 7 | Done (2026-11-12) | all rules on 3 seeds: none beats FedHEP+ overall; Bulyan/FLAME stop backdoors (0.5%) at −3.9 to −5.6 clean |
| Harder settings: 5 attackers, boosted backdoor, α = 0.05 | Done (2026-11-12) | FedHEP+ sign-flip gain grows at 5 attackers (+12.6); backdoor protection weakens (56%); no gain at α = 0.05 |
| Professor's follow-up: MAD per round, floor, patience, ALIE / Min-Max, q, 20 and 30 clients, held-out setting | Diagnostics, adaptive attacks, held-out, 20-client backdoor, q sign, α = 0.2 (three methods) done (2026-11-14); patience/two-sided at α = 0.2, two-sided at α = 0.5, q backdoor, 20-client sign, 30 clients running or queued | nine self-contained sweeps, 328 runs; assignment per machine in EXPERIMENT_LOG.md (run plan) |
| Next candidate: FedHEP+ screening + FLAME-style cluster check as evidence (hybrid) | Superseded (2026-11-14) | replaced by Phase 6 below |

## Phase 6 · FedHEP-Guard (proposed 2026-11-14)
Design, hypotheses H1–H7 and stage gates: `docs/RESEARCH_PLAN.md`. Evidence: `docs/LITERATURE_REVIEW.md` and the 2026-11-14 follow-up entry in `docs/EXPERIMENT_LOG.md`.

| Stage | Status | Exit criterion |
|---|---|---|
| 0 · Finish follow-up sweeps | Running (Mac and Acer) | all nine follow-up files plus `followup_two_sided` and `followup_alpha02` logged |
| 1 · Instrumentation (time per phase, peak memory, bytes) | Partly done (2026-11-14): per-round time and peak memory in every run | cost columns filled for every existing method on one setting |
| 2 · Ablation ladder R1–R8 on D1 and D2 | L0 (R5) implemented and queued first: `phase6_l0`, `phase6_l0_a02` | each rung kept or dropped by the gate; negative results logged |
| 3 · Protocol at the 2026 bar | Not started | grid sized from Stage 1 timings |
| 4 · Held-out validation (seeds 2027–2029, α = 0.1 / 50 clients, FEMNIST) | Reserved, untouched | improvement holds on settings never used for tuning |
| 5 · Paper | Not started | |
