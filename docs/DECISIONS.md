# Decision log

This document records the **33 engineering, evaluation, and algorithmic decisions (`D1`–`D33`)** made across Phases 1–6 of RFL Lab, followed by the archived **2026-11-12 Screening Review Memo**.

Rather than mixing active invariants with superseded experiments in a single flat list, decisions are organized below into **four tracks**:

1. **Core Architecture, Evaluation & Quality Invariants** — Active rules governing package structure, metrics, and statistical significance across the entire repository.
2. **Reference Parity & Faithful Port Rules** — Decisions ensuring Phase 1 reproduces the original FedHEP implementation bit-for-bit.
3. **Active Defense Design: FedHEP+ to FedHEP-Guard** — The active progression from robust z-score screening (Phase 3) to Global-step rules (Phase 5) and layered influence bounding + gap-gated screening (Phase 6).
4. **Superseded Decisions & Closed Explorations** — Negative results (e.g., parent drill-down, readmission, round-level worker momentum) preserved as scientific evidence so dead ends are never repeated.

---

## 1. Core Architecture, Evaluation & Quality Invariants

| # | Date | Status | Decision | Why |
|---|---|---|---|---|
| **D1** | 2026-11-06 | `Active` | Rebuild on a new structure instead of refactoring the authors' code in place. Their code stays untouched as the reference implementation. | The authors' code passes all settings through one 59-field config plus ~28 hidden in-code defaults, puts every method's local training in one 959-line class, and splits FedHEP into two engines that behave differently. Parity tests against it are what make the rebuild trustworthy. |
| **D2** | 2026-11-06 | `Active` | One `Detector` interface for both levels (clients in a parent, parents at Global). | Lets the drill-down defense be two detectors composed, and lets any client-level detector be reused at parent level. |
| **D3** | 2026-11-06 | `Active` | Quality gates: ruff, mypy, import-linter (layer rule), pytest + Hypothesis property tests, graphify code graph. | Each covers something the others don't: style and bug patterns, types, architecture, behaviour, structure overview. |
| **D4** | 2026-11-06 | `Active` | Graphify is used for the code graph only. | It maps structure (modules, links, clusters, most-connected nodes). It does not check correctness. Correctness comes from property tests that state invariants (e.g. "weight 0 removes a sender"), checked on hundreds of generated inputs. Full formal verification is not practical for PyTorch training code. |
| **D6** | 2026-11-06 | `Active` | Report accuracy the way the authors' code does: pooled over all clients' test images (sample-weighted). Also record the plain mean over clients and bottom-10%. | The paper's text says "mean over clients", but its numbers come from the pooled metric. Matching it is required for parity; recording both lets either be compared. |
| **D14** | 2026-11-06 | `Active` | Attackers are left out of the accuracy metrics; honest clients isolated by the defense are still scored. | As in the authors' `evaluate_ensemble`. |
| **D16** | 2026-11-06 | `Active` | Detection is scored from what each method itself reveals: Multi-Krum flags the clients it leaves out (trust = −Krum score); Defended FedHEP flags isolated and hard-rejected clients (trust = Global share); methods without a detector flag nothing. | One metric definition for every method, without adding detectors to methods that have none. |
| **D17** | 2026-11-06 | `Active` | Backdoor success is measured with each method's own prediction rule. | For FedHEP that includes masking classes a client never trained on, which blocks a backdoor whose target class the client lacks. That is part of the method, so it counts; the per-client view will show how much of FedHEP's robustness comes from it. |
| **D23** | 2026-11-08 | `Active` | A variant counts as an improvement only when its paired per-seed difference has the same sign on all three seeds and its mean exceeds the Phase 2 seed spread (about 1–1.5 points). | Three seeds can't support formal significance tests; consistent sign plus size is the honest bar. |

---

## 2. Reference Parity & Faithful Port Rules (Phase 1–2)

| # | Date | Status | Decision | Why |
|---|---|---|---|---|
| **D7** | 2026-11-06 | `Active` | Match the authors' deterministic pieces exactly (partitions, subsets, normalization, model and initial weights, learning-rate schedule, aggregation rule), but don't copy their random-number call order during training. | Copying incidental RNG consumption would tie the code to quirks of their implementation. Training parity is judged statistically across seeds instead. |
| **D8** | 2026-11-06 | `Active` | The server averages BatchNorm running statistics with the weights, as the authors do. (Updated in 1d: every server now works on the authors' flat state vector, so the integer batch counter is averaged too and truncated on load, as in their code.) | The counter is never read during training or evaluation, and averaging an integer is meaningless. |
| **D9** | 2026-11-06 | `Active` | FedHEP is ported as the authors' recorded runs execute it: all three heads train together for 5 epochs with binomial loss weights. Their two-stage schedule is not ported. | `high_cardinality_two_stage` is False in every recorded run, so that code path never executes. |
| **D10** | 2026-11-06 | `Active` | Keep the authors' quirks in the faithful port, documented where they occur: ensemble weights start from [0.333, 0.333, 0.334] (not the prior) and are stored as probabilities but reused as logits; grouping and parent decisions use the full flattened update including BatchNorm counters. | Phase 1 must reproduce their numbers. Each quirk becomes a candidate change in Phase 3, tested against this baseline. |
| **D11** | 2026-11-06 | `Active` | The FedHEP server works on flat state vectors (`rfl.core.state`) in state_dict order. | Grouping uses a random projection that maps coordinates by position; any other layout gives different groups. |
| **D12** | 2026-11-06 | `Active` | The random projection is generated in row chunks instead of one D × 256 matrix. | Theirs allocates 3.5 GB per regrouping. `RandomState` draws normals in a fixed stream, so the chunks are exactly their matrix (tested). |
| **D13** | 2026-11-06 | `Updated → D15` | Attacks live in `rfl/attacks` and are applied by the server around local training (targets for label flipping, uploads for the others); attackers are chosen with the authors' rule (each client independently with probability q), so the number of attackers varies by seed. | Parity with the authors' attack table requires the same attackers. A fixed-count rule can be added for Phase 2 as a separate, clearly named option. |
| **D15** | 2026-11-06 | `Active` | Phase 2 benchmarks use an exact attacker count (3 of 15) instead of the authors' probability rule. | With the probability rule the number of attackers varies by seed (1 to 6 at 20–30%), which dominated the spread in Phase 1c. A fixed count makes methods and seeds comparable. The authors' rule stays available for parity runs. |

---

## 3. Active Defense Design: FedHEP+ to FedHEP-Guard (Phase 3, 5, 6)

| # | Date | Status | Decision | Why |
|---|---|---|---|---|
| **D18** | 2026-11-07 | `Active` | Screening uses a robust z-score whose scale is max(1.4826 MAD, 0.1 × abs(median)), and does not screen fewer than 5 senders. | Without the floor, a tight consensus makes MAD tiny and an honest client 0.04 below the others gets z = −5 (seen on synthetic data). The floor scales with the scores, so shrinking cosines still leave z unchanged. |
| **D19** | 2026-11-07 | `Active` | Phase 3 screens variants on one seed first (seed 42), then confirms the best on all three. | 20 runs instead of 60 for the first pass; seed 42's Phase 2 rows (same attackers) are the matched baseline. A one-seed difference smaller than the Phase 2 seed spread is treated as no difference until confirmed. |
| **D22** | 2026-11-08 | `Active` | Isolation stays permanent; readmission and the cap are kept as options but not used going forward. | Three seeds: readmission brought attackers back late in training (when update directions carry little information) and the cap let backdoor attackers back in; both lowered accuracy under attack and raised backdoor success. |
| **D24** | 2026-11-09 | `Active` | Phase 3 result is FedHEP+ with the kept-population yardstick (3b), not the full-population one (3c). | 3c had more false isolations and lost the backdoor protection (54–74% success vs 21%), with no gain elsewhere. |
| **D29** | 2026-11-09 | `Active` | Phase 5 compares established rules inside FedHEP's Global step (Option A), not as standalone FedAvg defenses. Rules see weight updates only (BatchNorm buffers by coordinate median, as SCCF); parents run SCCF over the clients the rule accepts. Trimmed mean and Bulyan assume 3 attackers (the oracle the authors gave Multi-Krum). FLTrust's 100 root images are training images no client holds, repeated to the median client's size so the server takes as many SGD steps as a client. FLAME is run both on local models (the paper's equations) and on updates (the other reading). | Same model, personalisation and attacks as FedHEP+, so differences come from the rule. Weights only, because Phase 2 showed BatchNorm counters dominate distances. Root images from outside the clients' data keep the clients' data identical to every other method's. |
| **D30** | 2026-11-12 | `Active` | FedHEP+ stays the main method after Phase 5. The follow-up session's code (MAD logging, absolute floor, patience, ALIE / Min-Max, skew audit) is adopted as is, with ALIE's z set to the paper's formula (0.43 for n = 15, m = 3; 0.39 for n = 20, m = 4). Its 108-run generalization sweep is replaced by two smaller sweeps (attacks, q). | Three-seed Phase 5: no rule beats FedHEP+ overall. The follow-up code passed every check here; its z = 1.0 was stronger than the attack it names; about 30 of its generalization runs repeated existing ones. |
| **D32** | 2026-11-14 | `Active` | Backdoor recovery after isolation becomes hypothesis H8 of the research plan. | 20 clients, α = 0.5, FedHEP+ isolated all 5 backdoor attackers by rounds 11–15 in every seed, yet success (measured on honest clients only) kept rising afterwards, by 17–47 points by round 30 in all five runs where every attacker was isolated. Isolation alone does not remove an implanted backdoor. Note for H3 (reversible trust): D22 found that readmission brought attackers back late in training, so reversibility must be tested against that failure, not assumed safe. |
| **D33** | 2026-11-15 | `Active` | Worker momentum is dropped from layer L0; the ablation ladder is reordered so that L0 is the base and the screening rungs (two-sided distance, bimodality gate, reversible trust, extra signals) are added on top of it. | Momentum 0.5 lost 1.1–1.7 points in every D1 setting (one momentum step per round over 30 rounds is too few for the stale update to pay off). At α = 0.2, L0 alone beat FedHEP+ by +1.6 with no attack and +1.4 under the backdoor on every seed with zero false isolations, while every screening variant lost to no defense; so the gate must be shown to work on top of L0's floor. L0's weakness, sign flip at α = 0.5 (−4.7), is exactly what screening fixes, which is the case for layering. |

---

## 4. Superseded Decisions & Closed Explorations (Negative Results Record)

| # | Date | Status | Decision | Why It Was Superseded or Closed |
|---|---|---|---|---|
| **D5** | 2026-11-06 | `Updated → OPERATIONS.md` | Experiments run on your Mac (MPS); the cloud workspace only runs tests. | The cloud workspace has no GPU. |
| **D20** | 2026-11-07 | `Superseded → D22` | Readmission scores isolated clients against the reference and spread of the clients currently kept, and readmits at evidence > −cut/2 (hysteresis). | Scoring them against a reference that includes them would let attackers pull it; the gap between the isolate and readmit levels stops a client flickering in and out. |
| **D21** | 2026-11-07 | `Superseded → D22` | The exclusion cap is a third of all clients. | The usual Byzantine assumption is fewer than half the clients malicious; a third leaves room for 3 attackers plus 2 borderline honest clients out of 15 while making a total collapse impossible. |
| **D25** | 2026-11-09 | `Closed (Phase 4)` | Parents are judged against random groups of the same size (permutation test), not on the clients' yardstick. | A group mean is always more aligned than a single client, so on the client yardstick every parent looks good. Comparing with same-size random groups removes that size effect for any number of parents. |
| **D26** | 2026-11-09 | `Closed (Phase 4)` | The drill-down (v1) is not adopted; FedHEP+ (Phase 3) stays the best method. The code stays as an off-by-default option. | Phase 4, three seeds: no setting met D23; the concentrated backdoor (its target case) got worse (21.5% → 39.7% success) because one-round exclusion made attackers look more normal and delayed their isolation; honest parents were falsely flagged in up to 11 of 30 rounds with no attack. |
| **D27** | 2026-11-09 | `Closed (Phase 4b)` | Drill-down v2 turns a flagged parent into evidence against the members that share its unusual direction (leave-one-out alignment vs clients outside the parent), never a one-round exclusion; parents need ≥ 3 members. Comparing each parent with its own history was considered and dropped. | Addresses the three v1 failures found in the logs (exclusion hid the attackers' contrast; the stricter cut couldn't tell members apart; false flags from tiny or odd honest parents). Attackers are present from round 1, so a parent's own history would absorb the attack. |
| **D28** | 2026-11-09 | `Closed (Phase 4b)` | Drill-down v2 is not adopted, and Phase 4 is closed. FedHEP+ (Phase 3) stays the best method; we move to Phase 5 (wider search). | Phase 4b, three seeds: concentrated-backdoor success 21.5% → 1.9%, but it failed the accuracy criterion (−1.1 to −1.7) and the no-attack false-isolation criterion (19 vs 10). Honest members of a FedHEP parent align with each other (alignment z median 1.0, up to 2.8 in flagged parents) in the same range as attackers (3.0–3.6), so no cutoff separates them; a third design would face the same overlap. |
| **D31** | 2026-11-14 | `Updated → D33` | Phase 6 tests layer L0 (influence bound) first and alone, as a Global-step rule `arc_nnm_tm`: ARC, then NNM, then coordinate trimmed mean, f = assumed attackers (3 for 15 clients, 4 for 20); worker momentum kept by the server, β ∈ {0, 0.5}. | L0 addresses both measured failures without screening: it bounds what undetected ALIE / Min-Max attackers can do, and it isolates nobody, so it cannot make false isolations. ARC's paper defines k = ⌊2(f/n)(n − f)⌋ and the clip norm but not ARC's order relative to NNM; clipping first keeps one large row out of its neighbours' averages. The NNM and ARC papers use β = 0.9 per optimization step; here one momentum step is one round of a 30-round run, where β = 0.9 would keep a round's update at more than half weight for 6 rounds, so 0.5 (half-life one round) is tested against no momentum. |

---

## Historical Appendix: Screening Review Memo (2026-11-12)

> **Context (`formerly FOLLOWUP_REVIEW.md`):** This audit was written on `2026-11-12` between Phase 5 and Phase 6 to investigate why `FedHEP+` suffered false isolations under stronger heterogeneity and how to structure the follow-up diagnostic sweeps. Many of the open questions identified below were subsequently resolved in Phase 6 (`D30`–`D33`, `plus.two_sided`, `plus.gap_gate`, and `L0`).

### 1. Six Audit Findings Grounded in Phase 3–5 Runs

1. **MAD collapse is plausible, not established by legacy logs alone.**
   Historical Phase 3 logs recorded normalized $z$ and EMA evidence, but not raw cosine scores or raw MAD. While `FedHEP+` already uses $\max(1.4826\,\mathrm{MAD},\ 0.1\,\lvert\text{median}\rvert)$, that relative floor also shrinks when the median cosine approaches zero. Full-population scoring (`Phase 3c`, `D24`) failed to cure false isolations, showing population shrinkage alone is not the sole cause. Instrumentation added in `rfl/defense/screening.py` now logs raw scores, median, MAD, effective scale, `gap_ratio`, and reference size every round, alongside configurable `plus.mad_floor`.
2. **Temporal smoothing via EMA ($0.7$) vs. consecutive-round patience.**
   An EMA with decay $0.7$ can still isolate a client after a single extreme outlier round ($z < -5$). Independent consecutive-round streak settings `plus.isolation_patience` and `plus.exclusion_patience` were added so isolation requires both a multi-round streak and the EMA threshold.
3. **Isolated honest outliers are not simply the most label-skewed clients.**
   Reconstructing offline ground-truth training-label distributions (`R_skew`) showed that at $\alpha = 0.5$, isolated honest clients had lower mean `R_skew` ($0.443$ vs. $0.530$, i.e., more skewed), but at $\alpha = 0.05$ the relationship reversed ($0.155$ vs. $0.137$). Furthermore, client label histograms are private and self-reported updates are attacker-controlled, so granting looser thresholds based on `R_skew` is neither statistically calibrated nor Byzantinely safe.
4. **Coordinated adaptive attack support (`ALIE` and `Min-Max`).**
   Implemented coordinated parameter-update attacks (`rfl/attacks/adaptive.py`) preserving BatchNorm buffers, supporting both `colluders` (default) and `omniscient` oracle knowledge. Subsequent Phase 6 runs (`2026-11-15`) confirmed that one-sided screening isolates 0% of `ALIE`/`Min-Max` attackers (because their updates sit closer to the consensus than honest clients, giving positive $z$), whereas `plus.two_sided: true` isolates 100% of them by round 3.
5. **Generalization across heterogeneity ($\alpha$) and attacker fraction ($q$).**
   Matched 3-seed means prior to Phase 6 showed that `FedHEP+`'s sign-flip advantage held at 5/15 attackers ($\alpha = 0.5$) but vanished at extreme skew ($\alpha = 0.05$), and backdoor protection weakened at 5 attackers:

| Setting (3 seeds, Phase 3–5) | Defended FedHEP Acc | FedHEP+ Acc | Notes |
|---|---|---|---|
| $\alpha = 0.5$, clean (0 attackers) | **48.39%** | 47.92% | −0.47 pt clean cost |
| $\alpha = 0.5$, 3 sign flippers | 38.08% | **46.34%** | +8.26 pt gain |
| $\alpha = 0.05$, clean (0 attackers) | **65.71%** | 65.18% | High personalization regime |
| $\alpha = 0.05$, 3 sign flippers | **62.81%** | 62.75% | Gain disappears at $\alpha = 0.05$ |
| $\alpha = 0.5$, 5 sign flippers | 31.41% | **44.01%** | +12.60 pt gain |
| $\alpha = 0.5$, 5 backdoor attackers | **45.98%** (79.1% ASR) | 44.93% (56.3% ASR) | Weaker ASR reduction at 5 attackers |

6. **Evaluation scope includes wrongly isolated honest clients.**
   The evaluation exclusion set (`server.unscored`) contains only true attackers (`D14`). Isolated honest clients continue local training on their frozen shared state and remain scored in `pooled_acc`, `mean_client_acc`, and `bottom10`, ensuring false isolations directly penalize the reported accuracy and fairness metrics.

### 2. Diagnostic Artifacts & Commands from the 2026-11-12 Audit

- `scripts/diagnose_screening.py`: Reproducible audit and round-by-round MAD/score plotting without fabricating values for legacy runs.
- `docs/followup/best_legacy/` & `docs/followup/legacy/`: Per-client and per-run CSV/JSON audits across 30 `FedHEP+` runs and 45 readmission/cap runs.
- `configs/sweeps/followup_diagnostics.yaml` & `configs/sweeps/followup_holdout.yaml`: Paired baseline/floor/patience sweeps and the reserved $\alpha = 0.2$, 20-client (`D2`) holdout configuration.

```bash
MPLCONFIGDIR=/tmp/rfl-mpl python scripts/diagnose_screening.py outputs --filter p3__zmr__ --out docs/followup/best_legacy
python scripts/sweep.py configs/sweeps/followup_diagnostics.yaml --dry-run
python scripts/sweep.py configs/sweeps/followup_diagnostics.yaml
```
