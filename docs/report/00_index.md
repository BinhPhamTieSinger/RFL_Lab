# RFL Lab research report

**Robust federated learning for FedHEP.** We reproduced FedHEP and its published defense, tried several improvements, and propose a defense of our own, **FedHEP-Guard**. This report records each step: what was tried, what worked, what failed and why. Read the chapters in order the first time; to find a chart or table quickly, use the presenting kit at the end.

## The story in one minute

1. **FedHEP**, our team's personalized FL model, and its **published defense** were rebuilt and reproduced: every number is within 2 standard errors of the authors' code (chapters 2–3).
2. **The published defense never removes an attacker** and does nothing against a backdoor: 79% backdoor success, the same as no defense (chapter 3).
3. **FedHEP+**, our first model, removes crude attackers: +8.3 points under sign flip, backdoor success 79% → 21%. It leaves three failures: honest clients with unusual data are isolated, adaptive attackers are not seen, and a backdoor that got in stays (chapter 4).
4. **Several ideas failed:** checking groups first, classic robust rules (Median, Bulyan, FLAME, FLTrust), seven Phase 6 components, and the peer reference. Each failure shaped the next design (chapter 5).
5. **FedHEP-Guard fixes failures 2 and 3 at D1 and failure 1 under strong skew.** Adaptive attackers are caught (9 of 9), the plain backdoor stays at 0.3% for 100 rounds, and it holds up best when the model is poisoned (chapters 6–7).
6. **Still open:** under mild skew FedHEP-Guard isolates more honest clients than FedHEP+, mostly whole parent groups. A fix is being tested on the Acer, and the full benchmark runs on the GPU server (chapters 8–9).

![The research path](figures/diagrams/10_research_path.png)

## Chapters

| # | Chapter | Question it answers |
|---|---|---|
| 1 | [Federated learning, attacks and metrics](01_federated_learning.md) | What is FL? Which attacks and settings do we use? What do the numbers mean? Notation |
| 2 | [FedHEP (Phase 1)](02_fedhep.md) | What is FedHEP, what does it change in FedAvg, and did we reproduce it? |
| 3 | [Defended FedHEP (Phases 1c, 2)](03_defended_fedhep.md) | What does the published defense add, and where does it fail? |
| 4 | [FedHEP+ (Phase 3)](04_fedhep_plus.md) | Our first improvement: what it changes, what it fixed, where it breaks |
| 5 | [Ideas that did not work (Phases 4–7)](05_ideas_that_did_not_work.md) | What else was tried, and why each idea failed |
| 6 | [FedHEP-Guard: the model (Phase 6)](06_fedhep_guard.md) | Every mechanism: what it does, its formula, where it acts, why it is there; the exact configuration |
| 7 | [FedHEP-Guard: results](07_fedhep_guard_results.md) | What did it fix from FedHEP+? Every experiment, and every model side by side |
| 8 | [Open problems and current experiments](08_open_problems.md) | What is not solved, why, and what is running now (Phase 8) |
| 9 | [The canonical benchmark](09_benchmark.md) | Every model under the same attacks, settings and machine (results pending) |
| 10 | [Configuration reference](10_configuration_reference.md) | Every config key, symbol, preset, sweep file and run prefix |

## The models at a glance

| Model | Built on | What it adds | Best result | Main weakness |
|---|---|---|---|---|
| FedAvg | — | weighted mean of updates | baseline, 38.1% at D1 | collapses under sign flip and noise |
| FedHEP | FedAvg | three heads (root, parent, local), hierarchy | 47.8% at D1, 65.5% at α = 0.05 | no defense: sign flip −14 points |
| Defended FedHEP | FedHEP | SCCF trust weights, trust ledger | most accurate under attack in Phase 2 | never removes anyone; backdoor 79% |
| FedHEP+ | Defended FedHEP | robust z screen, permanent isolation, regrouping | sign flip +8.3, backdoor 79% → 21% | isolates unusual honest clients; blind to ALIE and Min-Max; backdoor stays |
| **FedHEP-Guard** | FedHEP+ | L0 bound, two-sided union screen, gap gate, patience, rewind, optional exit guard | adaptive attackers 9 of 9 caught; plain backdoor 0.3% over 100 rounds on every seed; best worst case under model poisoning in every GPU setting | under mild skew isolates honest clients, half of them whole parent groups; one late-caught stealth seed at 34%; D2 not yet run |

## Did FedHEP-Guard fix FedHEP+'s failures?

| FedHEP+ failure | Verdict | Evidence |
|---|---|---|
| 1. Honest but unusual clients isolated | **Fixed under strong skew; a new problem under mild skew** | α = 0.1: 1.7 per run against 7.7. α = 0.5: 4.3 and 14.7 against 2.7 and 4.3, mostly whole parent groups |
| 2. Adaptive attackers not seen | **Fixed** | ALIE and Min-Max 9 of 9 caught against 0 of 9; stealth backdoor 0.2–0.5% against 56–75% |
| 3. Backdoors that persist | **Fixed when caught early** | 0.3% over 100 rounds against 5–42% at 30 rounds; not at α = 0.1 or after a late catch |

The full scorecard, with machines and costs, is [chapter 7.1](07_fedhep_guard_results.md).

## Presenting kit: which chart answers which question

Paths starting `figures/` are in `docs/figures/`; `report/figures/` is `docs/report/figures/`.

| Question | Chart | Section |
|---|---|---|
| **Starting point** | | |
| Did we reproduce the paper? | `figures/phase1_parity.png` | 2.4 |
| Why do we need a new defense? | `figures/phase2_accuracy.png`, `figures/phase2_backdoor_asr.png`, `figures/defended_failure_diagnosis.png` | 3.4–3.5 |
| What did FedHEP+ achieve? | `figures/phase3b_confirm.png` | 4.3 |
| Why is FedHEP+ not enough? | `figures/followup_alpha02.png`, `figures/followup_adaptive.png`, `figures/followup_backdoor_persists.png` | 4.5 |
| **What failed** | | |
| Readmission, the cap, the full-population yardstick | `figures/phase3_variants_explained.png` | 4.4 |
| Why not just use FLAME or Bulyan? | `figures/phase5_three_seeds.png` | 5.2 |
| Components dropped in Phase 6 | `figures/phase6_dropped_components.png`, `figures/phase6_signals.png`, diagrams `11`–`17` | 5.3 |
| The peer reference | `figures/phase7_peer_paired.png`, `figures/phase7_peer_outcomes.png` | 5.4 |
| **FedHEP-Guard** | | |
| How does it work? | `report/figures/diagrams/08_fedhep_guard_round.png`, `report/figures/guard_coverage.png` | 6.1 |
| Why is each mechanism in the model? | "Why each part" tables | 2.2, 3.1, 4.1, 6.1 |
| What did it fix from FedHEP+? | `figures/phase6_honest_isolation_summary.png`, `figures/phase6_guard_durability.png` | 7.1 |
| Which part fixes what? | `figures/phase6_d1.png`, `figures/phase6_d2.png` | 7.2 |
| Does it remove a backdoor, over 100 rounds? | `figures/phase6_rewind.png`, `figures/phase6_guard_durability.png` | 7.3 |
| Does it survive an adaptive backdoor? | `figures/phase6_stealth.png` | 7.4 |
| What if the attacker count is unknown? | `figures/phase6_fmis.png` | 7.5 |
| Does it hold at 30 clients and under strong skew? | `figures/phase6_isolation_rounds_n15.png`, `figures/phase6_isolation_rounds_n30.png` | 7.6 |
| Every model side by side, per attack | `figures/all_models_mac_<attack>.png`, `figures/all_models_gpu_<attack>.png` | 7.7 |
| **Open problems** | | |
| Why are honest clients isolated? | `figures/phase6_gap_gate_tradeoff.png`, `figures/phase6_group_isolations.png` | 8.2 |
| What is being tested now? | `report/figures/diagrams/18_shared_signal.png` | 8.3 |
| Which model is best overall? | `report/figures/bench_d1_*.png`, `bench_d2_*.png` (when the GPU results are in) | 9 |

## Conventions

- **Every number** is a mean over seeds 42, 123 and 7 unless marked, from runs stored in `outputs*/`.
- **Every table names its machine.** Methods are only compared within one machine ([chapter 1.3](01_federated_learning.md)).
- **D1** = 15 clients, α = 0.5, 3 attackers. **D2** = 20 clients, α = 0.2, 4 attackers.
- **Diagrams** colour what each model changes against the previous one: green new, yellow changed, red dashed removed, grey unchanged. Their labels are typeset in LaTeX (`scripts/draw_report_diagrams.py`).
- **Click any figure or diagram** in the visualizer to enlarge it; inside the enlarged view use Fit page, Fit width or the zoom buttons, and press Escape to close.

## The lab notebook

This report is the organized account. The raw, dated record stays in `docs/` and is not rewritten:
- `EXPERIMENT_LOG.md`: every run batch, as it happened.
- `DECISIONS.md`: every design decision and why.
- `PREDICTIONS.md`: predictions made before runs, and their outcomes.
- `IDEAS.md`, `RESEARCH_PLAN.md`, `LITERATURE_REVIEW.md`, `PAPER_PLAN.md`.
- `OPERATIONS.md`: machines, commands and the branch workflow.

**Section numbers changed on 2026-11-23**, when chapter 6 was split into three. Older log entries use the old numbers:

| Old | New | Old | New |
|---|---|---|---|
| 6.1–6.7 | 6.1–6.7 | 6.13 | 8.1 |
| 6.8 | 7.2 | 6.14 | 6.8 |
| 6.9 | 7.3 | 6.15 | 5.4 |
| 6.10 | 7.4 | 6.16 | 8.3 |
| 6.11 | 7.5 | 7.1–7.4 | 9.1–9.4 |
| 6.12 | 7.6 and 8.2 | 7.5 | 7.7 |
| | | 8.x | 10.x |
