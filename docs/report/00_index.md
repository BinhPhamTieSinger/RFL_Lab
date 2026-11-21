# RFL Lab research report

**Robust federated learning for FedHEP:** what we reproduced, what we tried, what failed and why, and the defense we propose (FedHEP-Guard). Read the chapters in order the first time. Use the presenting kit below to find a chart or table quickly.

## The path of the research

![The research path](figures/diagrams/10_research_path.png)

## Chapters

| # | Chapter | Question it answers |
|---|---|---|
| 1 | [Federated learning, attacks and metrics](01_federated_learning.md) | What is FL? Which attacks and settings do we use? What do the numbers mean? Notation |
| 2 | [FedHEP (Phase 1)](02_fedhep.md) | What is FedHEP, what does it change in FedAvg, and did we reproduce it? |
| 3 | [Defended FedHEP (Phases 1c, 2)](03_defended_fedhep.md) | What does the published defense add, and where does it fail? |
| 4 | [FedHEP+ (Phase 3 and follow-up)](04_fedhep_plus.md) | Our first improvement: what it changes, what it fixed, where it breaks |
| 5 | [Ideas that did not work (Phases 4, 5, 6)](05_ideas_that_did_not_work.md) | Drill-down, classic rules and dropped components, and why each failed |
| 6 | [FedHEP-Guard (Phase 6)](06_fedhep_guard.md) | The current best model: every mechanism, its formula, where it acts, and its results |
| 7 | [The canonical benchmark](07_benchmark.md) | Every model under the same attacks, settings and machine |
| 8 | [Configuration reference](08_configuration_reference.md) | Every config key, symbol, preset, sweep file and run prefix |

## The models at a glance

| Model | Built on | What it adds | Best result | Main weakness |
|---|---|---|---|---|
| FedAvg | — | weighted mean of updates | baseline, 38.1% at D1 | collapses under sign flip and noise |
| FedHEP | FedAvg | three heads (root, parent, local), hierarchy | 47.8% at D1, 65.5% at α = 0.05 | no defense: sign flip −14 points |
| Defended FedHEP | FedHEP | SCCF trust weights, trust ledger | most accurate under attack in Phase 2 | never removes anyone; backdoor 79% |
| FedHEP+ | Defended FedHEP | robust z screen, permanent isolation, regrouping | sign flip +8.3, backdoor 79% → 21% | isolates unusual honest clients; blind to ALIE and Min-Max |
| FedHEP-Guard | FedHEP+ | L0 bound, two-sided union screen, gap gate, patience, rewind, optional exit guard | union screen: plain and adaptive backdoor under 1.5% at D1 (Mac); with the rewind, 0–1% over 100 rounds (Acer) | 4–6 honest clients isolated per run; D2 backdoor still open |

## Presenting kit: which chart or table answers which question

| Question | Chart | Table | Chapter |
|---|---|---|---|
| Did we reproduce the paper? | `figures/phase1_parity.png` | Phase 1 table | 2.4 |
| Why do we need a new defense? | `figures/phase2_accuracy.png`, `figures/phase2_backdoor_asr.png`, `figures/defended_failure_diagnosis.png` | Phase 2 table | 3.4–3.5 |
| What did FedHEP+ achieve? | `figures/phase3b_confirm.png` | FedHEP+ vs Defended, paired by seed | 4.3 |
| What were readmission, the cap and the full-population yardstick, and why were they dropped? | `figures/phase3_variants_explained.png`, `figures/phase3b_backdoor.png`, `figures/phase3c_confirm.png`, `figures/phase3c_backdoor.png` | variants table | 4.4 |
| Why is FedHEP+ not enough? | `figures/followup_alpha02.png`, `figures/followup_adaptive.png`, `figures/followup_backdoor_persists.png` | — | 4.5 |
| Why not just use FLAME or Bulyan? | `figures/phase5_three_seeds.png` | Phase 5 table | 5.2 |
| Why is each mechanism in the model? | — | "Why each part" tables; FedHEP-Guard summary of changes | 2.2, 3.1, 4.1, 5, 6.1 |
| How does FedHEP-Guard work? | `report/figures/diagrams/08_fedhep_guard_round.png`, `report/figures/guard_coverage.png` | summary of changes | 6.1 |
| Which part fixes what? | `figures/phase6_d1.png`, `figures/phase6_d2.png` | ladder tables | 6.8 |
| Why two signals? | `figures/phase6_signals.png` | signals table | 5.3, 6.4 |
| Does it isolate honest clients, and when? | `figures/phase6_isolation_rounds_n15.png`, `figures/phase6_isolation_rounds_n30.png` (compare `figures/phase3_cascade.png`) | grid summary; gap-gate table | 6.12 |
| Does it hold at scale and under strong skew? | — | grid summary per setting | 6.12 |
| Why not just raise the gap gate? | `figures/phase6_gap_gate_tradeoff.png` | — | 6.12 |
| What is being tested next? | `figures/phase7_peer_paired.png`, `figures/phase7_peer_outcomes.png` (when the Acer runs are in) | P22–P25 | 6.15 |
| Does it remove a backdoor? | `figures/phase6_rewind.png`, `figures/phase6_durability.png` | 100-round table | 6.9 |
| Does it survive an adaptive attacker? | `figures/phase6_stealth.png` | stealth table | 6.10 |
| What if the attacker count is unknown? | `figures/phase6_fmis.png` | f table | 6.11 |
| Which model is best overall? | `report/figures/bench_d1_*.png`, `bench_d2_*.png` | benchmark tables | 7 (when the GPU results are in) |

Paths starting `figures/` are in `docs/figures/`; `report/figures/` is `docs/report/figures/`.

## Conventions

- **Every number** is a mean over seeds 42, 123 and 7 unless marked, from runs stored in `outputs*/`.
- **Every table names its machine.** Methods are only compared within one machine ([chapter 1.3](01_federated_learning.md)).
- **D1** = 15 clients, α = 0.5, 3 attackers. **D2** = 20 clients, α = 0.2, 4 attackers.
- **Diagrams** colour what each model changes against the previous one: green new, yellow changed, red dashed removed, grey unchanged. Their labels are typeset in LaTeX (`scripts/draw_report_diagrams.py`).
- **Click any figure or diagram** in the visualizer to enlarge it; inside the enlarged view, click again (or "Actual size") for full resolution, and press Escape to close.

## The lab notebook

This report is the organized account. The raw, dated record stays in `docs/` and is not rewritten:
- `EXPERIMENT_LOG.md`: every run batch, as it happened.
- `DECISIONS.md`: every design decision and why.
- `PREDICTIONS.md`: predictions made before runs, and their outcomes.
- `IDEAS.md`, `RESEARCH_PLAN.md`, `LITERATURE_REVIEW.md`, `PAPER_PLAN.md`.
- `OPERATIONS.md`: machines, commands and the branch workflow.
