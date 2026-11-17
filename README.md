# RFL Lab

A modular PyTorch framework for robust federated learning research. It reproduces FedHEP (Federated Hierarchical Ensemble Personalization) and its defended variant, benchmarks them against established robust aggregation rules under poisoning and backdoor attacks, and serves as the test bench for new defenses.

All results, including negative ones, are recorded in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md). No number in this repository is reported without the run that produced it (`outputs/<run>/metrics.json` and `config.yaml`).

## Status

| Phase | Outcome |
|---|---|
| 1. Reproduction | FedAvg, FedHEP and Defended FedHEP match the authors' code: identical outputs in parity tests, and three-seed CIFAR-100 means within two standard errors. Five baselines (FedProx, SCAFFOLD, Ditto, FedRep, Multi-Krum) pass parity tests. |
| 2. Benchmark harness | Seven attacks, attacker placement control, per-round detection metrics, backdoor success rate, resumable sweeps. |
| 3. FedHEP+ | Robust z-score screening with a median reference, permanent isolation and regrouping. Better than Defended FedHEP on all three seeds under sign flipping (+8.3 points), label flipping (+3.1) and backdoors (success 79% to 21%), at a cost of 0.5 points with no attack. |
| 4. Drill-down defense | Parent-level detection works; member-level attribution does not. Not adopted. |
| 5. Established rules inside FedHEP | Median, trimmed mean, Bulyan, FLAME and FLTrust in FedHEP's Global step. None beats FedHEP+ overall; Bulyan and FLAME stop backdoors (0.5% success) but cost 3.9 to 5.6 points with no attack. |
| Follow-up (done) | Screening diagnostics: the MAD grows rather than shrinks. Adaptive attacks: one-sided screening isolates none; two-sided screening isolates every ALIE and Min-Max attacker. α = 0.2, 20 clients: FedHEP+ isolates about 9 of 20 honest clients with no attack and loses 2 points to Defended FedHEP. A backdoor keeps growing after poisoning stops, even with no defense. |
| 6. FedHEP-Guard (running) | Layer L0 (ARC, NNM, trimmed mean) alone: best defense at α = 0.2 with zero false isolations, but fails sign flip at α = 0.5. Next rungs queued: L0 + two-sided screening, then a gap gate that blames nobody unless the round's scores split. Per-head backdoor success logged to find where the backdoor lives. |
| Next | Literature review of 170 sources and the research plan for a new defense: [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md), [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md). |

Phase 5 results (CIFAR-100, Dirichlet α = 0.5, 15 clients, 3 attackers, 30 rounds, mean ± sd over 3 seeds). The held-out follow-up shows FedHEP+'s advantage does not yet carry over to stronger heterogeneity; see the experiment log.

| Method | No attack | Sign flip | Label flip | Backdoor success |
|---|---|---|---|---|
| Defended FedHEP | 48.39 ± 0.72 | 38.08 ± 1.01 | 43.23 ± 0.40 | 78.7% |
| FedHEP+ | 47.92 ± 0.60 | 46.34 ± 0.87 | 46.34 ± 0.87 | 20.8% |
| Bulyan | 44.52 ± 0.28 | 44.13 ± 0.76 | 43.94 ± 1.23 | 0.5% |
| FLAME | 42.81 ± 1.49 | 42.72 ± 2.50 | 41.61 ± 2.28 | 0.5% |

FedHEP+ gives identical results under sign flip and label flip because it leaves all three attackers out from round 1 in both cases, so the honest clients train identically; accuracy is measured on honest clients' test data.

![Accuracy over three seeds](docs/figures/phase5_three_seeds.png)

## Documentation

| Document | Contents |
|---|---|
| [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) | Plain-language account of every stage so far, with the key charts. Start here. |
| [`visualizer/docs/`](visualizer/docs/) | The RFL Lab visualizer (in progress): repository inventory and architecture. |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | How to set up either machine, run and queue sweeps, move results between the Mac and the Acer, and push. |
| [`docs/PREDICTIONS.md`](docs/PREDICTIONS.md) | Written before the GPU campaign: what each queued experiment is expected to show, the number that would refute it, the model we expect to publish, and the campaign plan. |
| [`docs/IDEAS_FRONTIER.md`](docs/IDEAS_FRONTIER.md) | Ideas from outside robust FL (model merging, neural collapse, spectral methods, pruning, conformal prediction), each with where it would plug into FedHEP and the run that would test it. |
| [`docs/IDEAS.md`](docs/IDEAS.md) | Ranked ideas for the next defense, with the evidence for each: layer-level and sign signals, depth exits, recovery, peer voting. |
| [`docs/PAPER_PLAN.md`](docs/PAPER_PLAN.md) | Claims and their status, closest published work, paper outline with figures mapped to runs, experiments needed before submission. |
| [`docs/LITERATURE_REVIEW.md`](docs/LITERATURE_REVIEW.md) | Literature review: aggregation rules, attacks, backdoor and distillation defenses, personalized and hierarchical FL, efficiency, privacy, evaluation standards, open gaps. |
| [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md) | Proposed method, hypotheses, experiment stages and efficiency plan for the next paper. |
| [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) | Every experiment: setting, results, charts, interpretation. Starts with diagrams and formulas for each method. |
| [`docs/methods_page/rfl_lab_methods.html`](docs/methods_page/rfl_lab_methods.html) | Self-contained page with the method diagrams and rendered formulas. |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Package layers, interfaces and the import rule. |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Phases, exit criteria and progress. |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Numbered design decisions with their reasons. |

## Repository layout

```
src/rfl/
  core/         shared types and the interfaces every mechanism implements
  data/         CIFAR-100, IID and Dirichlet partitioning
  models/       ResNet-9 and the multi-head FedHEP model
  attacks/      label flip, sign flip, gradient ascent, noise, backdoor, ALIE, Min-Max
  client/       local training rules: SGD, FedProx, SCAFFOLD, Ditto, FedRep, FedHEP
  grouping/     client-to-parent assignment by similarity
  defense/      SCCF soft cosine, trust ledger, robust z-score screening, classic rules
  aggregation/  FedAvg, parent momentum, Multi-Krum
  evaluation/   accuracy, backdoor success rate, detection metrics
  server/       round orchestration (composes the mechanisms above)
  experiment/   configuration, registry, runner, sweeps
configs/        one YAML file per method; configs/sweeps/ holds experiment grids
scripts/        run, sweep, summarize, plot, quality gates
tests/          property-based and parity tests
docs/           documentation, figures and diagram sources
outputs/        metrics and configuration of every completed run
```

## Installation

Requires Python 3.10 or later and PyTorch 2.2 or later.

```bash
git clone https://github.com/BinhPhamTieSinger/RFL_Lab.git
cd RFL_Lab
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/check.sh
```

CIFAR-100 is read from `../Paper_Reproduction/data` (set `data.root` in the configuration to change it). The device is chosen automatically in the order CUDA, Intel XPU, Apple MPS, CPU; set `run.device` to override.

## Running experiments

A single configuration, three seeds:

```bash
python scripts/run.py configs/cifar100_fedhep.yaml --seeds 42 123 7 \
    --set data.alpha=0.05 name=cifar100_fedhep_a0.05
python scripts/run.py configs/cifar100_fedhep_defended.yaml --seeds 42 123 7 \
    --set attack.kind=sign_flip attack.count=3 name=cifar100_def_sign_flip
```

Each seed writes `outputs/<name>_s<seed>/config.yaml` and `metrics.json` (per-round accuracy, bottom-10% client accuracy, backdoor success rate, detection precision, recall and F1, and the device used).

A sweep (a grid of methods, attacks and seeds; interrupted sweeps resume where they stopped):

```bash
python scripts/sweep.py configs/sweeps/followup_attacks.yaml --dry-run   # list the runs
python scripts/sweep.py configs/sweeps/followup_attacks.yaml             # run
python scripts/summarize.py outputs --filter n20__                       # mean ± sd over seeds
python scripts/plot_results.py outputs                                   # redraw docs/figures
python scripts/plot_followup.py outputs
```

All methods in one comparison must run on the same machine: results for the same seed differ slightly between GPU backends.

## Quality gates

`scripts/check.sh` runs every gate; all must pass before a result is recorded.

| Gate | Guarantees |
|---|---|
| ruff | Style and common bug patterns. |
| mypy | Type consistency across interfaces. |
| import-linter | The layer rule in `docs/ARCHITECTURE.md`; an upward import fails the build. |
| pytest with Hypothesis | Each mechanism's stated invariants hold on generated inputs. Parity tests require results identical to the authors' reference code (folder `../Topology-aware-FDL`, or set `RFL_REFERENCE`); they are skipped when the reference is absent. |
| graphify | Regenerates the code graph in `graphify-out/` so structural drift is visible. |

## Reproducibility

- Every run is fully described by its saved `config.yaml` and seed.
- Compared methods share data partitions, attacker identities and round budgets for a given seed.
- Results are reported as mean ± standard deviation over seeds 42, 123 and 7 unless stated otherwise.
