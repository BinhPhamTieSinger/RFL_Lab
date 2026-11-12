# RFL Lab

A modular framework for robust federated learning experiments. Its first job is to rebuild FedHEP cleanly and then improve it, starting with a drill-down defense in which Global identifies malicious parents and then checks the clients inside them.

**Status:** Phase 1 ports pass parity tests; FedAvg, FedHEP and Defended FedHEP also pass on CIFAR-100. Phase 2 testbench (attacks, placement, detection and backdoor metrics, sweeps) is built; the first benchmark needs your Mac. See `docs/ROADMAP.md` and `docs/EXPERIMENT_LOG.md`.

## Layout
```
src/rfl/
  core/         shared types + the interfaces every mechanism implements
  data/         datasets, partitioners
  models/       backbones and multi-head wrappers
  attacks/      label flip, sign flip, gradient ascent, noise, backdoor ✅
  client/       local training rules
  grouping/     client → parent assignment (similarity grouping ✅)
  defense/      SCCF soft-cosine ✅, trust ledger ✅, screening (robust z) ✅, drill-down defense
  aggregation/  FedAvg ✅, parent momentum ✅, robust aggregators
  evaluation/   metrics
  server/       round orchestration (composes the above, no algorithm logic)
  experiment/   config, registry, runner
tests/          property-based tests (Hypothesis)
docs/           ARCHITECTURE.md · ROADMAP.md · DECISIONS.md · EXPERIMENT_LOG.md
scripts/check.sh  runs every quality gate
graphify-out/   code graph: open graph.html in a browser, read GRAPH_REPORT.md
```

## Set up (macOS)
```bash
cd ~/Documents/AI_researching/Robust_Federated_Learning/RFL_Lab
source ../.venv/bin/activate
pip install -e ".[dev]"
bash scripts/check.sh
```

## Run an experiment
```bash
python scripts/run.py configs/cifar100_fedavg.yaml --seeds 42 123 7                      # IID
python scripts/run.py configs/cifar100_fedavg.yaml --seeds 42 123 7 \
    --set data.alpha=0.05 name=cifar100_fedavg_a0.05                                     # extreme skew
python scripts/run.py configs/cifar100_fedhep.yaml --seeds 42 123 7 \
    --set data.alpha=0.05 name=cifar100_fedhep_a0.05                                     # FedHEP
python scripts/run.py configs/cifar100_fedhep_defended.yaml --seeds 42 123 7 \
    --set attack.kind=sign_flip attack.rate=0.2 name=cifar100_def_sign_flip_q20         # Defended FedHEP under attack
```
Each seed writes `outputs/<name>_s<seed>/config.yaml` and `metrics.json`.

## Benchmark sweeps
```bash
python scripts/sweep.py configs/sweeps/phase2_core.yaml --dry-run   # list the runs
python scripts/sweep.py configs/sweeps/phase2_core.yaml             # run (resumes if interrupted)
python scripts/summarize.py outputs --filter p2__                   # mean ± sd table over seeds
python scripts/plot_results.py outputs                              # redraw the charts in docs/figures
```
Each run's `metrics.json` also holds per-round detection metrics (`detection`) and, for backdoors, attack success rate (`history[].asr`).

## Quality gates (`scripts/check.sh`)
| Gate | Guarantees |
|---|---|
| ruff | style and common bug patterns |
| mypy | type consistency across interfaces |
| import-linter | the layer rule in `docs/ARCHITECTURE.md`; an upward import fails the build |
| pytest + Hypothesis | each mechanism's stated invariants hold on hundreds of generated inputs; parity tests require identical results to the authors' code (folder `../Topology-aware-FDL`, or set `RFL_REFERENCE`) |
| graphify | regenerates the code graph so structural drift is visible |

The reference implementation (the authors' code) stays untouched in `../Topology-aware-FDL`.
