# 7. The canonical benchmark: every model, one machine

Earlier chapters compare models within one phase, often on different machines. This chapter compares **every model from every phase under the same attacks, settings, seeds and machine**: the comparison to show when someone asks which model is best.

## 7.1 Design

| | |
|---|---|
| Machine | lab GPU server (NVIDIA RTX PRO 4500, CUDA, deterministic kernels) |
| Settings | D1 (15 clients, α = 0.5, 3 attackers) and D2 (20 clients, α = 0.2, 4 attackers) |
| Attacks | none, sign flip, label flip, ALIE, Min-Max, backdoor, stealth backdoor |
| Seeds | 42, 123, 7; the same attackers for every model in a given seed |
| Rounds | 30 |
| $f$ given to robust rules | the true attacker count |
| Sweep files | `configs/sweeps/bench_fedhep_family.yaml` (546 runs), `configs/sweeps/bench_fedavg_family.yaml` (48 runs) |
| Figures and tables | `python scripts/plot_benchmark.py outputs` rewrites the figures below and the tables in this chapter |

**Models:**

| Model | Phase | Chapter | In one line |
|---|---|---|---|
| FedAvg | baseline | 1 | weighted mean, no defense, one head |
| Multi-Krum | baseline | 3 | keeps the 3 most central updates |
| FedHEP | 1 | 2 | three heads, hierarchy, no defense |
| Defended FedHEP | 1 | 3 | SCCF trust weights and trust ledger |
| FedHEP+ | 3 | 4 | robust z screen, permanent isolation, regrouping |
| FedHEP+ drill-down | 4 | 5.1 | FedHEP+ with the v2 parent check |
| Median, Trimmed mean, Bulyan, FLAME, FLTrust | 5 | 5.2 | classic rules at FedHEP's Global step |
| L0 | 6 | 6.2 | ARC → NNM → trimmed mean, no screen |
| R2 | 6 | 6.8 | L0 + two-sided screen + patience |
| R3 | 6 | 6.8 | R2 + gap gate 8 (root-head signal) |
| **FedHEP-Guard** | 6 | 6 | R3 + union screen + rewind |

FedAvg and Multi-Krum use one model, not a FedHEP server, so they face only no attack, sign flip, label flip and the plain backdoor. Every FedHEP-family run also records its accuracy and backdoor success with the depth-exit guard.

## 7.2 How to read the figures

- **Accuracy:** one panel per attack, one bar per model, coloured by phase. The error bar is ± one standard deviation over seeds.
- **Backdoor:** bars are mean success, dots are seeds, and the vertical mark is the same model with the exit guard.
- **Trade-off:** each point is a model. Left panel: accuracy with no attack against worst accuracy under the four poisoning attacks (best: top right). Right panel: accuracy against backdoor success (best: top left).
- **Cost:** honest clients isolated per run with no attack, and seconds per round.

## 7.3 Results

<!-- BENCH:figures START -->
**Status: running on the GPU server.** The figures and tables appear here once its results are pushed and `python scripts/plot_benchmark.py outputs` has run.
<!-- BENCH:figures END -->

### D1: 15 clients, α = 0.5, 3 attackers

<!-- BENCH:d1 START -->
Pending.
<!-- BENCH:d1 END -->

### D2: 20 clients, α = 0.2, 4 attackers

<!-- BENCH:d2 START -->
Pending.
<!-- BENCH:d2 END -->

## 7.4 What the benchmark will decide

These are set before the results, so the results can refute them (in the style of `PREDICTIONS.md`):
1. **FedHEP-Guard against FedHEP+.** FedHEP-Guard wins under ALIE, Min-Max and both backdoors at D1. At D2 it loses less than 1 point with no attack, where FedHEP+ loses 2.
2. **FedHEP-Guard against FLAME and Bulyan.** It matches their backdoor protection (under 2%) at D1, with 3–6 points more accuracy with no attack.
3. **The open question:** whether any model stops the backdoor at D2. None did on the Mac.
