# 9. Configuration reference

Every setting of every experiment is one key in `src/rfl/experiment/config.py`, written as `section.key` (for example `plus.reference`). An experiment is built in three layers:
1. a **preset** in `configs/*.yaml`;
2. **overrides** from a sweep file or the command line (`--set plus.reference=median`);
3. the **defaults** below, for any key nobody set.

An unknown key is an error, so a typo cannot silently fall back to a default.

## 9.1 Top level

| Key | Default | Values | Meaning |
|---|---|---|---|
| `name` | `experiment` | text | run name; the output folder is `<output_dir>/<name>_s<seed>` |
| `method` | `fedavg` | see 9.2 | the federated algorithm (server and client) |
| `model` | `resnet9` | `resnet9`, `resnet9_multihead` | one head (FedAvg family) or three heads (FedHEP family) |

## 9.2 Methods (`method`)

| Value | Model in this report | Reads sections | Chapter |
|---|---|---|---|
| `fedavg` | FedAvg | `data`, `train`, `run`, `attack` | 1 |
| `fedprox` | FedProx: adds $\frac{\mu}{2}\lVert w - w_t\rVert^2$ to the local loss | + `baseline.fedprox_mu` | 6.2 |
| `scaffold` | SCAFFOLD: control variates correct client drift | — | 6.2 |
| `ditto` | Ditto: a personal model regularized toward the global one | + `baseline.ditto_lambda` | 6.2 |
| `fedrep` | FedRep: shared trunk, private head | + `baseline.fedrep_head_epochs` | 6.2 |
| `multikrum` | Multi-Krum | + `baseline.krum_selected` | 6.4 |
| `fedhep` | FedHEP | + `fedhep` | 2 |
| `fedhep_defended` | Defended FedHEP | + `fedhep` | 3 |
| `fedhep_plus` | FedHEP+, drill-down, R2, R3, FedHEP-Guard (the `plus` keys choose which) | + `fedhep`, `plus` | 4, 5, 7.2 |
| `fedhep_robust` | a classic rule, or L0, at FedHEP's Global step | + `fedhep`, `robust` | 5.2, 7.3 |

## 9.3 `data`

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `dataset` | — | `cifar100` | `cifar100`, or `synthetic` (tests only) |
| `root` | — | `./data` | folder holding CIFAR-100 (the presets use `../Paper_Reproduction/data`; the GPU server passes `data.root=data`) |
| `train_subset` | — | 10000 | training images drawn from CIFAR-100 (seeded) |
| `test_subset` | — | 3000 | test images drawn from CIFAR-100 (seeded) |
| `num_clients` | $n$ | 15 | number of clients |
| `alpha` | $\alpha$ | `null` | Dirichlet concentration of the label split; `null` = IID ([chapter 1.2](01_federated_learning.md)) |

## 9.4 `train` (local SGD)

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `local_epochs` | $E$ | 3 (FedHEP presets: 5) | passes over the client's data per round |
| `batch_size` | — | 32 | mini-batch size |
| `lr` | $\eta_0$ | 0.05 | learning rate in round 1 |
| `min_lr` | $\eta_{\min}$ | 0.001 | learning rate in the last round (cosine schedule) |
| `momentum` | — | 0.9 | SGD momentum |
| `nesterov` | — | true | Nesterov momentum |
| `weight_decay` | — | $10^{-4}$ | L2 weight decay |

## 9.5 `run`

| Key | Default | Meaning |
|---|---|---|
| `rounds` | 25 (Phase 1c+ presets and sweeps: 30) | number of rounds $R$ |
| `eval_every` | 5 | evaluate every this many rounds; the last round is always evaluated |
| `seed` | 42 | seed for data split, attackers, initial weights and training; sweeps set 42, 123, 7 |
| `device` | `auto` | `auto` picks `cuda`, then `xpu`, then `mps`, then `cpu` |
| `output_dir` | `outputs` | where run folders are written |
| `deterministic` | false | deterministic GPU kernels: a rerun on the same device type gives identical numbers (used on the GPU server) |

## 9.6 `attack`

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `kind` | — | `null` | `null` (no attack), `sign_flip`, `gradient_ascent`, `random_noise`, `label_flip`, `backdoor`, `alie`, `min_max`, `stealth_backdoor`. The last three need a FedHEP-family method. Formulas in [chapter 1.4](01_federated_learning.md) |
| `count` | $m$ | `null` | exactly this many attackers, chosen at random by the seed (overrides `rate`) |
| `rate` | $q$ | 0.0 | the authors' rule: each client is an attacker with probability $q$ |
| `knowledge` | — | `colluders` | ALIE and Min-Max statistics from the attackers' own updates (`colluders`) or from the honest updates (`omniscient`) |
| `alie_z` | $z$ | 1.0 | ALIE's step; set to $\Phi^{-1}((n-s)/n)$, i.e. 0.43 at D1 and 0.39 at D2 |
| `stop_round` | — | 0 | attackers train honestly after this round (0 = never); durability tests |
| `placement` | — | `random` | groups at the start: `random`, `concentrated` (all attackers in parent 0) or `spread` (one per parent) |
| `backdoor_target` | — | 0 | the class the trigger maps to |
| `backdoor_fraction` | — | 0.5 | share of an attacker's images that are stamped and relabelled |
| `scale` | $\lambda$ | 1.0 | backdoor upload boost; above 1 is model replacement |

## 9.7 `fedhep` (all FedHEP-family methods)

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `num_groups` | $K$ | 3 | number of parents (groups) |
| `similarity_on` | — | `full` | what FedHEP's grouping compares: `full` flat update (authors, dominated by BatchNorm counters) or `weights` only |
| `exit_guard` | — | false | prediction only: also report accuracy and backdoor success with the depth-exit guard ([chapter 5.7](05_fedhep_guard.md)) |
| `head_agreement` | — | false | prediction only: also report head-agreement inference ([chapter 7.4](07_ideas_that_did_not_work.md)) |

## 9.8 `plus` (method `fedhep_plus`: FedHEP+, drill-down, R2, R3, FedHEP-Guard)

**Signal and reference:**

| Key | Symbol | Default | Values and meaning |
|---|---|---|---|
| `signal` | $s_i$ | `root_head` | what each client is scored on: `root_head` (root-head rows of its classes, as SCCF), `weights` (whole update without BatchNorm, personal heads included), `critical` (the most disputed layers), `weights_critical` (union of the two, FedHEP-Guard), `shared` (whole update without BatchNorm and without the personal heads, Phase 8, 8.3), `shared_critical` (union of `shared` and `critical`), `sign_align` (AlignIns sign agreement), `probe_logits`, `head_gap` (predictions on server probe images) |
| `reference` | $\tilde r$ | `mean_direction` | `mean_direction` (SCCF's mean of unit vectors), `median` (coordinate-wise median) or `peer` (a per-client median over its peers, Phase 7, 7.5, refuted) |
| `peer_share` | $\kappa$ (7.5) | 0.333 | with `reference: peer` (Phase 7): each client's reference is the coordinate median of this share of the other clients, those holding the most similar classes (Jaccard of class masks) |
| `shadow_reference` | — | `""` | also score every update against this reference (`median`, `peer`, `mean_direction`) and log it as `shadow_scores`, without acting on it |
| `critical_k` | $k$ | 5 | layers scored by the critical-layer signal |
| `sign_topk` | — | 0.3 | share of largest coordinates compared by `sign_align` |
| `probe_size` | — | 200 | server probe images for `probe_logits` and `head_gap` |

**Robust z:**

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `relative_floor` | $\rho$ | 0.1 | the scale is at least $\rho\lvert\operatorname{med}(s)\rvert$ |
| `mad_floor` | $a$ | 0.0 | absolute floor of the scale (0 = off) |
| `population` | — | `kept` | z measured against the clients still kept (`kept`) or every non-banned client (`all`, Phase 3c, rejected, 7.1c) |
| `two_sided` | — | false | also flag clients far **above** the others: $z \leftarrow -\lvert z\rvert$ (R2 onward) |
| `gap_gate` | $\gamma$ | 0.0 | above 0: bad evidence only in rounds whose sorted scores split, gap ratio $g \ge \gamma$ (R3: 8) |
| `high_gate` | $\gamma_{\mathrm{high}}$ | 0.0 | two-sided only: a client far **above** the others counts only in rounds with $g \ge \gamma_{\mathrm{high}}$; 0 = same as `gap_gate` (under test: 20, 8.2) |

**Decisions:**

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `z_cut` | — | 6.3 | left out this round when $z_i < -2.5$ |
| `exclusion_patience` | — | 1 | ... in this many consecutive rounds (R2 onward: 3) |
| `evidence_decay` | — | 0.7 | $E_i \leftarrow 0.7E_i + 0.3z_i$ |
| `evidence_cut` | — | 1.5 | isolated for good when $E_i < -1.5$ |
| `isolation_patience` | — | 1 | ... and $z_i < -1.5$ in this many consecutive rounds (R2 onward: 3) |
| `readmit` | — | false | isolated clients are re-scored and return when $E_i > -0.75$ (Phase 3b, rejected: brought back attackers, 7.1a) |
| `max_excluded` | — | 1.0 | at most this fraction left out per round, the lowest evidence first (Phase 3b cap, rejected at 0.34: let attackers back in, 7.1b) |
| `regroup` | — | false | re-form groups by weight-update similarity (FedHEP+ onward: true) |

**Global step:**

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `global_rule` | — | `sccf` | `sccf` (trust-weighted, as Defended FedHEP) or `arc_nnm_tm` (L0, R2 onward) |
| `global_f` | $f_0$ | 3 | attackers L0 assumes before screening; reduced by the clients excluded |
| `global_f_fixed` | — | false | keep $f = f_0$ after exclusions (tested, rejected) |

**Rewind:**

| Key | Default | Meaning |
|---|---|---|
| `rewind` | `none` | on isolation, restore `trunk`, `trunk_root` (FedHEP-Guard) or `all` shared coordinates to the Global model from before the first suspicious round |
| `rewind_window` | 10 | Global models kept for rewinding |

**Drill-down (Phase 4, off by default):**

| Key | Default | Meaning |
|---|---|---|
| `drill_down` | false | flag parents whose mean update is out of line (permutation test) |
| `group_cut` | 2.0 | a parent is flagged when its z < −2 |
| `drill_mode` | `exclude` | `exclude` (v1: stricter member cut this round) or `evidence` (v2: member alignment feeds evidence) |
| `drill_cut` | 1.0 | v1's stricter member cut |
| `min_group` | 2 | parents with fewer eligible members are not checked (v2 runs: 3) |

## 9.9 `robust` (method `fedhep_robust`: Phase 5 rules and L0)

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `rule` | — | `median` | `median`, `trimmed_mean`, `bulyan`, `flame`, `fltrust`, `arc_nnm_tm` (L0) |
| `assumed_attackers` | $f$ | 3 | attackers the rule assumes (trimmed per side, Bulyan, L0) |
| `flame_lambda` | $\lambda$ | 0.001 | FLAME's noise factor |
| `flame_on` | — | `models` | FLAME's cosine distances on `models` (paper) or `updates` |
| `root_size` | — | 100 | FLTrust's clean server images |
| `momentum` | $\beta$ | 0.0 | worker momentum before the rule (Phase 6, rejected at 0.5) |

## 9.10 `baseline` (single-model baselines, authors' values)

| Key | Symbol | Default | Meaning |
|---|---|---|---|
| `fedprox_mu` | $\mu$ | 0.01 | FedProx proximal weight |
| `ditto_lambda` | $\lambda$ | 0.05 | Ditto regularization weight |
| `krum_selected` | $m_K$ | 3 | updates Multi-Krum averages ($f$ is the true attacker count, the authors' oracle) |
| `fedrep_head_epochs` | — | 1 | FedRep head epochs per round |

## 9.11 Presets

| File | Sets |
|---|---|
| `configs/cifar100_fedavg.yaml` | `method: fedavg`, data root, IID |
| `configs/cifar100_fedhep.yaml` | `method: fedhep`, `model: resnet9_multihead`, `local_epochs: 5`, `num_groups: 3`, IID |
| `configs/cifar100_fedhep_defended.yaml` | `method: fedhep_defended`, multi-head model, `alpha: 0.5`, `local_epochs: 5`, `rounds: 30` |

Every other model is a preset plus overrides. The exact overrides of every model in the report, with its name, are in `MODELS` in `scripts/make_benchmark.py`.

## 9.12 Sweep files and run names

A sweep (`configs/sweeps/*.yaml`) is every combination of a few labelled axes, for every seed:

```yaml
seeds: [42, 123, 7]
common: {run.rounds: 30}                         # applied to every run
axes:
  - gg15_a05: {data.num_clients: 15, data.alpha: 0.5, attack.count: 3}
  - clean: {}
    sign: {attack.kind: sign_flip}
  - fedhep: {config: configs/cifar100_fedhep.yaml}   # "config" picks the preset
    guard: {config: configs/cifar100_fedhep_defended.yaml, method: fedhep_plus, plus.signal: weights_critical}
```

- **Run names** join the labels: `gg15_a05__sign__guard_s42`. Later axes override earlier ones; command-line `--set` overrides everything.
- **Running:** `python scripts/sweep.py <file>`. Finished runs (with `metrics.json`) are skipped, so rerunning resumes.
- **On several GPUs:** `scripts/gpu_sweep.py <files...> --procs 1`.

**Prefixes used so far:**

| Prefix | Study | Machine | Chapter |
|---|---|---|---|
| `p2__` | Phase 2 benchmark | Mac | 6.4 |
| `p3__`, `p3b__`, `p3c__` | FedHEP+ and its variants | Mac | 6.5, 7.1 |
| `p4__`, `p4b__` | drill-down | Mac | 7.2 |
| `p5__` | classic rules | Mac | 7.3 |
| `ts__`, `n20__`, `n30__`, `a02__` (FedHEP+ rows) | follow-up diagnostics | Mac, Acer | 6.6 |
| `p6__`, `a02__` (rungs) | the ladder at D1 and D2 | Mac | 6.9 |
| `g1__`, `c2__` | screening signals; ALIE puzzle | GPU | 7.4 |
| `m6__` | backdoor recovery | Mac | 6.10 |
| `x6__` | 100-round durability | Acer | 6.10 |
| `g2__` | the ladder at D2 with the whole-model signals | GPU | 6.9 |
| `c5__`, `c6__` | backdoor recovery (30 rounds) and 100-round durability | GPU | 6.10 |
| `m7__` | stealth backdoor | Mac | 6.11 |
| `x8_` | mis-specified $f$ | Acer | 6.12 |
| `x9__` | FedHEP-Guard over 100 rounds against both backdoors | Acer | 6.10 |
| `x10_` | peer reference against the global median (Phase 7, refuted) | Acer | 7.5 |
| `x11_` | shared signal: union screen with the whole update minus the personal heads (Phase 8) | Acer | 7.6 |
| `x12__` | FedHEP-Guard with the shared signal, 100 rounds, both backdoors (Phase 8) | Acer | 7.6 |
| `x13_` | shared union screen against the union screen at 30 and 50 clients (Phase 8 at scale) | Acer | 8.3 |
| `x14_` | union screen with the high-side gate (`plus.high_gate: 20`) and its controls | Acer | 8.2 (running) |
| `gg15_`, `gg30_`, `gg50_` | large grid (10, 20 and 30% attackers, 8 models) | GPU | 6.13 |
| `gg<n>_a<α>__<condition>__<model>` | the benchmark (`bench_*.yaml`); shares the grid's names, so the grid's 20%-attacker runs are its cells | GPU | 6.1 |
| `gs_` | stealth backdoor beyond D1 | GPU | pending |
| `par__` | device parity | all three | 1.3 |

## 9.13 What a run records

Each run folder holds `config.yaml` (the full resolved configuration) and `metrics.json`:

| Field of `metrics.json` | Content |
|---|---|
| `name`, `method`, `seed`, `alpha`, `device` | what ran, and where |
| `attackers` | the attacker client ids |
| `client_label_stats`, `client_train_sizes` | each client's label mix and data size |
| `history[]` | per evaluated round: `pooled_acc`, `mean_client_acc`, `bottom10`, backdoor success `asr` (backdoor runs), per-head success `asr_heads` (where recorded), exit-guard results `guarded`, head-agreement results `head_agreement`, elapsed time |
| `final` | last-round accuracy (`pooled`), mean client accuracy, bottom-10%, per-client accuracy |
| `detection[]` | per round: flagged clients, precision, recall, F1, AUROC, false-isolation rate, isolated clients |
| `diagnostics[]` | per round (FedHEP family): scores, z, evidence, excluded and isolated clients, groups, gap ratio, rewind target |
| `resources` | seconds per round, total seconds, peak process and device memory |
