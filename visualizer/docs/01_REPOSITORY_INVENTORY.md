# Visualizer, Stage 1: repository inventory

Written 2026-11-15 from the source code of `RFL_Lab/src/rfl` (3,238 lines, read in full), not from the documentation. Where the documentation and the code disagree, the code is described and the disagreement is listed in section 11. Every formula below gives the function that computes it.

## 1. What is in the repository folder

| Folder | What it is | In scope for the visualizer |
|---|---|---|
| `RFL_Lab/` | The maintained implementation: package `src/rfl`, configs, sweeps, tests, 1,100+ recorded runs, documentation. | **Yes.** It is the only implementation every result in the documentation comes from. |
| `Topology-aware-FDL/` | The FedHEP authors' code (with `fedhep_supplementary_material.zip`). | No. It is the reference that `RFL_Lab` is tested against for parity (`tests/test_parity_*.py`), not a second implementation to run. |
| `Paper_Reproduction/` | The earlier notebook-based reproduction, replaced by `RFL_Lab`. Holds the CIFAR-100 data (`data/`) that `RFL_Lab` reads. | Only as the data location. |
| `Papers/`, `Overview/` | PDFs. | No. |

## 2. How a run executes (entry points)

`scripts/run.py` and `scripts/sweep.py` both end in `rfl.experiment.runner.run_experiment(cfg)`:

1. Write `config.yaml`. Seed `random`, NumPy and PyTorch with `run.seed` (`set_seed`).
2. Load data (`rfl.data.cifar100.load_cifar100`): a seeded train subset (default 10,000) and test subset (default 3,000), per-channel normalization, no augmentation. The `synthetic` dataset exists for tests only.
3. Partition train and test pools separately with the same partitioner and seed (`rfl.data.clients.build_clients`).
4. Build the model (`MODELS`) and the method's server (`METHODS[cfg.method]`, `rfl.experiment.registry`).
5. For each round r = 1..R: `server.run_round(r)`, then record detection metrics from `server.report` and, every `eval_every` rounds and in the last round, accuracy and backdoor success.
6. Write `metrics.json`.

The server is the whole method. `run_experiment` never looks inside it except through `report`, `diagnostics`, `unscored`, `predict` and `predict_head`.

## 3. Method inventory

| Method key | Server | Model | Client rule | Grouping | Parent step | Global step | Scored with | Attackers scored |
|---|---|---|---|---|---|---|---|---|
| `fedavg` | `ConsensusServer` | ResNet-9, one head | `SGDTrainer` | none | none | sample-weighted mean (`FedAvg`) | global model | **yes** |
| `fedprox` | `ConsensusServer` | ResNet-9 | `FedProxTrainer` (proximal term μ) | none | none | `FedAvg` | global model | yes |
| `scaffold` | `ConsensusServer` | ResNet-9 | `ScaffoldTrainer` (control variates) | none | none | `FedAvg` | global model | yes |
| `multikrum` | `ConsensusServer` | ResNet-9 | `SGDTrainer` | none | none | `MultiKrum` (mean of the m lowest Krum scores) | global model | yes |
| `ditto` | `ConsensusServer` | ResNet-9 | `DittoTrainer` (global + personal model) | none | none | `FedAvg` | personal model | no |
| `fedrep` | `ConsensusServer` | ResNet-9 | `FedRepTrainer` (shared body, personal head) | none | none | `FedAvg` | personal model | no |
| `fedhep` | `FedHEPServer` | multi-head ResNet-9 | `FedHEPTrainer` | similarity (k-means, regroups) | weighted mean or copy of global, then adaptive momentum | `FedAvg` over all clients | own model, three heads blended | no |
| `fedhep_defended` | `DefendedFedHEPServer` | multi-head | `FedHEPTrainer` | **random, never regroups** | SCCF (k = 2) | SCCF (k = 3, hard rejection) + trust ledger | as FedHEP | no |
| `fedhep_plus` | `FedHEPPlusServer` | multi-head | `FedHEPTrainer` | random, or similarity if `plus.regroup` | SCCF over kept members | SCCF over kept clients, or L0 if `plus.global_rule: arc_nnm_tm` | as FedHEP | no |
| `fedhep_robust` | `RobustFedHEPServer` | multi-head | `FedHEPTrainer` | random, never regroups | SCCF over accepted members | one rule: median, trimmed mean, Bulyan, FLAME, FLTrust or L0 | as FedHEP | no |

Ten methods in four structural families: **consensus** (6, one level), **FedHEP** (1), **Defended FedHEP** (1), and two families built on Defended FedHEP (**FedHEP+**, **robust FedHEP**). A visualization designed for one family is wrong for another; section 4 gives each family's actual sequence.

## 4. Execution specifications

### 4.1 Consensus family (`rfl.server.fedavg_server.ConsensusServer.run_round`)

Per round, clients in id order:
1. Load the global state into a working model.
2. Attacker: `poison_data` (backdoor) and `poison_labels` (label flip).
3. `trainer.train(model, data, round, client_id, targets)`.
4. `upload = trainer.upload(...)` (whole flat state; Ditto and FedRep upload their shared part).
5. Attacker: `poison_upload(upload, start)`.
6. Update = upload − start.

Then `trainer.end_round()` (SCAFFOLD's server control variate), `aggregator.aggregate(updates)`, global ← start + step. Multi-Krum reports the clients it left out as flagged. One level only: no parents.

### 4.2 FedHEP (`rfl.server.fedhep_server.FedHEPServer.run_round`)

1. **Client training** (`_train_all`), clients in id order: load the global model, then the client's parent's `fc2_parent` head, then its own `fc2_local` head (if it has one); attacker hooks; `FedHEPTrainer.train`; keep the trained flat state. Uploads to Global and to the parent are computed separately (random noise draws twice).
2. **Coordinated attack** (ALIE, Min-Max only): after all clients trained, `CoordinatedAttack.poison_batch` replaces the attackers' uploads with one crafted update (section 6).
3. **Grouping step** (`SimilarityGrouping.step`) on the updates: may regroup (section 7). On the first similarity grouping (`Regroup.INITIAL`) all parents reset to the global model and every client's local head is dropped.
4. **Parent step**, per parent with members: if the mean member agreement (`mean_member_similarity`) is above 0.5, the parent becomes a copy of the round-start global model; otherwise start + sample-weighted mean of members' updates. Then adaptive momentum (section 8.2).
5. **Global step**: start + sample-weighted mean of **all** clients' updates (no defense).

### 4.3 Defended FedHEP (`rfl.server.defended_fedhep_server`)

1. Client training and coordinated attack as FedHEP.
2. **Sentinel**: a client with a non-finite upload is isolated for good.
3. **Parent step**, per parent, over active members: parent ← parent + SCCF(members' uploads − parent), norm bound k = 2, no hard rejection; the trust weights feed the trust ledger.
4. **Global step** over senders not isolated: start + SCCF(uploads − start), k = 3, hard rejection of cosine < −0.1; trust feeds the ledger.
5. Groups never change (no grouping step is called; verified by the observation spike).

### 4.4 FedHEP+ (`rfl.server.fedhep_plus_server`)

1. Client training, coordinated attack, sentinel as above.
2. **Scoring** (`rfl.defense.screening.screening_scores`): for each scored client, norm-bound its update (k = 3 × lower-quartile norm), take the signal (root-head rows of the classes it holds, or all weights), cosine to the reference (mean unit update or coordinate-wise median).
3. **Robust z** (`robust_z`), optionally two-sided (`one_sided`), optionally gated (`gap_ratio`, `plus.gap_gate`).
4. **Evidence** (`EvidenceTracker`): exponential moving average of z; isolation when below −`evidence_cut` (with `isolation_patience` consecutive rounds); temporary exclusion when z < −`z_cut` (with `exclusion_patience`). Optional Phase 4 drill-down (`group_z`, `member_alignment`). Cap on exclusions (`apply_cap`).
5. **Kept** = everyone not banned, isolated or excluded.
6. **Grouping step** on kept clients only, if `plus.regroup`.
7. **Parent step**: SCCF over kept members (as Defended FedHEP).
8. **Global step** over kept clients: SCCF, or L0 (ARC → NNM → trimmed mean) with f = `global_f` − excluded.
9. Diagnostics per round: scores, z, median, MAD, scale, floors, gap ratio, evidence, left out, isolated, excluded, groups, Global trust.

### 4.5 Robust FedHEP (`rfl.server.robust_fedhep_server`)

1. Client training, coordinated attack, sentinel (banned).
2. Optional worker momentum per client.
3. **Global step**: the chosen rule on weight rows; BatchNorm buffers by coordinate median. FLTrust first trains a server update on its root set with the global RNG state saved and restored.
4. **Parent step**: SCCF over the members the rule accepted.
5. Diagnostics: rule, accepted, scores, banned.

## 5. Client training (`rfl.client.fedhep.FedHEPTrainer.train`)

Per client per round, one fresh SGD optimizer (Nesterov momentum 0.9, weight decay 1e−4) with a cosine learning-rate schedule over rounds:
$$\eta_r = \eta_{\min} + (\eta_0 - \eta_{\min})\,\tfrac12\Big(1 + \cos\tfrac{\pi (r-1)}{R-1}\Big) \quad \text{(rfl.client.sgd.cosine\_lr)}$$

Label skew of the client's true labels (`label_stats`), with $H$ the label entropy over the $C$ classes:
$$r_{\text{skew}} = \frac{e^{H} - 1}{C - 1} \in [0, 1]$$

Loss weight of each head (`binomial_head_weights`), with $a = \max(1/2K, |Y_i|/C)$, then normalized to sum to 1:
$$\lambda_{\text{root}} = a + (1-a)\,r^2, \quad \lambda_{\text{parent}} = 2r(1-r), \quad \lambda_{\text{local}} = (1-r)^2$$

Per batch: the trunk's features $h$ feed the root and parent heads directly and the local head through $\text{anchor}\cdot h + (1-\text{anchor})\cdot \text{stopgrad}(h)$, anchor = min(1, 10/C) = 0.1 for CIFAR-100. Loss = $\lambda_{\text{root}}\,\text{CE}(z_{\text{root}}, y) + \lambda_{\text{parent}}\,\text{CE}(z_{\text{parent}} \text{ masked to held classes}, y) + \lambda_{\text{local}}\,\text{CE}_{w}(z_{\text{local}} \text{ masked}, y)$ with class weights $w_c \propto 1/\sqrt{n_c}$. Then one SGD step, then a step on the ensemble mixing logits (`_alpha_step`) mixed 50/50 with the heterogeneity prior.

Kept between rounds (`HeadMemory`): ensemble weights, the local head, the last batch's loss per head, $r_{\text{skew}}$, the held-class mask.

**Prediction** (`ensemble_logits`): clients with $r_{\text{skew}} \ge 0.85$ use the root head alone; others blend the three heads with per-batch confidence weights $\text{softmax}(-\text{entropy}_h - \text{loss}_h + \log\alpha_h)$, each head rescaled to the root head's spread; classes the client never trained on are pushed to −10⁴.

## 6. Data and attacks

**Partitioning** (`rfl.data.partition`): IID = shuffle and deal equal shares (remainder dropped). Dirichlet = one draw $p_{k} \sim \text{Dir}(\alpha \mathbf 1)$ per class, each class's samples cut by $p_k$, leftovers round-robin.

**Who attacks** (`rfl.attacks.byzantine`): `attack.count` = exactly that many, drawn from `RandomState(seed)`; else each client with probability `attack.rate` (authors' rule, count varies by seed).

| Attack | Where | What it does (code) |
|---|---|---|
| `label_flip` | training targets | $y \to C-1-y$; label statistics keep the true labels |
| `sign_flip` | upload | start − 1.5 · (trained − start) |
| `gradient_ascent` | upload | start − 5 · (trained − start) |
| `random_noise` | upload | start + N(0, 2²) per coordinate, fresh draw per upload |
| `backdoor` | training data | 3×3 white patch bottom-right on a fixed 50% of the attacker's images, relabelled to the target; optional upload boost `scale` |
| `alie` | all attackers' uploads, after training | weights: mean − z · std of the observed updates (colluders' or honest clients'); BatchNorm buffers kept from each attacker's own upload |
| `min_max` | same | mean − γ · std with the largest γ (bisection) that keeps the crafted update within the observed updates' maximum pairwise distance |

`attack.stop_round`: FedHEP family attackers train honestly after that round. Backdoor success is measured on honest clients' test images of non-target classes with the patch stamped (`rfl.evaluation.backdoor`), overall and per head.

## 7. Grouping (`rfl.grouping.similarity.SimilarityGrouping`)

Random initial groups (`RandomState(seed).randint`). From round `warmup_min` (2), the first similarity grouping happens when the mean cosine of each client's update with its previous one is at least 0.30, or at round `warmup_max` (10). It projects updates to 256 dimensions (`random_project`, Gaussian, seeded), normalizes them and runs 10 iterations of k-means. After that, the grouping is redone whenever more than 15% of clients are closer (cosine, margin 0.15) to another group's mean than to their own. `placement: concentrated | spread` rearranges the initial groups around the attackers.

## 8. Aggregation and defense formulas

### 8.1 FedAvg (`rfl.aggregation.fedavg`)
$$\Delta = \sum_i \frac{n_i}{\sum_j n_j}\,\Delta_i$$

### 8.2 FedHEP parent momentum (`rfl.aggregation.momentum.adaptive_momentum_beta`)
$$\beta = \text{clip}\Big(\frac{V}{V + \|\bar\Delta\|^2}, 0.1, 0.9\Big),\quad V = \tfrac1m \sum_i \|\Delta_i - \bar\Delta\|^2,\quad \theta_p \leftarrow \beta\,\theta_p + (1-\beta)\,\theta_p^{\text{new}}$$

### 8.3 SCCF (`rfl.defense.soft_cosine.SoftCosineAggregator`)
BatchNorm buffers: coordinate median. Weights: cap each row's norm at $k \times$ the lower-quartile norm; unit vectors $u_i$ (root-head rows of held classes when class masks are given); centroid $\bar u$; if $\|\bar u\| < 0.2$ everyone gets $1/n$; else
$$t_i = \text{softmax}_i\big(\cos(u_i, \bar u)/T\big),\quad T = 0.5,\ \times 0.9 \text{ if } \text{Var}(\cos) > 0.05 \text{ else } \times 1.05$$
Global tier: cosines below −0.1 get weight 0. Result: $\sum_i t_i\,\tilde\Delta_i$.

### 8.4 Trust ledger (`rfl.defense.trust_ledger`)
$$s_i \leftarrow 0.85\,s_i + 0.15\,\min(1, t_i n),\quad \text{isolated for good if } s_i < 0.25$$

### 8.5 FedHEP+ screening (`rfl.defense.screening`)
$$z_i = \frac{s_i - \operatorname{med}(s)}{\max\big(1.4826\,\text{MAD}(s),\ \rho\,|\operatorname{med}(s)|,\ \varepsilon_{\text{abs}}\big)}$$
with $\rho$ = `relative_floor` (0.1) and $\varepsilon_{\text{abs}}$ = `mad_floor` (0); all zeros with fewer than 5 scores. Two-sided: act on $-|z_i|$. Gap gate: act only if $\max_j(g_j)/\operatorname{med}_j(g_j) \ge$ `gap_gate`, $g_j$ the gaps between neighbouring sorted scores. Evidence $E_i \leftarrow 0.7\,E_i + 0.3\,z_i$; isolate if $E_i < -1.5$; leave out this round if $z_i < -2.5$.

### 8.6 Global-step rules (`rfl.defense.classic`)
Coordinate median; trimmed mean (drop f per side); Bulyan (θ = n − 2f by repeated Krum, then per coordinate the β = θ − 2f values nearest the median; needs n ≥ 4f + 3); FLAME (HDBSCAN on cosine distances, min cluster n/2 + 1, clip to the median distance S, add N(0, (λS)²)); FLTrust (trust ReLU(cos(Δᵢ, Δ₀)), rescale to ‖Δ₀‖); L0:
$$k = \lfloor 2 (f/n)(n-f) \rfloor \text{ largest rows clipped to the } (k{+}1)\text{-th norm (ARC)},\quad \hat\Delta_i = \tfrac{1}{n-f}\sum_{j \in \mathcal N_{n-f}(i)} \Delta_j \text{ (NNM)},\quad \text{then trimmed mean}$$

### 8.7 Multi-Krum (`rfl.aggregation.multikrum`)
Score = sum of squared distances to the n − f − 2 nearest others; mean of the m lowest. f = the true number of attackers (the authors' oracle).

## 9. Evaluation and detection metrics

- `pooled`: correct predictions over all scored clients' test images (sample-weighted mean). `mean_client`: plain mean over clients. `bottom10`: mean of the lowest ⌈10%⌉ client accuracies (`rfl.evaluation.accuracy`).
- Backdoor success: share of stamped non-target test images predicted as the target, honest clients only; per head (root, parent, local) for the FedHEP family.
- Detection per round (`rfl.evaluation.detection`): flagged set; precision, recall, F1; AUROC of the trust scores; false positives; malicious groups (majority attackers) and, with drill-down, group precision and recall.

## 10. Configuration inventory

| Section | Fields | Read by |
|---|---|---|
| `data` | `dataset` (cifar100, synthetic), `root`, `train_subset`, `test_subset`, `num_clients`, `alpha` (null = IID) | all |
| `train` | `local_epochs`, `batch_size`, `lr`, `min_lr`, `momentum`, `nesterov`, `weight_decay` | all |
| `run` | `rounds`, `eval_every`, `seed`, `device`, `output_dir` | all |
| `attack` | `kind`, `knowledge`, `alie_z`, `rate`, `count`, `stop_round`, `placement`, `backdoor_target`, `backdoor_fraction`, `scale` | all (`alie`, `min_max`, `stop_round`, `placement` only FedHEP family) |
| `fedhep` | `num_groups`, `similarity_on` | FedHEP family (`similarity_on` only `fedhep`) |
| `plus` | 22 fields (section 4.4) | `fedhep_plus` only |
| `robust` | `rule`, `assumed_attackers`, `flame_lambda`, `flame_on`, `root_size`, `momentum` | `fedhep_robust` only |
| `baseline` | `fedprox_mu`, `ditto_lambda`, `krum_selected`, `fedrep_head_epochs` | the matching consensus method |
| top level | `name`, `method`, `model` (resnet9, resnet9_multihead) | all |

Validation already in the code: unknown keys and wrong types are rejected by `load_config` (`_build`); several constructors raise `ValueError` on bad values (attack kind, placement, rule, alpha ≤ 0, momentum, global rule). The model must match the method (multi-head for the FedHEP family). Coordinated attacks require a FedHEP-family method.

## 11. Discrepancies and uncertainties found

| # | Finding | Consequence |
|---|---|---|
| D-1 | Consensus methods (FedAvg, FedProx, SCAFFOLD, Multi-Krum) score **all** clients including attackers; FedHEP-family methods and Ditto, FedRep exclude attackers (`unscored`). This follows the authors' code. | FedAvg's accuracy under attack is measured on a slightly different client set than FedHEP's. The documents compare them directly; the difference should be stated in the paper. |
| D-2 | At test time, clients with $r_{\text{skew}} \ge 0.85$ predict with the root head alone. The plain-language summary says the prediction "combines all three heads". | Under IID data every client has $r_{\text{skew}}$ between 0.91 and 0.95 (15 clients, CIFAR-100, computed with `label_stats`), so FedHEP's personalization is switched off at test time there. Relevant to the new IID sweep. |
| D-3 | `PlusConfig` defaults (`reference: mean_direction`, `regroup: false`) differ from what every reported FedHEP+ run uses (`median`, `true`). | The visualizer must show the configuration actually used, never the defaults. |
| D-4 | `attack.alie_z` defaults to 1.0; every reported ALIE run uses 0.43 (computed for n = 15, m = 3). | Same as D-3. |
| D-5 | Defended FedHEP and robust FedHEP never call the grouping step. | No regrouping phase may be drawn for them (confirmed by the observation spike, section 12). |
| D-6 | `predict_head` (added 2026-11-14) lacked `@torch.no_grad()`: evaluation built unused gradient graphs. | Fixed 2026-11-15. Results unaffected; evaluation was slower and used more memory. |
| D-7 | `fedhep_robust` with FLTrust trains a server update each round on a root set drawn outside the clients' data. | The visualizer must show this as a server-side training step for that rule only. |

## 12. What can be observed without changing the research code

Tested by a spike (`visualizer/backend/rfl_viz/observe.py`, `visualizer/backend/tests/test_observe_is_passive.py`): the visualizer wraps the builders in `METHODS` so that each server built by the unchanged `run_experiment` gets instance-level wrappers on `run_round`, `trainer.train`, `attack.poison_upload`, `grouping.members`, `grouping.step` and both SCCF aggregators. The wrappers only read arguments and results and measure wall time; they draw no random numbers. **For FedHEP+, Defended FedHEP and FedHEP with a backdoor, the observed run's `metrics.json` is identical to an unobserved run** (all fields except wall time), while the observer collects per-round, per-client, per-attack and per-aggregation events.

| Observable passively, exactly | How |
|---|---|
| Partition: samples and label histogram per client | `clients` after `build_clients` |
| Attackers and their kind | server attributes |
| Per client per round: training time, $r_{\text{skew}}$, last-batch loss per head, ensemble weights, whether labels were poisoned | `trainer.train` wrapper + `HeadMemory` |
| Each attack's effect: clean vs sent update norm and cosine | `poison_upload` / `poison_batch` wrappers |
| Grouping events and assignments | `grouping.step` wrapper |
| SCCF trust weights per parent and at Global | aggregator wrappers |
| Screening scores, z, median, MAD, scale, gap ratio, evidence, isolations | `server.diagnostics` (written by the algorithm itself) |
| Global-step rule acceptances and scores | `server.diagnostics` |
| Accuracy, backdoor success, detection metrics, resource use | the runner's own `metrics.json` |
| Stdout | the runner's own prints |

| Not observable without a change to the research code | Options |
|---|---|
| Loss of every mini-batch (only the last batch per head is kept) | (a) show last-batch losses only; (b) an opt-in, default-off callback in the trainers. |
| Per-batch gradients and optimizer internals | Optimizer step count and timing are available through PyTorch's global step hook; values would need (b). |
| Intermediate quantities inside SCCF (unit vectors, centroid norm, temperature) | Recompute from the same inputs with the same functions, labelled "recomputed". |
