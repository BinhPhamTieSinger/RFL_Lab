# Ideas for improving FedHEP's defense

Part A (ideas 1–9): ideas from robust FL itself, ranked by expected value for the paper. Part B (ideas 10–19, added 2026-11-17): ideas from neighbouring fields, ranked by how cheaply they can be tried.

# Part A · From robust federated learning

2026-11-16. Each idea: what it is, the evidence for it (published or from our runs), whether it is new, what it costs, and its status in the code. Ranked by expected value for the paper. Nothing here is a result until the runs listed under "Test" are done.

## Where we stand (why these ideas)

| Problem, from our runs | Evidence |
|---|---|
| Honest clients with unusual data are isolated as attackers | α = 0.2: about 9 of 20 per run with no attack (3 seeds + held-out seed) |
| Clever attacks (ALIE, Min-Max) sit *inside* the consensus | one-sided screening isolated none of 9; two-sided isolates all 9 |
| The gap gate fixes false isolations but hides the backdoor | R3: honest isolations 13 → 3; backdoor success 23% → 55% |
| The backdoor lives in the trunk and outlives its attackers | stop test: success keeps rising after poisoning stops, with no defense |
| The shared head carries it; the personal heads resist it | per-head success at D1 (9 runs): root 29–71%, parent 15–45%, local 8–22%; `figures/phase6_backdoor_heads.png` |
| Root-head cosine cannot see the backdoor; whole-model cosine can | FLAME rejects backdoor attackers in 27–28 of 30 rounds; whole-update screening gave 0.3% success (Phase 3, seed 42) |

So the open problem is narrow: **see the backdoor without reintroducing false isolations, and remove what got in before detection.**

## 1. Screen where the backdoor is: layer-level and whole-update signals

**What.** Score each client on the layers where clients disagree most (`plus.signal: critical`, k = 5 per round), or on the whole update (`weights`), under the same two-sided robust z and gap gate as R3.
**Evidence.** FedSurrogate (arXiv 2605.11122, May 2026, preprint) ranks layers by cross-client cosine dispersion and screens only the top 5: CIFAR-100 backdoor success 0.24%, false positives 2.3–9.1%, against FLAME's 48%. Backdoor-critical layers are mostly convolutional (Zhuang et al., ICLR 2024). Our own Phase 3 seed-42 run: whole-update screening gave 0.3% success.
**New?** Layer-level screening is published; combining it with a gap gate inside a hierarchical personalized model is not. The gate is what Phase 3 lacked.
**Cost.** One cosine per layer per client: negligible.
**Status.** Implemented and tested (`critical_layer_scores`; a unit test shows an attack hidden in one small layer stands out while the whole-update cosine barely moves). Queued: `phase6_signal` (Mac, `weights`), `gpu_d1_signals` (GPU server, all signals).
**Risk.** The layer-wise poisoning attack (Zhuang et al., ICLR 2024) targets exactly this; it must be in the adaptive-attack section.

## 2. Sign alignment on the largest coordinates

**What.** `plus.signal: sign_align`: for each client, the share of its top 30% coordinates whose sign agrees with the majority sign.
**Evidence.** AlignIns (CVPR 2025), tested on ResNet-9 / CIFAR-10 and VGG-9 / CIFAR-100 with 20 clients, α = 0.5: backdoor success 1.77% (CIFAR-10 non-IID average over BadNets, DBA, Neurotoxin) with 80.5% accuracy. It is a different kind of statistic from cosine: counts of agreeing signs, insensitive to update scale.
**New?** No (published); a strong, cheap baseline signal.
**Status.** Implemented and tested (sign flippers rank lowest; scaling an update changes nothing). Queued on the GPU server.

## 3. Exits at several depths: refuse the backdoor at prediction time

**What.** Each honest client fits three small classifiers on its own clean data, reading the trunk after `res1`, `layer2` and `layer3`. At test time, if the model's prediction gets little support from those exits, the client uses the exits' vote instead. The threshold flags 5% of the client's own held-out data.
**Evidence.** Orion (IJCAI 2023, single model): on backdoored inputs, shallow exits keep the true label while the deep output switches; 90–99% F1 at flagging backdoor inputs with 0.5–1% clean data, true label recovered about 80% of the time. It fails if the attacker knows the defender's exits (F1 6.7%); here every client fits its own privately, which is the condition under which Orion holds (97.2%).
**New?** Yes, as far as we found: no federated method uses depth exits against backdoors, and it addresses the problem no screen can (a backdoor already in the trunk, which our stop test shows keeps growing). It is also your "multiple outputs from ResNet-9" and "dynamic network" idea in one: the same exits could later serve as early exits to save compute.
**Cost.** Per evaluation, one forward pass over each client's training images and 200 gradient steps on three linear classifiers. No server data, no trigger knowledge, no change to training.
**Status.** Implemented as an evaluation-time add-on (`fedhep.exit_guard: true`, `rfl.defense.exit_guard`); the runner records guarded and unguarded metrics from the same run, so the comparison is exactly paired. Tests: training and the main metrics are unchanged with the guard on; the decision rule re-decides inputs the exits disagree with. On in every run of `gpu_d1_signals`.
**Risks.** (a) Shallow exits on CIFAR-100 are weak, so every flagged clean image may lose accuracy; the clean flag rate is recorded to measure exactly that. (b) A 3×3 corner patch is visible to shallow layers too; if the trunk's early layers carry the trigger, the exits will follow it. (c) An adaptive attacker who knows the scheme could push the trigger into early layers.

## 4. Remove the backdoor after detection (hypothesis H8)

**What.** When a client is isolated, rewind the shared layers to the stored Global model from before its first suspicious round and continue without it (variant a), or also replay stored honest updates (b).
**Evidence.** FedEraser (IWQoS 2021) and FedRecover (S&P 2023) recover models by replaying history without the removed clients. Our stop test makes this necessary: the backdoor grows after poisoning stops.
**Cost.** 0.2 GB per 30-round run to keep the Global models (6.9 MB each); about 4 GB to keep all client updates for replay at 20 clients.
**Status.** Variant (a) implemented and tested (`plus.rewind: trunk | trunk_root | all`, `plus.rewind_window`): on a new isolation the Global model and every parent return, on the chosen coordinates, to the stored model from the start of the client's first suspicious round; that round aggregates nothing. The per-head results say what to rewind: the trunk and the root head. Sweep `gpu_c5_recovery` (36 runs) and the 100-round `gpu_c6_durability`.

## 5. Output-space signals on a probe set

**What.** `probe_logits`: cosine of each uploaded model's root-head predictions on 200 probe images to the median predictions. `head_gap`: the KL gap between the root head and the parent and local heads on those images.
**Evidence.** Prediction-space aggregation is far more robust to untargeted attacks than weight averaging (FedAvg collapses at 10% Byzantine clients; prediction sharing holds at 45%), but plain logit averaging does not stop backdoors (FedDF, FedRAD beaten by A3FL). Probe images never contain the trigger, so these signals measure collateral damage from backdoor training, not the backdoor itself.
**New?** `head_gap` uses FedHEP's structure and is unpublished, but it has a known weakness: all three heads read the same final features, so a trunk backdoor probably shifts all of them together. The per-head success runs will show it directly.
**Status.** Implemented and tested; on the toy problem `probe_logits` catches sign flippers by round 3. Queued on the GPU server.

## 6. Same-parent peer voting ("Among Us")

**What.** Clients of one parent evaluate each other's updates on their own data; ballots are weighted by each voter's own trust from the Global screen; votes become one more piece of evidence, not a verdict.
**Evidence.** The basic mechanism is published: FedReview (per-update peer review, majority vote), BaFFLe (ICDCS 2021, client feedback on the global model), CrowdGuard (NDSS 2024, peer validation of hidden-layer outputs inside secure hardware: 100% detection with 20 clients). All need malicious voters below half. Non-IID data makes honest voters reject honest peers (Basil's authors: local loss cannot separate "malicious" from "different").
**New?** Only the combination: voters restricted to one similarity-formed parent (whose data is alike, which reduces wrong votes) and ballots weighted by independently computed trust. Presented as "clients vote" it would be rejected as not new.
**Cost.** Each client evaluates its parent's other members: about n²/K forward passes per round; models are shared with peers, which must be stated as a privacy cost.
**Status.** Design only. Build after ideas 1–3 are measured, because the backdoor (the remaining failure) is exactly what voting on clean local data struggles to see.

## 7. A self-consistency check that never compares clients

**What.** FedREDefense (ICML 2024): the server distils about 10 synthetic samples per client so that a few training steps on them reproduce the client's update; a client whose update cannot be reproduced (error > 0.6) is malicious.
**Evidence.** 100% detection and no false positives with 100 clients at α = 0.1, down to α = 0.05; benign errors 0.18–0.36 vs attacks 0.78–1.00. Because it never compares clients with each other, non-IID data does not cause false positives, which is our core problem. It misses attacks that genuinely train on poisoned data (label flipping), and probably our backdoor.
**Cost.** A bi-level optimization per client per round: expensive. A candidate once the GPU server timings are known.

## 8. A planted indicator task

**What.** BackdoorIndicator (USENIX Security 2024): the server plants a backdoor-like task on out-of-distribution images in the global model; backdoor training by a client tends to preserve it, so uploads with high indicator accuracy are rejected.
**Evidence.** CIFAR-100: 85% detection, 16% false positives. Needs no in-distribution data; CIFAR-10 images could serve as the out-of-distribution set for CIFAR-100. An attacker who first trains on clean data to erase the indicator lowers detection to 53%.
**Status.** Not implemented; lower priority than 1–4.

## 9. Head-agreement inference

**What.** FedHEP already predicts with three heads on one trunk. When the root head's prediction is not supported by the parent and local heads, let the personal heads decide. It is the exit guard's decision rule applied to heads that already exist: no extra classifiers, no held-out split, no threshold to fit, one forward pass as before.
**Evidence.** Our per-head numbers above. Qin et al. (KDD 2023) explain why: the shared trunk learns the trigger features, but a classifier trained only on clean local data never maps them to the target label (FedRep held BadNet below 10% where FedAvg reached 89%); re-initializing and retraining only the classifier locally cut success by 57% on average.
**New?** The mechanism (a local classifier resists backdoors) is published. Using the disagreement between a shared head and personal heads as a per-input backdoor signal inside a hierarchical ensemble is not, as far as we found.
**Cost.** None at training time; at prediction time, a comparison of three outputs the model already computes.
**Risks.** (a) The local head is the weakest head on data unlike the client's own, so overruling the root head may cost accuracy on rare classes; the bottom-10% metric will show it. (b) Natural-feature triggers (Bad-PFL, DCInject) are learned by the personal heads too, so the signal vanishes against them. Both are stated as predictions P11 and P12 in `PREDICTIONS.md`.
**Status.** Implemented and tested (`head_agreement_logits` in `rfl.client.fedhep`; `fedhep.head_agreement: true` records `history[].head_agreement` = pooled accuracy, bottom-10%, backdoor success and the overrule rate, with the main metrics unchanged). On in every GPU campaign sweep.

## Order of work

1. GPU server, first batch: `gpu_d1_signals` (ideas 1, 2, 5 with fresh baselines) and the exit guard on the backdoor runs (idea 3).
2. Idea 9 (head-agreement inference), re-evaluating the same runs.
3. Then: idea 4, rewinding the trunk and the root head, which the per-head results point at.
4. Then: idea 6, if 1–4 and 9 leave a gap.

The expected outcome of each step, with the number that would refute it, is in `PREDICTIONS.md`.

## Sources (Part A)

- FedSurrogate, arXiv 2605.11122 — https://arxiv.org/pdf/2605.11122
- AlignIns, CVPR 2025, arXiv 2503.07978 — https://arxiv.org/html/2503.07978v2
- Orion, IJCAI 2023 — https://www.ijcai.org/proceedings/2023/0096.pdf
- BDPFL, arXiv 2503.06554 (layer-wise distillation for personalized FL; reasons that triggers propagate in deeper layers) — https://arxiv.org/html/2503.06554v1
- CrowdGuard, NDSS 2024 — https://arxiv.org/pdf/2210.07714
- FedREDefense, ICML 2024 — https://proceedings.mlr.press/v235/xie24c.html
- BackdoorIndicator, USENIX Security 2024 — https://arxiv.org/html/2405.20862v1
- Qin et al., Revisiting Personalized Federated Learning: Robustness Against Backdoor Attacks, KDD 2023 — https://arxiv.org/pdf/2302.01677
- FedReview, BaFFLe, Basil, FedEraser, FedRecover, FLAME, Zhuang et al. (ICLR 2024): see `LITERATURE_REVIEW.md`

# Part B · From outside robust FL

2026-11-17. A deliberately wide search: fields next to robust federated learning whose tools we have not used (model merging, neural collapse, spectral methods, pruning, Shapley valuation, LLM fine-tuning, cryptographic verification). For each: what it is, the evidence, how it would plug into FedHEP, the cost, and the one run that would tell us whether it is worth more. Papers read in full are marked; those seen only by title are marked as such.

## Where each idea would sit

| Idea | Where in FedHEP | Cost to try | Expected gain |
|---|---|---|---|
| 10. Coordinate-wise sign election | Global aggregation, next to L0 | one rule, no data | ALIE and Min-Max without isolating anyone; a second bound to compare with ARC+NNM |
| 11. Fixed ETF root head | the root head | one line at model creation | the backdoor loses the layer it lives in |
| 12. Flip the dormant neurons | end of training, server | small clean set, one pass | removes an implanted backdoor without rewinding |
| 13. Spectral residual of updates | screening signal | a 50-round buffer, one PCA | a signal that sees "new directions", which is what a trigger adds |
| 14. Singular-value signature per layer | screening signal | five numbers per layer | zero-false-positive detection on LoRA adapters; untested on full updates |
| 15. Conformal rejection at inference | prediction time | calibration split per client | a principled version of the exit guard's threshold |
| 16. Conditional screening | FedHEP+ run loop | none | screening only when the round looks attacked: fewer false isolations by construction |
| 17. Shapley contribution scores | evidence source | expensive | treated as a baseline, not a component |
| 18. Retuning test (post-purification robustness) | evaluation, after the final round | no runs | the metric that separates a removed backdoor from a hidden one |
| 19. Anti-backdoor task vector | recovery, after a rewind | shadow data | future work |

## 10. Coordinate-wise sign election instead of a trimmed mean (FedSECA; read in full)

**What.** Each coordinate of the Global update gets a sign elected by a weighted vote (clients weighted by their sign concordance with the others), then is averaged only over the clients whose sign agrees with the elected one. Clients are never discarded whole; a heterogeneous honest client still contributes on the coordinates where it agrees.
**Evidence.** [FedSECA](https://arxiv.org/pdf/2411.03861): the only rule in its table that holds on both ALIE and Min-Max (CIFAR-10 F1 0.72 and 0.83; FedAvg 0.10 and 0.02; TIES-Merging 0.85 and 0.03). It needs no f. Weakness: it converged to a worse solution on one medical dataset (0.51 vs 0.77 with no attack), and pairwise sign comparison is quadratic in clients (456 s per round at 64 clients on CPU, ResNet-18).
**Fit.** Exactly the ALIE puzzle's conclusion: bound, do not isolate. At 15–20 clients the quadratic cost is nothing. The vote is also a natural evidence signal: a client's concordance ratio ρ is a trust score computed with no reference and no f.
**Test.** `robust.rule: sign_election` as a fourth rule in Robust FedHEP and as `plus.global_rule`, D1 ladder under clean, ALIE, Min-Max, sign flip: three seeds, 12 runs. Keep if it matches L0 under ALIE (47.5) and beats it under sign flip (41.7) with no attack within 0.5.

## 11. A fixed simplex-ETF root head (Trojan Cleansing with Neural Collapse; read in full)

**What.** Replace the shared root head's weights with a fixed simplex equiangular tight frame and never train it; the trunk is trained to fit it.
**Evidence.** [arXiv 2411.12914](https://arxiv.org/html/2411.12914v2): backdoored models break neural collapse at the final layer (the target class's weight norm is smaller than the others'); fixing the head to an ETF and fine-tuning the rest on 1–5% clean data drops BadNet success 95% → 3–4% and TrojanNN 99.9% → 0, with accuracy unchanged or up, and holds under an adaptive attacker who knows the head is fixed (2.6%).
**Fit.** Our per-head result says the root head is where the backdoor's mapping lives (root 29–71%, local 8–22%). A fixed ETF root head removes the attacker's ability to move that mapping at all; the trunk can still learn trigger features, but it must map them onto a fixed geometry shared by all classes. The parent and local heads stay trainable, so personalization is untouched. It is also free: one line at model creation, no clean data, no extra pass.
**Risk.** FedHEP's root head is the one all clients share and the Global SCCF screens on; a fixed head gives the root-head cosine signal nothing to look at, so screening would have to move to the trunk (the `critical` or `weights` signals, already queued).
**Test.** `fedhep.root_head: etf` on D1 backdoor and clean, 6 runs. Keep if root-head backdoor success falls below the local head's (under 25%) at a cost under 0.5 points.

## 12. Flip the weight updates of dormant neurons (FLAIN; read in full)

**What.** After training, find the neurons of the first linear layer whose average activation on a small clean set is lowest, and apply their accumulated weight update with the opposite sign (W₀ − ΔW), rescaled to the original norm; widen the set until clean accuracy on the auxiliary set drops by 3.5%.
**Evidence.** [FLAIN, arXiv 2408.08655](https://arxiv.org/html/2408.08655v2): CIFAR-10, α 0.5, DBA: success about 1.8% with at most 5% accuracy loss, where the other defenses in its table fail; IID with 40% attackers: 0.13% success, 0.6% accuracy loss. Not evaluated on CIFAR-100. The mechanism is only argued, not shown: backdoor neurons stay dormant on clean inputs.
**Fit.** A post-hoc removal that needs no stored models, as an alternative to the rewind (H8) when detection comes too late to rewind. Each honest client could run it on its own data for its own heads, which removes the server's need for clean data. ResNet-9's "first linear layer" is the head itself, so the flip acts on exactly the layer our per-head result blames.
**Test.** Server-side on the final Global model of the D1 backdoor runs (post-hoc, no retraining): success and accuracy before and after, 9 models. One afternoon.

## 13. Spectral residual as a screening signal (SpectralKrum; read in full)

**What.** Keep the last 50 aggregated updates, take a rank-50 PCA basis, and score each new update by how much of it lies outside that subspace (its orthogonal residual).
**Evidence.** [SpectralKrum, arXiv 2512.11760](https://arxiv.org/pdf/2512.11760v1): on CIFAR-10 at α 0.1 with 100 clients it did not beat Multi-Krum overall (46.4 vs 50.4) and cost 160 times more per round (1.3 s). The authors' own verdict: spectral defenses help when an attack adds orthogonal variance and not when it stays inside the benign subspace.
**Fit.** Weak as an aggregation rule, but as one more *signal* for our screen it is cheap at our scale and measures something none of our six signals do: whether an update points in a direction the training has never taken. A trigger feature is a new direction; ALIE is not. So it should separate the backdoor from heterogeneity, which is the open problem.
**Test.** `plus.signal: residual` on D1, clean and backdoor, 6 runs. Keep if backdoor attackers rank lowest in most rounds and no honest client is isolated with no attack.

## 14. Singular-value signature of each layer's update (LoRA backdoor detection; read in full)

**What.** For each layer's update, five numbers: top singular value, Frobenius norm, energy share of the top singular value, entropy of the normalized singular values, kurtosis of the entries. Poisoned updates have concentrated singular values with low entropy.
**Evidence.** [arXiv 2602.15195](https://arxiv.org/html/2602.15195): 100% detection, zero false positives on 300 LoRA adapters of three LLM families from weights alone, trigger unknown. Scoped to low-rank adapters; full updates not tested; non-adaptive attacker.
**Fit.** Our `critical` signal already works per layer with cosines; the spectral entropy of a convolutional layer's update (reshaped to out × in·k·k) is one SVD per layer per client, cheap at ResNet-9 size. The claim to test is whether a backdoor update is lower-rank than an honest one on a full model, which nobody has published.
**Test.** Offline first, on the client updates the observer records: spectral entropy per layer for attackers and honest clients in the existing D1 backdoor runs. If they separate, add `plus.signal: spectral`.

## 15. Conformal rejection at prediction time (PRISM-FCP and SD-CSFL; titles and abstracts only)

**What.** Each client calibrates a conformal score on its held-out split so that inputs whose prediction set is empty or too large are rejected or handed to the personal heads. [PRISM-FCP](https://arxiv.org/html/2602.18396v1) makes federated conformal prediction Byzantine-resilient through partial sharing; [SD-CSFL](https://openaccess.thecvf.com/content/WACV2026/papers/Alharbi_SD-CSFL_A_Synthetic_Data-Driven_Conformity_Scoring_Framework_for_Robust_Federated_WACV_2026_paper.pdf) (our own group, WACV 2026) scores conformity with synthetic data.
**Fit.** The exit guard and head-agreement inference both decide with an ad-hoc threshold (the 95th percentile of support). Conformal calibration gives the same decision a coverage guarantee, which is what a reviewer will ask for, and it reuses the 20% held-out split the exit guard already makes. SD-CSFL is the group's own line, so the connection is expected.
**Test.** Replace the exit guard's percentile threshold with a conformal quantile at 1 − α, α ∈ {0.05, 0.1}; same runs, evaluation only.

## 16. Screen only when the round looks attacked (Kick Bad Guys Out; title and abstract only)

**What.** [arXiv 2310.04055](https://arxiv.org/pdf/2310.04055) runs anomaly detection only when a cheap round-level statistic says the round is anomalous, and verifies the server's filtering with a zero-knowledge proof.
**Fit.** Our gap gate already is a conditional activation: nobody is blamed unless the scores split. The paper's framing is the useful part for the write-up: "detection as a conditional step" is an existing category, and the gate is our instance of it, so the related-work section should say so. The zero-knowledge part is out of scope.

## 17. Shapley-style contribution scores (FedSV and relatives; read the baseline it builds on)

**What.** Score each client by the marginal change in validation accuracy its update brings, averaged over subsets.
**Evidence.** The surveyed baseline ([arXiv 2602.21721](https://arxiv.org/pdf/2602.21721.pdf)) evaluates all subsets per round (exponential), needs a representative server validation set, and was tested against one label-flipping client with IID honest clients only; no false-positive rate reported.
**Fit.** Not as a component: it needs server data, and on non-IID data a client with unusual data has a negative marginal contribution on a global validation set by construction, which is our false-isolation problem restated. Worth one line in related work as what we deliberately avoid.

## 18. Post-purification robustness: a metric we must report (Superficial safety; read in full)

**What.** [arXiv 2410.09838](https://arxiv.org/pdf/2410.09838) shows that every purification defense it tested (ANP, I-BAU, FT-SAM, FST, BTI) reaches near-zero backdoor success and then recovers 55–100% of it after five epochs of fine-tuning on 1,000 images containing as few as 1–10 poisoned ones (the "retuning attack"), or after query-only perturbations (66% on average). Only exact purification with the true trigger stays low (2.2%). The explanation, from linear mode connectivity: purified models stop close to the backdoored model along the backdoor-connected path; models that move farther along it (their PAM: 4.6% after retuning on CIFAR-10, 1.0% on CIFAR-100) do not come back.
**Fit.** This is the test the rewind (H8) and the 100-round durability runs must pass, and the paper's own argument predicts the rewind passes it: a model rewound to *before* the attack is not on the backdoor path at all, unlike a model purified *after* it. Head-agreement inference and the exit guard change nothing in the weights, so they are immune to retuning by construction, which is an argument for prediction-time defenses the paper makes for us.
**Test.** Add a post-hoc evaluation to the runner, `eval.retuning: true`: after the final round, fine-tune the Global model for 5 epochs on 1,000 held-out images with 1% carrying the trigger, and record success before and after (P-ASR). No new runs: it reuses the saved final models of every backdoor run. Report P-ASR next to ASR in every backdoor table; it is the number that separates "removed" from "hidden".

## 19. An anti-backdoor task vector instead of a rollback (task arithmetic for merging; read in full)

**What.** [arXiv 2606.12498](https://arxiv.org/pdf/2606.12498): in model merging, subtracting an estimated backdoor vector (IBVS) costs about 21 points of accuracy; *adding* a learned anti-backdoor vector, trained on 10,000 unlabeled shadow images so that features along the interpolation path resist worst-case prompt perturbations, keeps accuracy (58.7 vs 63.6 undefended) and drops success 98% → 0.5% without knowing which source was poisoned or what the trigger is.
**Fit.** A rewind needs stored models; an anti-backdoor vector needs shadow data and a clean reference. FedHEP's server has neither, but the rewound model *is* a clean reference once H8 works, which suggests a two-step recovery: rewind, then train the anti-backdoor vector between the rewound and the current model and apply it instead of discarding the rounds in between. Speculative; worth a line in future work, not a run yet.

## Read and set aside

- [Safe-FedLLM, arXiv 2601.07177](https://arxiv.org/html/2601.07177v1): a probe classifier on LoRA parameter changes for federated LLM fine-tuning (harmful data, not triggers). Its own limitations (probe tied to one backbone and seed, drifts with rounds) rule it out for vision FL. Krum and trimmed mean barely moved its safety metrics, which matches our Phase 5 finding that classic rules do little against targeted harm.
- [AugMP, cai2026graph](https://www.tkn.tu-berlin.de/bib/cai2026graph-preprint/cai2026graph-preprint.pdf): an attack that learns a graph autoencoder over benign LoRA updates to forge malicious updates that pass distance and cosine thresholds (up to 26 points of accuracy lost). A reminder that a defense-aware attack belongs in C8; a learned forgery of our root-head cosine is the strongest adaptive attacker we could face.

## Seen by title only (to read next)

- [FedGraph, ICLR 2026](https://iclr.cc/virtual/2026/10019300): graph-based aggregation against backdoors in federated LLM fine-tuning; a client-similarity graph is close to FedHEP's grouping. The preprint link found so far is a different paper.
- [FedPoisonTTP](https://www.alphaxiv.org/abs/2511.19248.md): poisoning federated test-time adaptation; a threat the exit guard and head-agreement inference would face if clients adapt at test time.

## What this changes in the plan

Three of these are cheap enough to add to the campaign without waiting: idea 11 (a fixed ETF root head, one line), idea 10 (sign election as a fourth rule; a colleague has a first implementation in the Mac working tree and pilots on the Acer), and idea 12 as a post-hoc pass on the saved backdoor models. Ideas 13 and 14 are offline analyses of updates we already record. Idea 18 is not optional: the retuning test is what a 2026 reviewer will ask of any removal claim, and it costs no runs. None of them replaces anything queued; they would form a stage C3b of about 30 runs after `gpu_c2_alie`.

## Sources (Part B)

- FedSECA — https://arxiv.org/pdf/2411.03861
- Trojan Cleansing with Neural Collapse — https://arxiv.org/html/2411.12914v2
- FLAIN — https://arxiv.org/html/2408.08655v2
- SpectralKrum — https://arxiv.org/pdf/2512.11760v1
- Weight-space detection of backdoors in LoRA adapters — https://arxiv.org/html/2602.15195
- SafeSparse (sparsified FL, untargeted poisoning) — https://arxiv.org/html/2505.01454v5
- Contribution scoring baseline — https://arxiv.org/pdf/2602.21721.pdf
- FedNSAM (flatness in FL; no robustness claims) — https://arxiv.org/pdf/2602.23827
- PRISM-FCP — https://arxiv.org/html/2602.18396v1
- SD-CSFL, WACV 2026 — https://openaccess.thecvf.com/content/WACV2026/papers/Alharbi_SD-CSFL_A_Synthetic_Data-Driven_Conformity_Scoring_Framework_for_Robust_Federated_WACV_2026_paper.pdf
- Kick Bad Guys Out — https://arxiv.org/pdf/2310.04055
- Superficial safety of backdoor defenses — https://arxiv.org/pdf/2410.09838
- Task arithmetic for backdoor mitigation in model merging — https://arxiv.org/pdf/2606.12498
- Safe-FedLLM — https://arxiv.org/html/2601.07177v1
- AugMP — https://www.tkn.tu-berlin.de/bib/cai2026graph-preprint/cai2026graph-preprint.pdf
- Titles only: FedGraph (ICLR 2026), FedPoisonTTP (links above)


## Integration follow-up

### 10. Sign election in the Global step

The most consequential detail in the published ablation is what happens **after** the vote. Removing variance reduction and sparsification took CIFAR-10 F1 under ALIE from 0.72 to 0.03. This motivates testing the bounded, sparse update in FedHEP. The authors evaluated five CIFAR-10 clients at Dirichlet α = 1 and measured F1. The lab's D1 comparison uses 15 CIFAR-100 clients at α = 0.5 and measures accuracy. [Benjamin et al., Sections 4–5 and Table 1](https://arxiv.org/html/2411.03861v2)

Set `robust.rule: sign_election` for Robust FedHEP, or `plus.global_rule: sign_election` to aggregate the clients retained by the FedHEP+ screen. Each client votes on the sign of each weight delta, with vote strength determined by pairwise sign concordance. The server clips row norms at their median, clamps each coordinate at the median clipped magnitude, and keeps entries strictly above each client's raw 90th magnitude percentile. It averages the surviving values whose signs won the vote. A tie or empty set gives a zero step on that coordinate. BatchNorm buffers use the existing coordinate median. The logged concordance is a voting weight. A client's contribution to the average varies by coordinate.

This trial uses equations 4–10 with server momentum off to isolate the aggregation rule. The published benchmark used server momentum 0.5. The lab's `robust.momentum` is **client** momentum and serves a different purpose. [Implementation](../src/rfl/defense/classic.py) · [Published equations and settings](https://arxiv.org/html/2411.03861v2)

### Pilot and decision

Two pairs of CPU runs used seed 42, 15 clients, α = 0.5, 1,500 training images, 1,000 test images, four rounds and one local epoch. Both rules trained on the same partition and were scored on the same clients within each condition. The clean and ALIE conditions have different evaluation populations because the runner excludes attackers when scoring accuracy.

| Final accuracy (%) | L0 pooled | Sign election pooled | L0 bottom decile | Sign election bottom decile |
|---|---:|---:|---:|---:|
| Clean | 3.70 | 3.60 | 0.74 | 0.74 |
| ALIE, three participants | 3.13 | 3.51 | 0.37 | 0.83 |

These small, undertrained runs establish that the rule can train end to end. Their paired differences cover one seed. The more informative observation concerns concordance as a trust signal. In clean round 3, honest clients' votes ranged from 0.067 to 1.0. The Spearman correlation across 15 clients between label entropy and voting weight was +0.80 that round (+0.36, +0.65 and +0.62 in rounds 1, 2 and 4). Sample count and label diversity also vary together in this partition. In the ALIE run, all three attackers received a vote of 1.0 in round 1 and 0.867 in rounds 2–4. Across rounds the AUROCs were 0.50, 0.625, 0.417 and 0.542. Use concordance for the ballot only. The pilot gives little support for using low concordance to isolate a client. [Clean L0](../outputs/frontier_pilot_l0_clean_s42/metrics.json) · [Clean election](../outputs/frontier_pilot_sign_election_clean_s42/metrics.json) · [ALIE L0](../outputs/frontier_pilot_arc_nnm_tm_alie_s42/metrics.json) · [ALIE election](../outputs/frontier_pilot_sign_election_alie_s42/metrics.json)

**Full comparison.** Run the 30-round D1 grid after the matching GPU controls finish. It adds 12 sign-election runs across clean, sign flip, ALIE and Min-Max at seeds 42, 123 and 7. The grid references the same 12 L0 control names and configurations as `gpu_d1_signals` and can fill missing controls. Use the same device and output directory for matched comparisons. Report paired final pooled and bottom-decile accuracy differences for every seed. Apply idea 10's clean 0.5-point budget and sign-flip improvement criterion against those controls. Operationalize “matches L0 under ALIE” as mean accuracy within 0.5 points or better. Report Min-Max separately. [D1 sweep](../configs/sweeps/gpu_c3b_sign_election.yaml)

### 18. Does sign election suppress updates for rare classes?

Wang et al. found that separately fine-tuned tasks can rely on largely disjoint subsets of parameters. A possible analogue in FedHEP is a root-classifier row updated by only a few clients who hold that class. Global magnitude sparsification and a cross-client ballot could both suppress a useful class-specific change. The published experiments used task checkpoints. Transfer to successive local FL deltas is an open question. [Localizing Task Information for Improved Model Merging and Compression, ICML 2024](https://arxiv.org/abs/2405.07813)

I replayed the pilot's first clean round and grouped the 100 root-classifier weight rows by the number of clients holding each class. The denominator for each column is the sum of squared raw root-row deltas across the relevant classes and all clients. The first fraction counts that energy on entries above each client's whole-model sparsity cutoff. The second also requires the entry's sign to match the applied Global step. These fractions describe which raw changes pass the filters. Actual step size also depends on clipping, clamping and averaging. [Reproducible diagnostic](../scripts/diag_class_support.py) · [Pilot configuration](../outputs/frontier_pilot_sign_election_clean_s42/config.yaml)

| Clients holding class | Classes | Energy above cutoff | Energy above cutoff and aligned |
|---:|---:|---:|---:|
| 5 | 4 | 97.51% | 73.40% |
| 6 | 8 | 98.15% | 67.71% |
| 7 | 13 | 97.76% | 27.10% |
| 8 | 25 | 97.73% | 44.35% |
| 9 | 23 | 97.51% | 75.69% |
| 10 | 22 | 97.61% | 73.82% |
| 11 | 5 | 97.18% | 75.55% |

Sparsification alone preserved at least 97% of the raw root-row energy in every group. Sign disagreement removed much more, with the largest loss among classes held by seven clients. That pattern gives little support to a simple “fewest clients lose most” explanation in this round. Before changing the rule, repeat the measurement later in training and check whether a deficit in the same classes predicts lower held-out recall than L0 across seeds.

### Revisions to the other frontier tests

- **Fixed root head (11).** FedGELA already uses a globally fixed simplex classifier with local adaptation for partially class-disjoint data. The FedHEP research question is whether fixing only its shared head reduces root and ensemble backdoor success while preserving personal accuracy. Use a frozen, norm-matched random head as a control for freezing alone and use the same trunk-based screen in every arm. The server must enforce the fixed weights and bias when applying uploads. [FedGELA, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/65b721a1df04c1098567f70d483d6468-Abstract-Conference.html) · [Trojan Cleansing with Neural Collapse](https://arxiv.org/abs/2411.12914)
- **Post-hoc pruning and spectral signals (12–14).** The checked-in D1 metrics contain scores and per-round diagnostics. The local output folders examined contain no final weight checkpoints, and the visualizer records samples and norms of updates. For 12, retain the initial head weights, final model and clean calibration data. FLAIN selects low-activation **inputs** to the head, corresponding to columns of its weight matrix. A skewed client calibration set can under-activate features needed by other classes. For 13–14, compute the statistics while the full deltas are in memory. Start a residual basis with prior rounds only and leave enough orthogonal dimensions to measure a residual. [FLAIN, Algorithm 1](https://arxiv.org/html/2408.08655v2) · [Runner](../src/rfl/experiment/runner.py) · [Observer](../visualizer/backend/rfl_viz/observe.py)
- **Conformal rejection (15).** The 20% guard split is drawn from examples already used to train the trunk. A finite-sample claim would require reserving exchangeable clean calibration examples before any model or exit fitting. For a fixed anomaly score and a single evaluation, the corrected rank can bound the marginal clean flag probability. Label-set coverage requires a label-dependent score, and trigger detection still needs empirical validation. FedHEP's head-agreement rule itself has no fitted threshold. [Exit-guard implementation](../src/rfl/defense/exit_guard.py) · [Romano, Sesia and Candès, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html)

### Follow-up on the five title-only references

The authors' abstracts clarify which of these papers changes a FedHEP experiment.

| Paper | What transfers to this lab |
|---|---|
| FedGraph | This ICLR 2026 **workshop** paper measures the structure of a client-similarity graph in federated LLM fine-tuning. A tight cluster and a low individual voting weight are different observations. Compare within-cluster similarity on honest groups with similar label distributions before using the graph to flag clients. [Authors' preprint](https://arxiv.org/abs/2610.18430) |
| Safe-FedLLM | Its lightweight classifiers take LoRA updates as inputs. The lab's existing probes measure model outputs on images. The two approaches require different data to fit a detector. Revisit a learned update classifier if spectral features first separate clients on a seed held out from feature selection. [Safe-FedLLM](https://arxiv.org/abs/2601.07177) |
| Feature-space task arithmetic | LFPM optimizes a corrective task vector along an interpolation path in a merged model. Applying it to one federated checkpoint would require defining the clean reference and the correction objective. The current rewind is a better specified control. [LFPM](https://arxiv.org/abs/2606.12498) |
| Post-purification robustness | Min et al. report re-learning after renewed exposure to poisoned samples. The lab's stop test measures persistence during clean continuation. For rewind, record the clean accuracy and backdoor success immediately after restoring weights, then after 5 and 10 clean rounds on the same seed. A rebound there would establish persistence under the lab's protocol. [Min et al.](https://arxiv.org/abs/2410.09838) |
| FedPoisonTTP | Its setting involves online adaptation and subsequent cross-client sharing. The current head-agreement rule reads frozen predictions, and the exit guard fits locally without uploading its exits. This threat belongs in scope if test-time updates are later shared. [FedPoisonTTP](https://arxiv.org/abs/2511.19248) |
