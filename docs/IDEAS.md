# Ideas for improving FedHEP's defense

2026-11-16. Each idea: what it is, the evidence for it (published or from our runs), whether it is new, what it costs, and its status in the code. Ranked by expected value for the paper. Nothing here is a result until the runs listed under "Test" are done.

## Where we stand (why these ideas)

| Problem, from our runs | Evidence |
|---|---|
| Honest clients with unusual data are isolated as attackers | α = 0.2: about 9 of 20 per run with no attack (3 seeds + held-out seed) |
| Clever attacks (ALIE, Min-Max) sit *inside* the consensus | one-sided screening isolated none of 9; two-sided isolates all 9 |
| The gap gate fixes false isolations but hides the backdoor | R3: honest isolations 13 → 3; backdoor success 23% → 55% |
| The backdoor lives in the trunk and outlives its attackers | stop test: success keeps rising after poisoning stops, with no defense |
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
**Status.** Not implemented. Wait for `followup_backdoor_heads` (which head carries the backdoor) to decide what to rewind: the whole trunk, or only some layers.

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

## Order of work

1. GPU server, first batch: `gpu_d1_signals` (ideas 1, 2, 5 with fresh baselines) and the exit guard on the backdoor runs (idea 3).
2. Then: idea 4, informed by the per-head results.
3. Then: idea 6, if 1–4 leave a gap.

## Sources

- FedSurrogate, arXiv 2605.11122 — https://arxiv.org/pdf/2605.11122
- AlignIns, CVPR 2025, arXiv 2503.07978 — https://arxiv.org/html/2503.07978v2
- Orion, IJCAI 2023 — https://www.ijcai.org/proceedings/2023/0096.pdf
- BDPFL, arXiv 2503.06554 (layer-wise distillation for personalized FL; reasons that triggers propagate in deeper layers) — https://arxiv.org/html/2503.06554v1
- CrowdGuard, NDSS 2024 — https://arxiv.org/pdf/2210.07714
- FedREDefense, ICML 2024 — https://proceedings.mlr.press/v235/xie24c.html
- BackdoorIndicator, USENIX Security 2024 — https://arxiv.org/html/2405.20862v1
- FedReview, BaFFLe, Basil, FedEraser, FedRecover, FLAME, Zhuang et al. (ICLR 2024): see `LITERATURE_REVIEW.md`
