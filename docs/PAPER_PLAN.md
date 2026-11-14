# Paper plan

Status 2026-11-14. This document turns the results so far into a paper: what can be claimed today, what still needs experiments, how the paper differs from the closest published work, and the outline with every figure mapped to the runs behind it. Method details are in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md); results in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); the broad literature in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).

**Rule for this document:** a claim is listed as supported only when completed runs show it. Everything else is marked pending, with the experiment that decides it.

## 1. Thesis

Working thesis (the second half is pending):

> In hierarchical personalized federated learning, defenses that judge a client by its distance from the consensus fail in both directions: they systematically remove honest clients whose data is unusual, and they cannot see attackers that hide inside the consensus. A defense that bounds every client's influence and separates "different" from "malicious" with more than one signal *[pending: FedHEP-Guard]* keeps accuracy and fairness under both crude and adaptive attacks at a measured, small cost.

Working titles (pick after the method results):
1. *Different Is Not Malicious: Robust Hierarchical Personalized Federated Learning under Data Heterogeneity*
2. *When Robust Aggregation Punishes the Honest Minority: A Study and a Defense for Hierarchical Personalized FL*

## 2. Claims and their status

| # | Claim | Status | Evidence |
|---|---|---|---|
| C1 | Distance-based screening isolates honest clients systematically under non-IID data: the same clients, early in training, whatever the attack. | **Supported** | α = 0.2, 20 clients: about 9 of 20 honest clients isolated per run with no attack, first in round 5, on 3 seeds and one held-out seed; the same clients flagged across attacks (log, 2026-11-14 entries). |
| C2 | Those false isolations cost accuracy and fairness, not just a few points of average accuracy. | **Supported** | α = 0.2, no attack: −2.0 accuracy on every seed and bottom-10% 42.25 to 38.37 against Defended FedHEP. |
| C3 | The robust z-score's scale collapses during training, so a single round can isolate an honest client for good. | **Supported** | Logged MAD and floor per round; z down to −11.4 late in training (log, 2026-11-14 morning). |
| C4 | In-distribution attacks (ALIE, Min-Max) sit *above* the consensus; none of the tested defenses isolates them. | **Supported** at α = 0.5 (3 seeds; 30 runs, 5 defenses); at α = 0.2, FedHEP+ isolated no Min-Max attacker either. | Attackers' median z +5.2 (ALIE) and +2.9 (Min-Max). |
| C5 | Removal-based screening is far better than trust-weighting against many crude attackers. | **Supported** | 5–6 of 15 sign flippers: FedHEP+ 40–44% vs Defended FedHEP 27–32%. |
| C6 | Isolating backdoor attackers does not remove the backdoor; it keeps growing afterwards. | **Observed**, cause **pending** | 20 clients: +17 to +47 points after the last isolation in all 5 runs where every attacker was isolated. Decided by `followup_backdoor_stop` (attackers stop at round 10 or 15). |
| C7 | An influence bound (ARC, NNM, trimmed mean) limits the damage of undetected attackers without false isolations. | **Pending** | `phase6_l0`, `phase6_l0_a02`. |
| C8 | FedHEP-Guard improves accuracy and bottom-10% accuracy over FedHEP+, Defended FedHEP and established rules, under crude, adaptive and backdoor attacks, on held-out settings. | **Pending** | Research plan, Stages 2–4. |
| C9 | Its cost in time and memory is small and measured on two kinds of laptop GPUs. | **Pending** | Stage 1 instrumentation is in place; numbers come with the Phase 6 runs. |

**If C7–C9 fail**, C1–C6 still make a measurement paper (an analysis of why distance-based defenses fail in hierarchical personalized FL, with the negative results). That is publishable at a workshop or as a journal analysis paper, but a main-track paper needs a method that works.

## 3. Closest published work and how the paper differs

| Work | What it does | What this paper adds |
|---|---|---|
| Errami and Bergou, "Tolerating Outliers: Gradient-Based Penalties for Byzantine Robustness and Inclusion", IJCAI 2024 ([PDF](https://www.ijcai.org/proceedings/2024/0435.pdf)) | Shows robust aggregators discard benign outliers under non-IID data; re-includes flagged clients with weights ∝ min(1, 1/‖g_i − g_F‖). Evidence on overall accuracy (MNIST, CIFAR-10). | Measures *which* clients are lost and *when* (per-round false isolation, bottom-10% accuracy), in a hierarchical personalized model; our soft quarantine is close to their re-inclusion idea, so it must be compared against it. |
| Ajibuwa and Odeyomi, "On the Hidden Costs of Fairness in Byzantine-Robust Decentralized FL under Data Heterogeneity", IEEE IPCCC 2025 ([record](https://profiles.ncat.edu/en/publications/on-the-hidden-costs-of-fairness-in-byzantine-robust-decentralized/)) | Six robust methods all reduce group fairness relative to weighted averaging, even without attacks; attributed to treating deviation as malice. | Same phenomenon, in centralized hierarchical personalized FL, with a defense that targets it. C1–C2 confirm their finding, so we must not claim the phenomenon as new. |
| Zhang et al., HadAGG, "Malicious or Benign?", IEEE TNSE vol. 13, 2026 ([record](https://pure.bit.edu.cn/en/publications/malicious-or-benign-enhancing-the-contribution-of-model-updates-i/)) | Cosine similarity plus federated Shapley values sort updates into benign, malicious and anomalous-but-benign; corrects malicious updates by projection. | The closest method. Differences to establish: no server-side utility evaluation (Shapley needs data and many evaluations), hierarchy-aware two-vantage trust, adaptive attacks, cost. Must read the full paper and, if code is available, run it as a baseline. |
| FedCAP, Li et al., arXiv 2410.13083 (venue not stated on the page read) ([HTML](https://arxiv.org/html/2410.13083v1)) | Personalized robust FL: similarity-weighted customized aggregation, permanent removal by an update-norm threshold. Reports 100% false negatives under LIE (ALIE). | Its blind spot is C4. A natural personalized-FL baseline. |
| Allouah et al., NNM (AISTATS 2023) and ARC (ICLR 2025) | Pre-aggregation steps with guarantees under heterogeneity. | Used as layer L0, inside a hierarchical model, with measured interaction with screening. |
| Hierarchical defenses: SHFL (arXiv 2024), PPBR (IEEE TMC 2026), TWRM (FGCS 2024), Cost-TrustFL, R-HFL | Score edges and clients level by level or as a product of trust. | A fused client score with the parent's view discounted by trust in the parent, for similarity-formed groups that can be captured. TWRM's combination rule is paywalled and arXiv 2610.24036 could not be opened from this workspace: both must be read before claiming this point. |
| Backdoor durability: Neurotoxin (ICML 2022), Chameleon (ICML 2023) | Backdoors fade once poisoning stops; Chameleon measures lifespan, about 900 rounds for a pixel pattern on CIFAR-10 with ResNet-18 ([PDF](https://arxiv.org/pdf/2304.12961)). | C6 reports a backdoor that *grows* after its attackers are removed. If `followup_backdoor_stop` confirms it without a defense, this is a notable observation; recovery (H8) follows FedEraser and FedRecover. |

## 4. Outline (main track, about 8 pages plus appendix)

| Section | Content | Figures and tables (existing ones in bold) |
|---|---|---|
| 1 Introduction | The problem; the two failure directions; contributions | Teaser: **`followup_alpha02.png`** (false isolations) beside **`followup_adaptive.png`** (adaptive attackers missed) |
| 2 Background and threat model | FedHEP; attacker capabilities (crude, in-distribution, backdoor, defense-aware); attacker fraction; what the server sees | **`diagrams/1_model.png`**, **`diagrams/2_fedhep_round.png`** |
| 3 Why distance-based screening fails | C1–C5 as a measurement study: false isolation over rounds, scale collapse, adaptive attackers above the consensus, the many-attacker regime | **`why_z_grows.png`**, **`followup_diagnostics.png`**, **`followup_attack_rate.png`**, table of α = 0.2 per-seed differences |
| 4 FedHEP-Guard | L0 influence bound; multi-signal screen with bimodality gate; two-vantage reversible trust; soft quarantine; recovery | **`diagrams/9_proposed_round.png`**, **`diagrams/10_trust_fusion.png`**, formulas from the research plan |
| 5 Experimental setup | Datasets, partitions, clients, participation, rounds, seeds, attacks, baselines, metrics, hardware | Setup table |
| 6 Results | Main table; adaptive attacks including a defense-aware one; backdoor and recovery; ablation ladder; held-out validation; cost | Pending figures: main bar chart, ablation table, cost table |
| 7 Related work | Section 3 of this plan, condensed | – |
| 8 Limitations and ethics | Privacy (individual updates are inspected), single model family, laptop-scale clients | – |
| Appendix | Reproduction of FedHEP (Phase 1 parity), BatchNorm finding, Phase 4 negative result, full per-seed tables | **`phase1_parity.png`**, **`phase4_drilldown.png`**, **`phase5_three_seeds.png`** |

## 5. Minimum experiments before submission

In priority order. Estimates use 12.5 min per 30-round run on either machine.

| # | Experiment | Decides | Runs | About |
|---|---|---|---|---|
| E1 | `followup_backdoor_stop` (Acer) | C6 | 6 | 1.3 h |
| E2 | `phase6_l0`, `phase6_l0_a02` (Mac) | C7 | 54 | 11 h |
| E3 | Ablation ladder R1–R8 on D1 and D2 | C8, design | ~240 | 50 h (25 h on both machines) |
| E4 | Baselines for the closest work: Errami and Bergou's re-inclusion (simple to implement), FedCAP, HadAGG if code exists, Ditto and FedRep under attack | Positioning | ~90 | 19 h |
| E5 | Defense-aware adaptive attack against the final method | Reviewer expectation | ~30 | 6 h |
| E6 | Second and third dataset (CIFAR-10, FEMNIST natural partition), 50 clients with 50% participation, 100 rounds for the main setting | Generality | sized after Stage 1 timings | likely needs cloud GPU time |
| E7 | Held-out validation (seeds 2027–2029, α = 0.1 / 50 clients, FEMNIST) | C8 | ~60 | depends on E6 |

E1 and E2 are already in the queue. E3 starts when E2 shows whether L0 helps.

## 6. Venue and timing

From the literature review (dates must be rechecked on official pages before relying on them): AAMAS 2027 has passed (8 October 2026). IJCAI 2027 is reported, unverified, at about 11 January 2027, which leaves too little time for E3–E7. Realistic targets: a mid-2027 conference deadline (for example NeurIPS 2027, dates not yet announced) or a journal (IEEE TIFS, IEEE TDSC, IEEE TNSE, where HadAGG appeared). A workshop version of the measurement study (C1–C6) could go earlier.

## 7. Threats to validity to state in the paper

- One dataset (CIFAR-100) and one model family so far.
- 30 rounds and full participation; the field expects 100+ rounds and partial participation (E6).
- Attacker fraction fixed at 20% in most settings; q runs vary it.
- The first held-out seed (2026) has been used once and is now a development data point; new held-out settings are reserved.
- Results on the Mac (Apple GPU) and the Acer (Intel GPU) are never mixed within a comparison.

## 8. Sources consulted for this plan (new since the literature review)

- [Errami and Bergou, IJCAI 2024](https://www.ijcai.org/proceedings/2024/0435.pdf)
- [Ajibuwa and Odeyomi, IPCCC 2025](https://profiles.ncat.edu/en/publications/on-the-hidden-costs-of-fairness-in-byzantine-robust-decentralized/)
- [Zhang et al., HadAGG, IEEE TNSE 2026](https://pure.bit.edu.cn/en/publications/malicious-or-benign-enhancing-the-contribution-of-model-updates-i/)
- [FedCAP, arXiv 2410.13083](https://arxiv.org/html/2410.13083v1)
- [Chameleon, ICML 2023](https://arxiv.org/pdf/2304.12961)
- [ARC, arXiv 2405.14432](https://arxiv.org/html/2405.14432v5)
