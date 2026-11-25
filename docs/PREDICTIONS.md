# Predictions before the GPU campaign

2026-11-16, written while the lab GPU server is being set up and before any run on it has finished. Each prediction names the run that tests it, the number we expect, and what would prove it wrong. The point of writing this first is that the GPU results will then confirm or refute something stated in advance rather than explain something after the fact. Nothing here is a result.

## 1. What we know going in

Evidence from our own runs (CIFAR-100, ResNet-9, D1 = α 0.5, 15 clients, 3 attackers; D2 = α 0.2, 20 clients, 4 attackers), all in `EXPERIMENT_LOG.md`:

| Fact | Numbers |
|---|---|
| Honest clients are isolated when data is heterogeneous | D2, no attack: about 9 of 20 per run (FedHEP+); R3 with gate 8: 3 |
| Two-sided screening catches ALIE and Min-Max; one-sided catches none | 9 of 9 and 9 of 9 isolated in round 3 |
| The gap gate removes false isolations but hides the backdoor | D1: honest isolations 13 → 3 → 0 (gate 8 → 12); backdoor success 23% → 51% → 62% |
| The influence bound (L0) neutralizes ALIE without isolating anyone | D1 ALIE: L0 47.56, identical to L0 with no attack |
| Isolating ALIE attackers costs accuracy | R3 under ALIE: 45.7–45.9 on every seed, 1.0–2.3 below L0; the same step appears in Phase 3: one-sided FedHEP+ (isolates none) 47.6, two-sided (isolates all three) 45.5 |
| The backdoor outlives its attackers | stop test: 4–13% success at round 10 → 28–53% at round 30 after poisoning stops, no defense |
| The backdoor lives in the shared head, not the personal ones | per-head success (figure below): root 29–71%, parent 15–45%, local 8–22%, across R2, R3-gate-8 and R3-gate-12 |
| Whole-model signals see the backdoor; the root-head cosine does not | FLAME rejects backdoor attackers in 27–28 of 30 rounds; Phase 3 whole-update screening gave 0.3% success (seed 42) |
| The depth-exit guard works on a toy problem | success 93.6% → 6.4%, accuracy 100 → 99.6, 9% of clean inputs flagged |

![Backdoor success per head](figures/phase6_backdoor_heads.png)

Evidence from the literature that bears on what comes next:

- **Keeping the classifier local blocks backdoors** (Qin et al., KDD 2023, [arXiv 2302.01677](https://arxiv.org/pdf/2302.01677)): the shared feature extractor learns the trigger features, but a classifier trained only on clean local data never maps them to the target label. FedRep held BadNet below 10% and Blended below 20% where FedAvg and Ditto reached 89–98%. Re-initializing and retraining only the classifier on local data ("Simple-Tuning") cut success by 56.6% on average. This is the same gradient as our root → parent → local heads: FedHEP's local head is a FedRep-style classifier.
- **Attacks built for personalized FL defeat that protection** (Bad-PFL, ICLR 2025; DCInject, [arXiv 2602.18489](https://arxiv.org/html/2602.18489), 2026): triggers made from natural features or global frequency statistics survive personalization with 92–100% success on CIFAR-100 at 100 clients, α 0.5, 10% attackers. Our pixel-patch backdoor is the easy case for personal heads; the paper must include one of these.
- **Theory of what each defense family can and cannot do** (Hammer and Anvil, [arXiv 2509.08089](https://arxiv.org/pdf/2509.08089)): with δ the distance between the mean benign and mean malicious update, robust aggregation and detection remove backdoors above some δ₁ (large deviations are visible), post-hoc removal by fine-tuning works below some δ₂ (small deviations are easy to overwrite), and either alone leaves an exploitable window. Norm bounding is broken by one adaptive client at any usable threshold (90% success at m = 1). Our gap gate is exactly a δ threshold, and our per-head finding plus removal (H8) is the second family; the paper's claim should be the combination.
- **The 2026 evaluation bar** (BackFed, [arXiv 2507.04903](https://arxiv.org/html/2507.04903v2)): 100 clients with 10 per round at α 0.5, a 300-round attack window followed by 300 rounds without attack, metrics ASR over the last 30 attack rounds, peak ASR, and *lifespan* (rounds with success above 50% after the attack stops); detector false-positive rates reported (FLAME 39%, DeepSight 26%, FLDetector 5% on FEMNIST). Fixed-pattern backdoors lived 263 rounds after the attack under FedAvg; a higher server learning rate (0.4–0.6) forgot them faster. Dynamic attacks (A3FL, IBA) defeated nearly every defense.
- **Two-stage trust-then-inspect with layer-level statistics** ([arXiv 2605.19253](https://www.alphaxiv.org/abs/2605.19253.md), 2026): per-layer sign counts, kurtosis, cosine to a trusted reference, clustered per layer, with a median/MAD reputation after a warm-up; 9–11% success on CIFAR-10 ResNet-9 at 20 clients with 6 attackers, but no false-positive rate reported. Close to our L1 design; our contribution over it is the false-isolation measurement and the gate.
- **Recovery after detection** (FAUN, [arXiv 2605.02110](https://arxiv.org/abs/2605.02110); SCRUB-FL, [arXiv 2606.22700](https://arxiv.org/abs/2606.22700), both 2026): both keep only a short window of the removed clients' updates or a synthesized trigger distribution, and report near-zero success after a few unlearning rounds. They confirm H8's premise (detection alone leaves the backdoor in the model) and give the storage budget the paper should match: a window of updates, not the full history.

## 2. Predictions for the queued experiments

Confidence is a subjective probability that the prediction holds as stated. "Refuted if" is the number that would make us drop the idea.

### `gpu_d1_signals` (150 runs, first batch on the server)

| # | Prediction | Confidence | Refuted if |
|---|---|---|---|
| P1 | The CUDA baselines (FedHEP, Defended, FedHEP+, L0, R3) land within 1 point of the Mac numbers with no attack, and the ordering between them is the same | 0.8 | any baseline moves by more than 2 points, or L0 stops beating R3 under ALIE |
| P2 | `weights` and `critical` signals both cut backdoor success below 25% (from 51% with the root-head signal), with `critical` the lower of the two | 0.6 | both stay above 40% |
| P3 | Those two signals bring back false isolations: 2–5 honest clients per run with no attack, because whole-model and layer cosines are noisier under heterogeneity than the root-head cosine | 0.6 | zero honest isolations and success below 25% at the same time (then the signal is simply better and the gate is unnecessary) |
| P4 | `sign_align` isolates every sign flipper and no honest client, but does not see the backdoor (success above 40%): the trigger changes few coordinates' signs | 0.65 | success below 25% |
| P5 | `probe_logits` and `head_gap` catch sign flippers and nothing else; neither moves the backdoor below 40%, because probe images never contain the trigger | 0.75 | either reaches below 25% |
| P6 | The depth-exit guard cuts the ensemble's backdoor success at least in half in every backdoor run, at a cost of 0.5–1.5 points of accuracy, flagging 5–10% of clean inputs | 0.55 | success falls by less than a third, or accuracy drops by more than 2 points |
| P7 | The exit guard flags a larger share of clean inputs for the bottom-10% clients than for the rest, so the bottom-10% accuracy falls more than the pooled accuracy | 0.6 | bottom-10% falls no more than pooled |

### The ALIE accuracy puzzle (30 runs, to queue after the first batch)

Isolating the ALIE attackers under R3 lowers accuracy by 1.0–2.3 points on every seed, while L0 alone, which isolates nobody, is unaffected. Three explanations, each with its run:

| Candidate | Test | Prediction |
|---|---|---|
| (a) Isolation triggers regrouping, which reshuffles parents mid-training | R3 under ALIE with `plus.regroup: false` | small effect: at most 0.5 of the gap |
| (b) ALIE updates are honest updates shifted by 0.43σ, so they still carry the attackers' data; isolating them forfeits 20% of the training data | compare R3 under ALIE with R3 under sign flip and Min-Max | **most of the gap**: R3 scores 45.9–46.2 under all three attacks, which is "12 honest clients" accuracy, while L0 under ALIE keeps all 15 |
| (c) After 3 isolations the trimmed mean's f falls to 0, so L0 degenerates to a plain mean over the remaining 12 | R3 under ALIE with `global_f` held at 3 | a measurable part: 0.3–0.8 points, since the trimmed mean over honest clients only acts as a mild regularizer |

Confidence that (b) explains more than half: 0.7. Consequence for the paper, if confirmed: for attacks that keep attackers near the honest mean, **bounding beats isolating**, and the method should isolate only when the bound is insufficient (sign flip) or the harm is targeted (backdoor). That is an argument for soft, reversible trust (H3) rather than permanent isolation, made with numbers.

### Recovery, H8 (60 runs)

| # | Prediction | Confidence | Refuted if |
|---|---|---|---|
| P8 | Rewinding the trunk and the root head to the Global model from before the first suspicious round, then training on without the isolated clients, cuts final backdoor success to below 15% from 51% (R3, gate 8), at a cost under 1 point, when detection happens by round 10 | 0.6 | success stays above 30% |
| P9 | Rewinding the trunk alone, keeping all heads, leaves success above 30%: the root head's mapping from trigger features to the target label is re-learned from the still-poisoned parent models unless they are also rewound | 0.5 | trunk-only rewind reaches below 15% |
| P10 | Replaying stored honest updates (variant b) recovers the accuracy lost by the rewind (within 0.3 points of no rewind) but costs 4 GB of stored updates per 20-client run, so variant (a) is what the paper proposes | 0.6 | variant (a) loses more than 1.5 points |

### Head-level inference (new idea 9 in `IDEAS.md`; about one hour to implement, evaluation-only)

The per-head numbers suggest a zero-cost defense: when the root head's prediction is not supported by the parent and local heads, let the personal heads decide. This is the exit-guard decision rule applied to the heads FedHEP already has, so it needs no extra classifiers, no held-out split and no threshold fitting.

| # | Prediction | Confidence | Refuted if |
|---|---|---|---|
| P11 | Head-agreement inference lowers ensemble backdoor success to within 5 points of the local head's own rate (8–22%) in every backdoor run, at a cost below 0.5 points with no attack, because the local head is already the best head on each client's own data at α 0.5 | 0.65 | success stays within 10 points of the ensemble's current rate, or accuracy drops by more than 1 point |
| P12 | Against a natural-feature trigger (Bad-PFL style), the local head's advantage shrinks to under 10 points and head-agreement inference no longer helps | 0.6 | the local head still resists by 20 points or more |

### Scale and protocol (Stage 3)

| # | Prediction | Confidence | Refuted if |
|---|---|---|---|
| P13 | At 100 rounds the gap gate's cost appears: with the root-head signal, backdoor success keeps rising to above 80% by round 100 in runs where it was 51% at round 30 (our stop test shows it grows without attackers) | 0.7 | it plateaus below 60% |
| P14 | At 50 and 100 clients with 10 per round, false isolations per round fall (more clients, fewer per-round outliers) but the fraction of honest clients ever isolated over 100 rounds rises above the 30-round figures, so "isolated at least once" must be reported, not the per-round count | 0.6 | the ever-isolated fraction is lower at 100 clients |
| P15 | At 40% attackers, L0 fails first (NNM mixes attackers into honest neighbours once attackers are 8 of 20) and the two-sided screen with the gate still isolates the sign flippers, so the full method degrades gracefully to about the FedHEP+ numbers | 0.5 | the full method falls below Defended FedHEP at 40% |
| P16 | A defense-aware adaptive attack that keeps every signal inside the gate's band (our planned attack) achieves at most half the backdoor success of the plain attack, because staying inside the band bounds its per-round deviation and L0 bounds its influence; this is the δ window of Hammer and Anvil and is where recovery (H8) must do the work | 0.5 | the adaptive attack reaches the plain attack's success |


### Large GPU grid (2026-11-18, `gpu_grid_n{15,30,50}`)

| # | Prediction | Confidence | Refuted if |
|---|---|---|---|
| P17 | The union screen (R3 with `weights_critical`: a client's z is the lower of its whole-update and critical-layer z) keeps backdoor success below 5% on all three seeds at 15 clients, α 0.5, 3 attackers, where each signal alone missed one seed; its honest isolations with no attack stay near the whole-update signal's (13 over 3 seeds), because the critical-layer screen isolated nobody on its own | 0.6 | any seed above 10%, or more than 20 honest isolations over 3 no-attack runs |
| P18 | At α 0.1 every screen isolates more honest clients with no attack than at α 0.5 (strong label skew looks like an attack), and R3's gap gate keeps that count at no more than half of FedHEP+'s | 0.6 | R3 isolates as many honest clients as FedHEP+ at α 0.1 |
| P19 | At 50 clients with 15 sign flippers, L0 and every R3 variant stay within 3 points of their own no-attack accuracy, while Defended FedHEP loses more than 10 | 0.55 | an R3 variant loses more than 5 points, or Defended FedHEP loses less than 5 |


### Mac and Acer, while the grid runs (2026-11-18, `mac_c7_stealth`, `acer_c8_fmis`)

| # | Prediction | Confidence | Refuted if |
|---|---|---|---|
| P20 | The stealth backdoor (blended toward the honest mean until it sits inside Min-Max's bound) gets past the whole-update and union screens more often than the plain backdoor does, but carries less backdoor: with 3 attackers its success under no defense is lower than the plain backdoor's | 0.55 | the screens catch it as often as the plain backdoor, or its undefended success matches the plain one |
| P21 | Telling the robust rules f = 6 when 3 attack costs L0 at least 1 point with no attack (it trims more honest updates), while R3 with the union screen loses less, because its trimmed mean only acts on the clients the screen kept; f = 1 lets sign flip through L0's trimmed mean but not R3's screen | 0.5 | R3 loses as much as L0 at f = 6, or L0 at f = 1 holds sign flip within 1 point |
| P22 | (Phase 7, paired, Acer.) Scored on the same updates, the peer reference separates the 3 backdoor attackers from honest clients better than the global median at α 0.1 (mean per-round AUROC on \|z\|, whole update, at least 0.10 higher), and about as well at α 0.5 (within 0.05) | 0.4 | at α 0.1 the peer AUROC is not higher than the median's |
| P23 | (Paired.) With no attack, the peer reference halves the share of honest client-rounds with \|z\| > 2.5 on the whole update at α 0.1; at α 0.5, where most clients hold most classes, it changes it by less than a third | 0.5 | the share falls by less than half at α 0.1 |
| P24 | (Acting.) The union screen acting on the peer reference isolates at most half as many honest clients per run with no attack as the same screen on the median (both α), and still keeps backdoor success under 5% on every seed at α 0.5 | 0.35 | more than half as many honest isolations, or any α 0.5 seed above 5% |
| P25 | (Acting.) At α 0.1 the peer reference lowers backdoor success by at least 10 points on average against the median reference (same seeds, Acer) | 0.3 | a smaller drop, or a rise |
| P26 | (Phase 8, Acer, α 0.5, no attack.) With the shared signal (whole update without the personal heads), no run isolates 3 or more honest members of one parent group in the same round, and honest isolations per run fall to at most half the union screen's on the Acer (3.8 per run over 6 runs) | 0.55 | a whole-group isolation in any no-attack run, or more than 1.9 honest isolations per run |
| P27 | (Phase 8.) The shared union screen keeps backdoor success under 1% on every seed at α 0.5 (30 rounds), as the union screen does | 0.55 | any α 0.5 seed at 1% or more |
| P28 | (Phase 8.) The shared union screen still isolates every sign-flip, ALIE and Min-Max attacker at α 0.5 | 0.8 | any of the 27 attackers left in |
| P29 | (Phase 8, 100 rounds.) FedHEP-Guard with the shared signal holds the plain and the stealth backdoor under 1% at round 100 on every seed (with the whole-update signal: stealth seed 7 reached 33.8%) | 0.4 | any seed at 1% or more at round 100 |
| P30 | (Phase 8 at scale, Acer, `acer_c13_shared_scale`.) At 30 clients, α 0.5, no attack, the shared union screen isolates at most half as many honest clients per run as the union screen in the same sweep, with no whole-group event (3 or more honest members of one parent in one round) | 0.5 | more than half as many, or any whole-group event |
| P31 | (Phase 8 at scale.) At 50 clients, α 0.1, no attack, the shared signal removes less than 60% of the union screen's honest isolations, because there about half are single clients flagged when the gap gate opens by chance (cause 1, report 8.2) | 0.6 | the shared union screen isolates fewer than 40% as many honest clients as the union screen |
| P32 | (Phase 8 at scale.) At 30 clients, α 0.5, with 6 backdoor attackers, the shared union screen keeps backdoor success under 5% on every seed (the union screen on the GPU: 1.2 / 11 / 3.5% with 9 attackers, 0.9% with 6) | 0.5 | any seed at 5% or more |

## 3. The model we expect to publish

If the predictions above mostly hold, the published method is:

1. **L0, influence bound**, always on: ARC clipping → NNM → trimmed mean. Carries ALIE and Min-Max on its own (P1).
2. **Two-sided robust screen with a gap gate**, on a layer-level signal (P2), with the gate set so that nobody is blamed unless the round's scores split (gate 12 on D1; the D2 value comes from the ladder).
3. **Isolation that is soft and reversible** rather than permanent, justified by the ALIE puzzle (bounding beats isolating when attackers sit near the mean) and by H3.
4. **Head-agreement inference** (P11): the personal heads overrule the shared head when they disagree with it. New, hierarchy-native, zero training cost.
5. **Rewind on detection** (P8): the trunk and the root head return to the last Global model before the first suspicious round. Storage: one Global model per round (6.9 MB), a window of 10 rounds.

Expected D1 numbers for this model, against the strongest baselines on the same device (mean of 3 seeds; the table is the claim the paper has to defend):

| | No attack | Sign flip | ALIE | Min-Max | Backdoor success | Honest clients isolated |
|---|---|---|---|---|---|---|
| Defended FedHEP (published) | 48.4 | 38.1 | 47.5 | 45.2 | 79% | 0 |
| FedHEP+ (Phase 3, one-sided) | 47.9 | 46.3 | 47.6 | 44.9 | 21% | 13 (no attack) |
| FedHEP+ two-sided | — | — | 45.5 | 45.7 | — | — |
| FLAME | 42.8 | 42.7 | — | — | 0.5% | many |
| **Expected FedHEP-Guard** | **47.3 ± 0.3** | **46.2 ± 0.5** | **47.0 ± 0.5** | **46.0 ± 0.5** | **below 10%** | **0–1** |

The honest claim, if this holds: within 1 point of the undefended model with no attack, no honest client isolated, every tested untargeted attack neutralized, and the backdoor held below 10% at prediction time and removed after detection, at the cost of one extra forward pass per prediction and one stored model per round. What it does not claim: resistance to natural-feature triggers (P12) and to attackers above 40%.

Things that would change the model:
- If P2 fails (layer-level signals do not see the backdoor), the backdoor defense rests entirely on heads and rewind, and the screen stays on the root-head signal.
- If P11 fails, the depth-exit guard (P6) is the fallback at prediction time, at a higher cost.
- If P8 fails, the paper's backdoor claim is prediction-time only, and the lifespan metric (BackFed) becomes the headline negative result.

## 4. The campaign: what "thousands of runs" buys

Throughput is unknown until the pilot finishes. For planning, one 30-round D1 run is about 13 minutes on the Mac; a single RTX PRO 4500 should be at least three times faster, and three runs per GPU across four GPUs give roughly 10–20 runs per minute of wall time at the queue, so 100–200 runs per hour. The pilot replaces these guesses. The campaign in order, with the decision each stage feeds:

| Stage | Runs | Hours at 150 per hour | Decides |
|---|---|---|---|
| C1 `gpu_d1_signals` | 150 | 1 | P1–P7: the signal and whether the exit guard is worth its cost |
| C2 ALIE puzzle | 30 | 0.2 | (a), (b), (c): isolate or bound |
| C3 head-agreement inference | 0 new (re-evaluate C1's saved models) or 45 | 0.3 | P11 |
| C4 D2 ladder on CUDA (L0, R2, R3 gate 8 and 12, best signal) | 75 | 0.5 | the gate value at α 0.2 |
| C5 recovery H8 (variants a, b, trunk-only; 3 attack timings) | 60 | 0.4 | P8–P10 |
| C6 100-round durability: best two rungs and three baselines, backdoor on, then off at round 50 | 45 | 1 | P13; lifespan metric |
| C7 scale: 20, 50, 100 clients × 10%, 20%, 30%, 40% attackers × 5 attacks × 5 methods × 3 seeds | 900 | 6–12 (longer runs at 100 clients) | P14, P15; the paper's main table |
| C8 adaptive attacks: defense-aware, layer-wise poisoning, natural-feature trigger, on-off attacker | 120 | 1 | P12, P16 |
| C9 baselines at the 2026 bar: Krum, RFA, centred clipping, NNM + ARC alone, FLDetector, AlignIns, Ditto and FedRep under attack | 480 | 3 | the comparison table |
| C10 efficiency: sketching k ∈ {512, 2048, 8192}, timing and memory for every method | 90 | 0.6 | H6; the cost table |
| C11 five seeds for every headline table (seeds 2027–2029 added) | about 600 | 4 | confidence intervals |
| C12 held-out: α 0.1 with 50 clients; FEMNIST with 100 clients; frozen method | 150 | 2–4 | Stage 4 |
| Total | about 2,700 | 20–30 hours of queue time | |

So the whole programme is one to two days of the four GPUs, or a week on two. The binding constraint is not compute but the order of decisions: C1–C3 fix the method, C4–C6 fix its parameters, and only then is C7–C12 worth running, because every run in C7–C12 would have to be repeated if the method changed. Nothing in C7–C12 is queued until C1–C6 are read.

Runs that are cheap on the server and were out of reach on the laptops: 100 rounds (C6), 100 clients (C7), five seeds everywhere (C11). These are the three things a 2026 reviewer asks for first.

## 5. Risks to the predictions

- **The Mac and CUDA disagree by more than noise (P1 fails).** Then every published number is CUDA-only and the Mac runs become development evidence. Already allowed for by the device rule.
- **Several runs per GPU change results.** The pilot checks this by repeating seeds across 1, 2 and 4 runs per GPU; cuDNN non-determinism can make results differ at the fourth decimal even on one process, which is harmless, but a systematic shift would be a bug.
- **The exit guard's 20% held-out split starves small clients at 100 clients.** Then the guard needs the exit classifiers fitted on fewer images, and its cost rises for exactly the clients that matter.
- **Natural-feature triggers (P12) may defeat the heads and the rewind together**, because the trigger is in the honest data too. That would be an honest limitation, not a failure of the paper.

## Sources

- Qin et al., Revisiting Personalized Federated Learning: Robustness Against Backdoor Attacks, KDD 2023 — https://arxiv.org/pdf/2302.01677
- DCInject (2026) — https://arxiv.org/html/2602.18489
- Bad-PFL, ICLR 2025 — https://proceedings.iclr.cc/paper_files/paper/2025/file/6225ca7eaef477a05d01e90b0615ab4f-Paper-Conference.pdf
- Hammer and Anvil: Toward a Theory of Backdoors in Federated Learning — https://arxiv.org/pdf/2509.08089
- BackFed benchmark — https://arxiv.org/html/2507.04903v2
- Two-stage robust aggregation for OTA-FL (2026) — https://www.alphaxiv.org/abs/2605.19253.md
- FAUN, adversarial update-based federated unlearning (2026) — https://arxiv.org/abs/2605.02110
- SCRUB-FL (2026) — https://arxiv.org/abs/2606.22700
- H+ similarity-aware aggregation — https://arxiv.org/html/2509.24330v1
- FedSurrogate, AlignIns, Orion, CrowdGuard, FedREDefense, BackdoorIndicator: see `IDEAS.md` and `LITERATURE_REVIEW.md`

## Outcomes so far (appended; the predictions above are unchanged)

| Prediction | Outcome | Run |
|---|---|---|
| ALIE (a): regrouping explains at most 0.5 of the gap | **Held:** 0.35 of 1.8 points | `m6__r3_noregroup`, 2026-11-17 |
| ALIE (c): fixed f recovers 0.3–0.8 points | **Refuted, in direction:** fixed f costs another 1.1 points | `m6__r3_ffixed` |
| ALIE (b): lost data explains most of the gap | **Consistent** (about 1.4 points left once (a) and (c) are removed), not yet tested directly | — |
| P6 cost: exit guard costs 0.5–1.5 points | **Better than predicted:** 0.33 points, 3.1% of clean inputs flagged (no-backdoor runs only) | `m6__*`, 19 runs |
| P11 cost: head agreement costs under 0.5 points | **Held:** 0.18 points, 3.0% of inputs overruled (no-backdoor runs only) | `m6__*`, 19 runs |
| P6 effect: exit guard at least halves backdoor success | **About held:** 72% → 39% and 67% → 39% over 100 rounds; 74% → 35% on an undetected D1 run (cuts of 42–52%) | `x6__fedhep`, `x6__l0`, `m6__*` |
| P8: rewind to before the first suspicious round brings success below 15% | **Not met:** 16–30% when the attacker is caught (from 36–45%); unchanged when nobody is caught. The rewind cannot reach poisoning done before the first suspicious round | `m6__r3_*__backdoor3` |
| P9: trunk-only rewind leaves success above 30% | **Mixed:** 22.5% and 30.0%; rewinding the root head too adds only 1–4 points | `m6__r3_trunk`, `m6__r3_tr` |
| P11 effect: head agreement brings success near the local head's | **Refuted:** 3–9 points of reduction; the parent head carries the backdoor too, so the personal heads rarely agree against the ensemble | `m6__*`, `x6__*` |
| P13 (root-head gated screen rises above 80% by round 100) | Pending (`x6__r3g8` running). Related: with no detection the backdoor stays at 64–75% for 50 rounds after the attack stops | `x6__fedhep`, `x6__l0` |
| P1: CUDA baselines within 1 point of the Mac with no attack, same ordering | **Mostly held:** 4 of 5 baselines within 0.7 (FedHEP +1.4); orderings swap only within half a point; L0 still beats R3 under ALIE (47.29 vs 46.33). 105 identical runs: mean difference 0.72 | `g1__*`, `docs/results/device_pairs.json` |
| P2: whole-update and critical-layer signals cut backdoor success below 25%, critical the lower | **Half held:** whole update 24% on average (0.6, 71, 0.6% by seed; all three seeds under 1% on the Mac); critical layers 50% (catches one seed). Critical is not the lower | `g1__r3w`, `g1__r3c`, `m6__r3w` |
| P3: those signals bring back 2–5 honest isolations per run | **Held for whole update** (4.3 per run); critical layers isolates nobody | `g1__r3w__clean`, `g1__r3c__clean` |
| P4: sign alignment isolates every sign flipper and no honest client, misses the backdoor | **Held** exactly: 3/3, 0 honest, backdoor 76% | `g1__r3s` |
| P5: probe logits and head gap catch sign flippers and nothing else | **Refuted both ways:** neither catches the sign flippers on CUDA (both catch Min-Max); head gap catches the backdoor on two seeds (10%, 0.7%) | `g1__r3p`, `g1__r3h` |
| P6 (CUDA): exit guard halves backdoor success | **Held:** 79 → 42%, 64 → 35%, 57 → 32% | `g1__*__backdoor3` |
| P13: root-head gated screen above 80% by round 100 | **Refuted:** 23, 71, 21% (the attackers are caught by round 14 on two seeds) | `x6__r3g8` |
| P8 at 100 rounds (Acer) | **Not met with the root head, met with the whole update:** trunk-and-root rewind takes the root-head screen from 23 / 71 / 21% to 8 / 28 / 15% (cost 1.1 points), and the whole-update screen from 1 / 17 / 1% to 0 / 1 / 0% (cost 0.4 points) | `x6__r3_tr`, `x6__r3w_tr` |
| P17: the union screen keeps backdoor success below 5% on all three seeds (15 clients, α 0.5, 3 attackers), honest isolations near the whole-update signal's 13 | **Held exactly:** 0.6 / 0.5 / 0.6% where the whole-update screen alone gave 0.6 / 71.4 / 0.6%; 13 honest isolations over 3 runs | `gg15_a05__*__r3wc` |
| P18: at α 0.1 every screen isolates more honest clients than at α 0.5; R3 at most half of FedHEP+ | **Half refuted:** FedHEP+ does (2.7 → 7.7 per run at 15 clients), but R3 stays at 1.3 and the union screen falls (4.3 → 1.7); R3 is far below half of FedHEP+ (1.3 against 7.7) | `gg15_*__clean__*` |
| P20: stealth backdoor passes the whole-update and union screens more often, but carries less backdoor | **Refuted in its premise, half held in effect:** undefended it reaches 71–83% (the plain one 74–84%); it defeats FedHEP+ (56–75% from 5–29%) and the whole-update screen on one seed (61%), but not the union screen (0.2–0.5%) | `m7__stealth3__*` |
| P21: f = 6 costs L0 at least 1 point with no attack, R3 less; f = 1 lets sign flip through L0 | **Half refuted:** f = 6 costs L0 nothing (47.56 against 47.43), while R3 loses 1.2 points; f = 1 costs L0 5 points under sign flip, as predicted | `x8_*` |
| P22: paired, the peer reference separates backdoor attackers better at α 0.1 (AUROC +0.10), about as well at α 0.5 | **Refuted at α 0.1, held at α 0.5:** whole-update AUROC 0.65 against 0.69 for the median (shadow arm), 0.60 against 0.66 (peer arm); at α 0.5 0.97–1.00 against 0.98 | `x10_*__backdoor3__*` |
| P23: paired, no attack, the peer reference halves honest client-rounds with \|z\| > 2.5 at α 0.1, less than a third change at α 0.5 | **Mixed:** at α 0.1 0.035 → 0.005 in the arm acting on the median (held), but 0.080 → 0.067 in the arm acting on the peer reference (refuted); at α 0.5 −24% and −37% | `x10_*__clean__*` |
| P24: acting on the peer reference halves honest isolations and keeps α 0.5 backdoor under 5% | **Refuted:** honest isolations rose (α 0.5: 4.0 → 7.0 per run; α 0.1: 0.3 → 0.7); α 0.5 backdoor 0.4 / 0.3 / **43.8%** | `x10_*__r3wc_peer` |
| P25: acting on the peer reference lowers α 0.1 backdoor by 10 points | **Refuted:** it rose, 11.3% → 26.9% | `x10_a01__backdoor3__*` |
| P2 at D2 (CUDA): whole update and critical layers below 25% | **Refuted:** whole update 47 / 40 / 4.7% (30%), critical layers 45%; at D2 no rung tested gets below 35% on average | `g2__r3w`, `g2__r3c` |
| P3 at D2 (CUDA): those signals bring back 2–5 honest isolations per run | **About held:** 1.7 (whole update) and 2.3 (critical layers), against FedHEP+'s 8.7 | `g2__*__clean` |
| P8 on CUDA: rewind brings success below 15% | **Not met:** root head 57% → 42–45%; whole update unchanged (catches by round 4 or never) | `c5__*__backdoor3`, `g1__r3*__backdoor3` |
| P9 on CUDA: trunk-only rewind leaves success above 30% | **Held:** 45% (trunk) against 42% (trunk and root head) | `c5__r3_trunk`, `c5__r3_tr` |
| P13 on CUDA: root-head gated screen above 80% by round 100 | **Refuted:** 7 / 71 / 49% | `c6__r3g8` |
| P18 at 30 and 50 clients: R3 isolates at most half of FedHEP+'s honest clients at α 0.1 | **Refuted:** R3 5.0 against 2.3 (30 clients) and 6.0 against 1.0 (50 clients); FedHEP+ isolates fewer at α 0.1 than at α 0.5 once there are 30 clients | `gg30_a01__clean`, `gg50_a01__clean` |
| P26 (Phase 8): no whole-group event and at most 1.9 honest isolations per run with the shared signal, α 0.5, no attack | **Refuted:** 2.3 per run (1 / 3 / 3), and one whole-group event of 3 on seed 42. Across all five α 0.5 conditions honest isolations fell by three quarters (0.9 against 3.4 per run) | `x11_a05__clean__r3sc` |
| P27 (Phase 8): backdoor under 1% on every α 0.5 seed | **Refuted:** 0.5 / 0.3 / 78%; on seed 7 the attackers were caught in round 25 | `x11_a05__backdoor3__r3sc` |
| P28 (Phase 8): every sign-flip, ALIE and Min-Max attacker isolated at α 0.5 | **Held:** 27 of 27, all in round 3 | `x11_a05__{sign3,alie3,minmax3}__r3sc` |
| P29 (Phase 8): FedHEP-Guard with the shared signal under 1% at round 100 on every seed, both backdoors | **Refuted:** plain 0.4 / 0.5 / 80%, stealth 1.1 / 0.5 / 62% (seed 7 again) | `x12__*__r3sc_tr` |
| P30 (Phase 8 at scale): at 30 clients, α 0.5, no attack, the shared union screen at most half the union screen's honest isolations, no whole-group event | **Refuted:** 7.3 against 6.0 per run, and one whole-group event of 4 (seed 7) | `x13_n30a05__clean__*` |
| P32 (Phase 8 at scale): at 30 clients with 6 backdoor attackers, the shared union screen under 5% on every seed | **Refuted:** 0.9 / 63 / 73% (the union screen: 0.6 / 1.0 / 1.7%) | `x13_n30a05__backdoor6__*` |
| P31 (Phase 8 at scale, 50 clients): the shared signal removes less than 60% of the union screen's honest isolations | **Held:** 15.0 against 22.3 per run (a third removed); whole-group isolations 13 against 27, single ones 32 against 40 (3 runs each) | `x13_n50a01__clean__*` |
