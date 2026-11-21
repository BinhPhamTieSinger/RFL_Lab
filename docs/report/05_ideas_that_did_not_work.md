# 5. Ideas that did not work (Phases 4, 5 and parts of 6)

Each idea here was a hypothesis with a pass criterion set before the runs. None met its criterion. They are reported in full because each one explains part of FedHEP-Guard's design ([chapter 6](06_fedhep_guard.md)).

| Phase | Idea | Why we tried it | What it changed | Why it failed |
|---|---|---|---|---|
| 4 | Drill-down v1 | FedHEP+ still let 21% of the backdoor through; attackers pushing one direction should stand out more **together** than alone (in a synthetic test, 3 backdoor attackers each looked normal, z > −0.4, while their group scored −2.5 to −4.3) | flagged groups' members face a stricter cut | one-round exclusion hid the backdoor attackers: backdoor 21.5% → 39.7% |
| 4b | Drill-down v2 | v1 failed because a one-round exclusion hides the attacker; evidence that builds over rounds does not | flagged groups' members gain evidence if they share the group's direction | members of a FedHEP group are genuinely alike: honest groups isolated together (−1.1 to −1.7 points) |
| 5 | Classic robust rules at FedHEP's Global step | before inventing a defense, check whether a published one already fixes FedHEP+'s weak spots | Median, Trimmed mean, Bulyan, FLAME, FLTrust | none beats FedHEP+ overall; Bulyan and FLAME stop the backdoor but cost 3.9–5.6 points with no attack |
| 6 | Worker momentum under L0 | the robust-learning literature shows that averaging each client's updates over rounds shrinks the noise attackers hide in | each client's update averaged over rounds before aggregation | worse in every setting |
| 6 | Output-space and sign signals | a backdoor might show in what the model predicts (probe images) or in the sign pattern of the update (AlignIns), where a cosine might miss it | probe logits, head gap, sign alignment | miss the backdoor or the sign flippers |
| 6 | Head-agreement inference | personal heads are trained only on the client's own clean data, so they might not carry the backdoor | personal heads overrule the ensemble when they agree against it | removes only 1–13 points of backdoor success on average: the parent head carries it too |
| 6 | Holding f fixed | R3 lost 1.8 points under ALIE; lowering $f$ after exclusions might let ALIE back into the trimmed mean | L0 keeps assuming $f$ attackers after the screen removed them | −1.1 points under ALIE: it made things worse |
| 6 | Rewinding every shared weight | the personal heads might also hold part of the backdoor | rewind personal heads too | costs 0.6 points against rewinding the trunk and root head, for no gain |

## 5.1 Phase 4: check groups first, then their members

**Hypothesis.** Attackers pushing one direction (one backdoor) stand out more **together** than one by one. So first find the group (parent) that contains them, then look inside it.

![Drill-down, added to FedHEP+](figures/diagrams/06_drill_down.png)

**Parent check (both versions).** The cosine of each parent's mean update to the reference is compared with 64 random groups of the same size, a permutation test:

$$z_g = \frac{\cos(\bar v_g, \tilde r) - \operatorname{med}(\text{null})}{\max\bigl(1.4826\,\mathrm{MAD}(\text{null}),\ 0.01\lvert\operatorname{med}(\text{null})\rvert\bigr)}, \qquad z_g < -2 \Rightarrow \text{parent flagged}$$

**v1 (`plus.drill_down: true`, `drill_mode: exclude`).** Members of a flagged parent face a stricter cut that round: $z_i < -1$ instead of $-2.5$.

**v2 (`drill_mode: evidence`).** v2 asks which members of a flagged parent share its unusual direction. It runs only for parents flagged by the parent check, and works in three steps.

**Step 1: each member's residual.** What is left of member $i$'s signal once the shared part, the median reference, is taken out:

$$u_i = v_i - \tilde r$$

The residual is client $i$'s own deviation from what everyone does. Honest clients' residuals are mostly their own data and noise, so they point in unrelated directions. Attackers pushing the same backdoor share one extra direction, so their residuals point the same way.

**Step 2: alignment with the rest of the group.**

$$d_{-i} = \frac{1}{\lvert g\rvert - 1}\sum_{j \in g,\ j\neq i} u_j, \qquad a_i = \cos\bigl(u_i,\ d_{-i}\bigr)$$

- $d_{-i}$ is the mean residual of the **other** members of $g$. Leaving $i$ out ($j \neq i$) matters: if $u_i$ were in the mean, $a_i$ would be high partly because $i$ is compared with itself.
- $a_i$ is high when $i$ deviates in the same direction as its group-mates. Several attackers in one parent coordinating on one backdoor all get a high $a_i$. An honest member next to them does not, because its residual has nothing in common with theirs.
- Averaging over the other members builds up what they share and cancels what is individual, so the shared attack direction stands out in $d_{-i}$.
- The assumption is that honest members share nothing beyond the reference. In FedHEP it is false: members of one parent train from the same parent head, so honest residuals share a group direction too. This is why v2 failed (below).

**Step 3: is $a_i$ unusually high? A null from outside the parent.** Any residual is somewhat aligned with a group mean by chance, so $a_i$ needs a yardstick. It is the same cosine for clients that are **not** in $g$, against the same leave-one-out directions:

$$N_g = \bigl\{\cos(u_k,\ d_{-i}) : k \notin g,\ i \in g\bigr\}, \qquad z^{\text{align}}_i = \frac{a_i - \operatorname{med}(N_g)}{\max\bigl(1.4826\,\mathrm{MAD}(N_g),\ 0.1\,\lvert\operatorname{med}(N_g)\rvert\bigr)}$$

$N_g$ pools every outside client against every member's direction $d_{-i}$, which gives $(n - \lvert g\rvert) \times \lvert g\rvert$ values, enough for a stable median. With fewer than 5 values every $z^{\text{align}}_i$ is 0. A member is suspicious when $z^{\text{align}}_i$ is large and **positive**: more aligned with its group-mates than outsiders are.

**How it feeds the decision.** High alignment is turned into bad evidence. For members of a flagged parent the evidence update uses the worse of the two signals; nobody is left out for a round on alignment alone:

$$E_i \leftarrow 0.7\,E_i + 0.3\,\min\bigl(z_i,\ -z^{\text{align}}_i\bigr)$$

**Legend:**

| Symbol | Meaning |
|---|---|
| $\bar v_g$ | parent $g$'s mean signal (parent check) |
| null | the parent check's cosine for 64 random groups of the same size |
| $v_i$, $\tilde r$ | member $i$'s signal and the median reference (chapter 4.1) |
| $u_i$ | member $i$'s residual: its signal minus the reference |
| $g$, $\lvert g\rvert$ | the flagged parent and its number of eligible (not isolated) members, at least 3 (`plus.min_group`) |
| $d_{-i}$ | the mean residual of $g$'s other members |
| $a_i$ | member $i$'s alignment with the rest of its group |
| $N_g$ | the null: outside clients' cosines to the members' leave-one-out directions |
| $z^{\text{align}}_i$ | robust z of $a_i$ against $N_g$; same scale floor as the screen (0.1 of the median) |
| $z_i$, $E_i$ | the screen's z and evidence (chapter 4.1) |

**Results** (D1, Mac, 3 seeds; "concentrated" = all 3 attackers in one parent):

| Setting | FedHEP+ alone | v1 | v2 |
|---|---|---|---|
| No attack: accuracy | 48.07 | 47.95 | 46.95 |
| No attack: honest isolated (3 seeds) | 10 | 8 | 19 |
| Concentrated backdoor: success | 21.5% | **39.7%** | **1.9%** |
| Concentrated backdoor: accuracy | 46.42 | 45.92 | 44.69 |

![Drill-down v1](../figures/phase4_drilldown.png)

![Drill-down v2](../figures/phase4b_drilldown.png)

**Why it failed:**
- **The parent check works.** It flagged the attackers' parent with precision 0.85.
- **v1's one-round exclusion made attackers harder to catch.** With the backdoor kept out of the model, honest updates no longer pushed against it, so the attackers' z moved toward normal and isolation came later (seed 42: never).
- **v2 catches the attackers in rounds 2–3, but it also isolates honest groups.** Members of a FedHEP parent train from the same parent head, so their updates share a parent-specific direction. Honest members' alignment z (median 1.0, and up to 2.8 in falsely flagged parents) overlaps the attackers' (3.0–3.6). No cut separates them.

**What carried forward:** removing a client for one round can hide it; evidence that builds over rounds does not. Group-level coordination is real, but in FedHEP honest groups are coordinated too.

## 5.2 Phase 5: established robust rules inside FedHEP

**Question:** would a published defense, plugged into FedHEP's Global step, beat FedHEP+?

![Where the rules plug in](figures/diagrams/07_phase5_rules.png)

| Rule | Global step | Reference |
|---|---|---|
| Median | coordinate-wise median of the updates | Yin et al., 2018 |
| Trimmed mean | per coordinate, drop the $f$ largest and $f$ smallest values, average the rest: $\mathrm{TM}_f(X)_j = \frac{1}{n-2f}\sum_{i=f+1}^{n-f} X_{(i),j}$ | Yin et al., 2018 |
| Bulyan | pick $n - 2f$ updates by repeated Krum, then a trimmed mean around the median per coordinate | El Mhamdi et al., 2018 |
| FLAME | HDBSCAN on cosine distances keeps the majority cluster; clip to the median norm; add Gaussian noise $\lambda S$ with $\lambda = 0.001$ | Nguyen et al., 2022 |
| FLTrust | trust $= \mathrm{ReLU}(\cos(\Delta_i, \Delta_0))$, with $\Delta_0$ the server's update on 100 clean root images; updates rescaled to $\lVert\Delta_0\rVert$ | Cao et al., 2021 |

Parents keep SCCF over the clients the rule accepts. $f = 3$ at D1.

**Results** (D1, Mac, 3 seeds):

| Model | No attack | Sign flip | Label flip | Noise | Backdoor success | Honest left out per round |
|---|---|---|---|---|---|---|
| Defended FedHEP | **48.39** | 38.08 | 43.23 | 45.80 | 78.7% | 0 |
| **FedHEP+** | 47.92 | **46.34** | **46.34** | 45.41 | 20.8% | 1.1–1.9 |
| Median | 47.57 | 39.97 | 44.59 | **46.59** | 76.5% | 0 |
| Trimmed mean | 47.91 | 39.03 | 44.44 | 46.01 | 77.0% | 0 |
| Bulyan | 44.52 | 44.13 | 43.94 | 44.64 | **0.5%** | 3.0–6.0 |
| FLAME | 42.81 | 42.72 | 41.61 | 42.97 | **0.5%** | 2.6–5.1 |
| FLTrust | 47.33 | 45.66 | 42.85 | 44.34 | 64.2% | 2.6–3.6 |

![Phase 5: every rule, three seeds](../figures/phase5_three_seeds.png)

![Phase 5: harder settings](../figures/phase5_harder.png)

**Harder settings:**
- With **5 sign flippers**, FedHEP+ gains 12.6 points over Defended FedHEP.
- With **5 backdoor attackers**, FedHEP+'s backdoor success rises to 56%: five colluders pull the median toward themselves. FLAME holds at 0.5%.
- At **α = 0.05**, FedHEP+'s gain disappears (62.75 against 62.81), and FLAME loses 13 points with no attack.

**Why none was adopted:**
- Median and trimmed mean never remove anyone, so sign flippers and backdoors get through.
- Bulyan and FLAME stop the backdoor by leaving out 3–6 honest clients **every round**, which costs 4–6 points with no attack.
- FLTrust's 100 root images are a thin reference for 100 classes.

**What carried forward:**
- FedHEP+'s weak spot (the backdoor) is FLAME's strength, and FLAME's weak spot (honest clients left out) is FedHEP+'s strength. This motivated a layered design: an influence bound underneath a screen.
- The trimmed mean returns in Phase 6 as the last stage of L0.

## 5.3 Phase 6 components that were tested and dropped

Seven components were tested while building FedHEP-Guard and left out. Each subsection below gives the formula, where it acts (diagram), why we tried it, the results and why it was dropped.

| Component | Config | Result in one line | Source runs | Section |
|---|---|---|---|---|
| Worker momentum under L0 | `robust.momentum: 0.5` | 0.9–1.7 points worse than L0 alone in every D1 condition | `p6__l0m` | 5.3a |
| Probe-logit and head-gap signals | `plus.signal: probe_logits`, `head_gap` | miss the sign flippers; head gap catches the backdoor on 2 of 3 seeds but isolates 9 honest clients | `g1__r3p`, `g1__r3h` | 5.3b |
| Sign-alignment signal (AlignIns MPSA) | `plus.signal: sign_align` | every sign flipper and no honest client, but misses the backdoor (71–83%) | `g1__r3s` | 5.3c |
| Critical layers alone | `plus.signal: critical` | catches the backdoor on 1 of 3 seeds; kept as half of the union screen | `g1__r3c` | 5.3d |
| Head-agreement inference | `fedhep.head_agreement` | costs 0.18 points, removes 1–13 points of backdoor success: the parent head carries the backdoor too | `g1__*`, `m6__*` | 5.3e |
| Holding $f$ fixed after exclusions | `plus.global_f_fixed: true` | −1.1 points under ALIE on the Mac, −1.2 on CUDA | `m6__r3_ffixed`, `c2__r3_ffixed` | 5.3f |
| Rewinding every shared weight | `plus.rewind: all` | 0.6 points below rewinding only the trunk and root head, no backdoor gain | `m6__r3w_all`, `m6__r3_all` | 5.3g |
| Readmission, cap, full-population yardstick | see [chapter 4.4](04_fedhep_plus.md) | all worse | `p3b__*`, `p3c__*` | 4.4 |

![Results of the dropped components](../figures/phase6_dropped_components.png)

All runs: D1 (15 clients, α = 0.5, 3 attackers), 30 rounds, seeds 42, 123 and 7. Machines are named per result and never mixed in one comparison.

### 5.3a Worker momentum under L0 (`robust.momentum: 0.5`)

![Worker momentum: where it acts](figures/diagrams/11_dropped_momentum.png)

**What it does.** The server keeps a running average of each client's updates and gives L0 the averages instead of this round's updates:

$$m_i^{(t)} = \beta\, m_i^{(t-1)} + (1-\beta)\,\Delta_i^{(t)}, \qquad \Delta = \mathrm{TM}_f\Bigl(\mathrm{NNM}_f\bigl(\mathrm{ARC}_f(m_1^{(t)}, \dots, m_n^{(t)})\bigr)\Bigr)$$

**Legend:** $\Delta_i^{(t)}$ is client $i$'s update in round $t$; $m_i^{(t)}$ is its momentum, set to $\Delta_i^{(1)}$ the first time; $\beta = 0.5$ is the share of the past kept; ARC, NNM and TM are L0's three steps (6.2).

**Why we tried it.** In robust-optimization theory (Karimireddy, He and Jaggi, 2021), attackers hide inside the noise of honest updates. Averaging each client over rounds shrinks that noise, so robust rules separate attackers better.

**Results** (Mac, L0 alone, no screen):

| | No attack | Sign flip | ALIE | Min-Max | Backdoor (acc.) | Backdoor success |
|---|---|---|---|---|---|---|
| L0 | **47.56** | **41.65** | **47.56** | **45.47** | **46.48** | **62.5%** |
| L0 + worker momentum | 46.43 | 40.71 | 45.82 | 44.44 | 45.18 | 64.5% |

**Why it was dropped.** It is worse in every condition, by 0.9–1.7 points, and does not help against the backdoor. A likely reason, not tested on its own: the theory assumes one SGD step per round, where per-round noise is large. Here every upload already averages 5 local epochs, so there is little noise left to average away, and momentum mostly mixes in directions from earlier rounds, after the model has moved on.

### 5.3b Output-space signals: probe logits and head gap (`plus.signal: probe_logits`, `head_gap`)

![Output-space signals: where they act](figures/diagrams/12_dropped_output_signals.png)

**What they do.** Instead of comparing weights, the server runs each uploaded model on $P = 200$ probe images (the FLTrust root set, images outside every client's data, no labels needed) and compares what the models predict:

$$\text{probe logits: } p_i = \mathrm{vec}\bigl(z_r(x_1), \dots, z_r(x_P)\bigr), \quad s_i = \cos\bigl(p_i,\ \operatorname{med}_k p_k\bigr)$$

$$\text{head gap: } s_i = \frac{1}{P}\sum_{x} \Bigl[\mathrm{KL}\bigl(q_r(x)\,\|\,q_l(x)\bigr) + \mathrm{KL}\bigl(q_r(x)\,\|\,q_p(x)\bigr)\Bigr]$$

The score then goes through the unchanged R3 screen (two-sided robust z, gap gate, patience, evidence).

**Legend:** $z_r(x)$ are the root head's logits of upload $i$ on probe image $x$; $\mathrm{vec}$ stacks them into one vector of $P \times C$ numbers; $q_r, q_p, q_l$ are the softmax outputs of the root, parent and local heads; KL is the Kullback–Leibler divergence; the median is taken over the reference clients.

**Why we tried them.** The backdoor lives in the shared trunk, and the attackers never split off in weight-cosine space (root-head signal: 32–70% backdoor success). A backdoored model may still *behave* differently. The head gap adds a second idea: a backdoor moves all three heads together (small gap), while honest personalization moves the local head alone (larger gap).

**Results** (GPU):

| Signal | Sign flippers isolated | Min-Max isolated | Backdoor success, seeds 42 / 123 / 7 | Honest isolated, no attack (3 runs) |
|---|---|---|---|---|
| Root head (R3, for reference) | 9 of 9 | 9 of 9 | 70 / 32 / 68% | 4 |
| Probe logits | **0 of 9** | 9 of 9 | 71 / 74 / 83% | 1 |
| Head gap | **0 of 9** | 9 of 9 | 0.7 / 10 / 83% | **9** |

**Why they were dropped.** Neither catches the sign flippers, which every weight signal catches in round 1. Probe logits never catch the backdoor. Head gap catches it on two seeds, but at the price of 9 honest isolations, the most of any signal. Both also cost a forward pass of every upload on the probe set every round.

### 5.3c Sign-alignment signal (`plus.signal: sign_align`, AlignIns MPSA)

![Sign alignment: where it acts](figures/diagrams/13_dropped_sign_align.png)

**What it does.** From AlignIns (CVPR 2025): the principal sign is the majority sign of each coordinate over the reference clients. Each client is scored by how many of its largest coordinates agree with it:

$$\pi_j = \mathrm{sgn}\Bigl(\sum_k \mathrm{sgn}(\Delta_{k,j})\Bigr), \qquad s_i = \frac{1}{\lvert T_i\rvert}\sum_{j \in T_i} \mathbb 1\bigl[\mathrm{sgn}(\Delta_{i,j}) = \pi_j\bigr]$$

**Legend:** $\Delta_{i,j}$ is coordinate $j$ of client $i$'s update (weights only, BatchNorm buffers left out); $\pi_j$ is the principal sign; $T_i$ is the set of client $i$'s largest 30% of coordinates by size (`plus.sign_topk: 0.3`); $s_i$ is a share between 0 and 1, higher = more aligned. The score is scale-free: multiplying an update by 100 does not change it.

**Why we tried it.** Cosine scores are dominated by the largest coordinates and by update size. A sign count ignores size, so a backdoor spread over many small coordinates might show.

**Results** (GPU): every sign flipper and every Min-Max attacker isolated, with **no** honest isolation in any condition. Backdoor success 71 / 74 / 83% (seeds 42 / 123 / 7): no backdoor attacker caught.

**Why it was dropped.** It is the cleanest detector of model poisoning, but it misses the backdoor completely. A backdoor attacker trains honestly on half its data, so its largest coordinates mostly carry the majority sign; the trigger sits in a few coordinates the sign count does not weigh. Model poisoning is already caught by the cheaper weight signals.

### 5.3d Critical layers alone (`plus.signal: critical`)

![Critical layers: where they act](figures/diagrams/14_dropped_critical.png)

**What it does.** After FedSurrogate (arXiv 2605.11122): score each client layer by layer, and keep only the layers where clients disagree most:

$$c_{i\ell} = \cos\bigl(\Delta_i^{(\ell)},\ \operatorname{med}_k \Delta_k^{(\ell)}\bigr), \qquad L^\star = \mathrm{top}_5\ \mathrm{std}_i\, c_{i\ell}, \qquad s_i = \frac{1}{5}\sum_{\ell \in L^\star} c_{i\ell}$$

**Legend:** $\Delta_i^{(\ell)}$ is client $i$'s update to shared layer $\ell$ (the personal parent and local heads are never scored); $c_{i\ell}$ its cosine to that layer's median; $L^\star$ the 5 layers whose cosines vary most across clients this round (`plus.critical_k: 5`).

**Why we tried it.** A backdoor that sits in a few trunk layers is diluted in the whole-update cosine, which is dominated by the largest layers. Scoring the most disputed layers separately should expose it.

**Results** (GPU): every sign flipper and ALIE attacker isolated, no honest client isolated with no attack. Backdoor success 0.5 / 65 / 83% (seeds 42 / 123 / 7).

**Why it was dropped as a single signal, and kept in the union.** It catches the backdoor on one seed only. But it catches it on a **different** seed from the whole-update signal (0.6 / 71 / 0.6%), so the union of the two catches all three: this is FedHEP-Guard's signal [1] (chapter 6.4).

**The six signals side by side** (GPU, R3 with each signal):

![Six screening signals compared (GPU server)](../figures/phase6_signals.png)

| Signal | Backdoor success, seeds 123 / 42 / 7 | Honest isolated with no attack (3 seeds) |
|---|---|---|
| Root head | 32 / 70 / 68% | 4 |
| Whole update | **0.6** / 71 / **0.6**% | 13 |
| Critical layers | 65 / **0.5** / 83% | 0 |
| Sign alignment | 74 / 71 / 83% | 0 |
| Probe logits | 74 / 71 / 83% | 1 |
| Head gap | 10 / **0.7** / 83% | 9 |

### 5.3e Head-agreement inference (`fedhep.head_agreement: true`)

![Head agreement: where it acts](figures/diagrams/15_dropped_head_agreement.png)

**What it does.** A prediction-time rule only; training is unchanged. For each test input, if the client's two personal heads agree with each other and disagree with the FedHEP ensemble, the personal heads decide:

$$\hat y = \begin{cases} \arg\max\Bigl(\dfrac{z_l}{\sigma(z_l)} + \dfrac{z_p}{\sigma(z_p)}\Bigr) & \text{if } \hat y_p = \hat y_l \neq \hat y_{\mathrm{ens}} \\[6pt] \hat y_{\mathrm{ens}} & \text{otherwise} \end{cases}$$

**Legend:** $z_p, z_l$ are the parent and local heads' logits on the client's classes (ACLM mask); $\hat y_p, \hat y_l, \hat y_{\mathrm{ens}}$ are the classes predicted by the parent head, the local head and the ensemble (chapter 2.2); $\sigma(\cdot)$ is the standard deviation of a head's logits, so neither head dominates by scale. (The code also rescales the sum to the ensemble's spread, which does not change the predicted class.)

**Why we tried it.** The backdoor is learned by the shared trunk and mapped to its target class by the shared root head. The personal heads are trained on the client's own clean data, so they might keep the true label where the ensemble follows the trigger.

**Results.**
- **Cost** (Mac, 19 runs without a backdoor): 3.0% of inputs overruled, 0.18 points of accuracy on average (worst run 0.61).
- **Effect** (GPU, backdoor success with the ensemble → with head agreement):
  - Defended FedHEP 79 → 69%, L0 64 → 55%, R3 with the root head 57 → 50%;
  - FedHEP+ 22 → 18%, whole-update screen 24 → 23%, undefended FedHEP 79 → 78%.
- **Effect** (Mac, root-head screens with each rewind): 3.5–6.6 points on average, up to 11 on one seed.

**Why it was dropped.** It removes 1–13 points of backdoor success, never most of it. The parent head is shared by a group and carries the backdoor too (chart below), so on triggered inputs the personal heads rarely agree **against** the ensemble. Removing the attackers (the screen) and the backdoor (the rewind) does far more.

![Backdoor success of each head](../figures/phase6_backdoor_heads.png)

### 5.3f Holding $f$ fixed after exclusions (`plus.global_f_fixed: true`)

![Holding f fixed: where it acts](figures/diagrams/16_dropped_f_fixed.png)

**What it does.** L0's trimmed mean drops $f$ values per side, per coordinate. In FedHEP-Guard, $f$ falls by the number of clients the screen has excluded. The variant keeps $f$ at the assumed count:

$$\text{FedHEP-Guard: } f = \max\bigl(0,\ \min(f_0 - \lvert X\rvert,\ \lfloor (n_{\mathrm{kept}}-1)/2\rfloor)\bigr), \qquad \text{tested: } f = \min\bigl(f_0,\ \lfloor (n_{\mathrm{kept}}-1)/2\rfloor\bigr)$$

**Legend:** $f_0$ is the assumed attacker count (`plus.global_f`, 3 at D1); $X$ is the set of clients excluded this round; $n_{\mathrm{kept}}$ the number of clients aggregated.

**Why we tried it.** R3 lost 1.8 points to L0 under ALIE (the ALIE puzzle). One candidate cause: after the three attackers are isolated, $f$ falls to 0 and the trimmed mean becomes a plain mean, so a fixed $f$ might recover 0.3–0.8 points (prediction in `docs/PREDICTIONS.md`).

**Results** (R3, gate 8):

| | No attack | ALIE | ALIE, seeds 42 / 123 / 7 |
|---|---|---|---|
| Mac, $f$ falls (kept) | **47.36** | **45.74** | 46.63 / 46.30 / 44.28 |
| Mac, $f$ fixed at 3 | 47.16 | 44.65 | 45.86 / 45.15 / 42.93 |
| CUDA, $f$ falls (kept) | **47.89** | **46.33** | 46.87 / 45.85 / 46.27 |
| CUDA, $f$ fixed at 3 | 47.87 | 45.09 | 45.61 / 44.54 / 45.13 |

**Why it was dropped.** The prediction was wrong in direction, on both machines and every seed. Once the attackers are gone, a fixed $f = 3$ makes the trimmed mean discard the 3 highest and 3 lowest **honest** values of every coordinate, every round. That throws information away for nothing. With no attack nobody is excluded, so the two rules coincide and the difference is near zero.

### 5.3g Rewinding every shared weight (`plus.rewind: all`)

![Rewind scope: where it acts](figures/diagrams/17_dropped_rewind_all.png)

**What it does.** When a client is isolated, the Global model and every parent go back to the stored model from before the client's first suspicious round (chapter 6.6). A mask $M$ chooses which coordinates are restored:

$$w \leftarrow M \odot w_{t^\star} + (1 - M) \odot w, \qquad M_j = \begin{cases} 1 & j \in \text{trunk or root head (FedHEP-Guard, trunk\_root), else } 0 \\ 1 & \text{every } j \text{ (tested: all)} \end{cases}$$

**Legend:** $w$ is a shared model (Global or a parent); $w_{t^\star}$ the Global model stored at the start of round $t^\star$; $\odot$ is element-wise multiplication; `trunk` restores the trunk only.

**Why we tried it.** If the parent head carries the backdoor too (5.3e), rewinding only the trunk and root head might leave part of it in place.

**Results** (Mac, 3 backdoor attackers):

| Rewind | Whole-update screen: accuracy | backdoor success, seeds 42 / 123 / 7 | Root-head screen: accuracy | backdoor success, seeds 42 / 123 / 7 |
|---|---|---|---|---|
| none | **46.09** | 0.5 / 0.5 / 0.9% | — | — |
| trunk | 45.74 | 0.5 / 0.5 / 0.3% | **46.47** | 74 / 23 / 30% |
| trunk + root head (FedHEP-Guard) | 45.83 | 0.4 / 0.2 / 0.3% | 46.46 | 74 / 19 / 29% |
| everything | 45.25 | 0.5 / 0.3 / 0.3% | 46.46 | 74 / 16 / 27% |

**Why it was dropped.** With the whole-update screen it costs 0.6 points against rewinding the trunk and root head, and gains nothing: the attackers are caught by rounds 3–6, so there is little backdoor to remove. With the root-head screen it lowers success by 1–2 points on the seeds where an attacker is caught, at no accuracy cost, but that screen is not the one FedHEP-Guard uses. Rewinding the personal heads also discards the personalization learned since $t^\star$, which the attackers barely touch. With the trunk and root head rewound, the parent head's share of the backdoor plausibly has no trigger features left to respond to.
