# 4. FedHEP+: find attackers and remove them (Phase 3 and the follow-up diagnostics)

**Idea:** Defended FedHEP already ranks attackers below honest clients, but never acts on the ranking ([chapter 3](03_defended_fedhep.md)). FedHEP+ turns the ranking into removal. It uses a cut that keeps its meaning as training goes on, and a reference that a minority cannot pull.

## 4.1 What FedHEP+ changes in Defended FedHEP

FedHEP+ keeps SCCF for aggregation and replaces the trust ledger with **screening**. Screening decides which clients enter both tiers. It also re-forms the groups by weight similarity.

![FedHEP+: one round, changes against Defended FedHEP](figures/diagrams/05_fedhep_plus_round.png)

**[1] Score.** The cosine of each client's (norm-bounded) root-head signal $v_i$ to the **coordinate-wise median** update $\tilde r$. A median is not pulled by a minority the way a mean is:

$$s_i = \cos(v_i, \tilde r), \qquad \tilde r_j = \operatorname{med}_k\, v_{k,j}$$

**[2] Robust z-score.** How far client $i$ falls below the typical score, in units of the typical spread:

$$z_i = \frac{s_i - \operatorname{med}(s)}{\max\bigl(1.4826\cdot\mathrm{MAD}(s),\ \rho\,\lvert\operatorname{med}(s)\rvert,\ a\bigr)}$$

- $1.4826\cdot\mathrm{MAD}$ estimates the standard deviation from the middle of the data, so a few attackers cannot inflate it.
- The floor $\rho\lvert\operatorname{med}(s)\rvert$ ($\rho = 0.1$) stops a very tight consensus from turning tiny differences into huge z.
- The z-score is unchanged when every cosine shrinks together. This is the property that Defended FedHEP's fixed temperature lacks.
- Fewer than 5 senders are not screened.

**[3] Exclusion.** A client with $z_i < -2.5$ is left out of this round's aggregation.

**[4] Evidence and isolation.** An exponential moving average of z decides permanent isolation, so one bad round is not enough:

$$E_i \leftarrow 0.7\,E_i + 0.3\,z_i, \qquad E_i < -1.5 \Rightarrow i \text{ isolated for good}$$

**[5] Regrouping.** Kept clients are re-grouped by the similarity of their **weight** updates, with BatchNorm counters left out (the counter problem of [chapter 2.5](02_fedhep.md)).

**Legend:**

| Symbol | Meaning | Config key, value |
|---|---|---|
| $v_i$ | client $i$'s signal: root-head rows of its classes, norm-bounded | `plus.signal: root_head` |
| $\tilde r$ | coordinate-wise median of the signals | `plus.reference: median` |
| $s_i$ | client $i$'s score | — |
| $\rho$ | relative floor of the scale | `plus.relative_floor: 0.1` |
| $a$ | absolute floor of the scale (0 = off) | `plus.mad_floor: 0` |
| $-2.5$ | exclusion cut | `plus.z_cut: 2.5` |
| $E_i$, $0.7$ | evidence and its decay | `plus.evidence_decay: 0.7` |
| $-1.5$ | isolation cut | `plus.evidence_cut: 1.5` |

## 4.2 What it changes against Defended FedHEP

| | Defended FedHEP | FedHEP+ |
|---|---|---|
| Reference | mean of unit vectors | coordinate-wise **median** update |
| Turning scores into a decision | softmax at a fixed temperature | **robust z-score** (scale-free) |
| Removing a client | ledger score below 0.25 (never happened) | left out at $z < -2.5$; **isolated for good** at $E < -1.5$ |
| Who is aggregated | everyone, down-weighted | only kept clients (SCCF over them) |
| Groups | random, fixed | **regrouped** by weight similarity |

## 4.3 Results (Phases 3 and 3b)

D1, 3 attackers, Mac, 3 seeds. The comparison rows are the Phase 2 runs with the same seeds and attackers.

![FedHEP+ against Defended FedHEP, three seeds](../figures/phase3b_confirm.png)

| Model | No attack | Sign flip | Label flip | Random noise | Backdoor (acc.) | Backdoor success |
|---|---|---|---|---|---|---|
| Defended FedHEP | 48.39 ± 0.72 | 38.08 ± 1.01 | 43.23 ± 0.40 | 45.80 ± 0.70 | 46.42 ± 0.37 | 78.7% |
| **FedHEP+** | 47.92 ± 0.60 | **46.34 ± 0.87** | **46.34 ± 0.87** | 45.41 ± 1.15 | 46.26 ± 0.56 | **20.8%** |
| Change, paired by seed | −0.47 | **+8.26** | **+3.11** | −0.38 | −0.16 | **−57.9 points** |

- The gains under sign flip, label flip and backdoor hold **on every seed**, by far more than the seed spread. This is the project's first confirmed improvement.
- Sign and label flippers are isolated in round 1. That is why both rows are identical: from round 2 on, the same 12 honest clients train alone.
- The median reference is what exposes the backdoor: with the mean reference, backdoor success stays near 80% (seed 42: 33% against 80%).

![FedHEP+ backdoor](../figures/phase3b_backdoor.png)

## 4.4 Variants tried and rejected inside Phase 3

| Variant | What it changes | Result | Kept? |
|---|---|---|---|
| Readmission (`plus.readmit`) | isolated clients keep being scored and return once $E_i > -0.75$ | attackers came back late in training: backdoor 21% → 39%, sign flip −1.7 | no |
| Cap (`plus.max_excluded: 0.34`) | at most a third left out per round | let backdoor attackers back in (59–62%) | no |
| Full-population yardstick (`plus.population: all`) | median and MAD over every non-banned client, isolated ones too | more false isolations (87 against 46) and the backdoor got through (54%) | no |
| Whole-update signal, one-sided (`plus.signal: weights`) | cosine of the whole update instead of the root head | backdoor 0.3% on seed 42 but many honest isolations | not then; it returns in Phase 6 |

![The false-isolation cascade](../figures/phase3_cascade.png)

**The cascade.** Each isolation shrinks the group the next z-scores are measured against. The spread shrinks with it, so the next honest client looks more extreme. Re-grouping and scoring only the clients still kept limited this, but did not remove it.

## 4.5 Where FedHEP+ fails: the follow-up diagnostics

The professor's questions led to five tests that changed the picture (Mac and Acer, `followup_*.yaml`).

**a) Why honest clients get extreme z mid-training.** The spread does **not** shrink: it grows six- to eightfold after round 14. What fades is the floor. Around round 14 the denominator is at its smallest, so a normal difference looks extreme, and one bad round can isolate an honest client for good. Requiring 3 bad rounds in a row (patience 3) cut false isolations by 42%.

![Why honest z-scores get extreme](../figures/why_z_grows.png)

**b) Adaptive attacks are invisible to a one-sided screen.** Under ALIE and Min-Max, no defense isolated a single attacker. These attackers sit **closer** to the reference than honest clients do, so their z is far **above** the others, while FedHEP+ looks only below. Min-Max cost about 3 points under most defenses.

![Adaptive attacks](../figures/followup_adaptive.png)

**c) It does not carry over to stronger skew.** In the held-out D2 (α = 0.2, 20 clients), FedHEP+ isolated about 9 of 20 **honest** clients per run with no attack. It lost 2 points to Defended FedHEP on every seed, and the bottom-10% lost 4. The clients it flags are the ones with unusual data. It still won under sign flip (+1.5 on every seed). No method stopped the backdoor there (43–47%).

![Held-out setting, α = 0.2](../figures/followup_alpha02.png)

**d) Many sign flippers: far ahead.** With 5 or 6 of 15 attackers, FedHEP+ kept 40–44%, while Defended FedHEP fell to 27–32% and undefended methods collapsed. With 10 of 15 (a majority) every method failed, as theory predicts.

![Accuracy against the number of attackers](../figures/followup_attack_rate.png)

**e) A backdoor outlives its attackers.** With 20 clients and 5 backdoor attackers, FedHEP+ isolated every attacker by rounds 11–15. Backdoor success then **kept rising** by 17–47 points by round 30. Once a backdoor is in the shared trunk, removing the attackers does not remove it.

![Backdoor persists after isolation](../figures/followup_backdoor_persists.png)

## 4.6 What improved, and what still fails

**Improved over Defended FedHEP (D1):** sign flip +8.3, label flip +3.1, backdoor success 79% → 21%, and the bottom-10% under sign flip 31.2 → 40.8. All of this costs 0.5 points with no attack.

**Still fails:**
1. **Honest but unusual clients look like attackers**: about 9 of 20 isolated at α = 0.2.
2. **Adaptive attackers are not seen**, because the screen is one-sided.
3. **A backdoor that got in stays**, even after its attackers are removed.
4. **More colluders pull the median**: with 5 backdoor attackers, success rises to 56% (chapter 5.2).

The lesson: "far from the typical update" catches crude attacks well, even many of them. But honest clients with unusual data are also far from typical, and clever attackers stay close to it. The next two chapters follow this lesson: [chapter 5](05_ideas_that_did_not_work.md) covers what did not fix it, and [chapter 6](06_fedhep_guard.md) what did.
