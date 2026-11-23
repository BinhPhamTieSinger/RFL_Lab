# 4. FedHEP+: find attackers and remove them (Phase 3)

**Idea:** Defended FedHEP already ranks attackers below honest clients, but never acts on the ranking ([chapter 3](03_defended_fedhep.md)). FedHEP+ turns the ranking into removal. It uses a cut that keeps its meaning as training goes on, and a reference that a minority cannot pull.

This chapter describes the model and its variants. Its results are in chapter 6.5–6.7; the variants that were tried and rejected are explained in full in chapter 7.1.

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

**[5] Regrouping.** Kept clients are re-grouped by the similarity of their **weight** updates, with BatchNorm counters left out (the counter problem of [chapter 6.3](06_results.md)).

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

### Why each part is in FedHEP+

Each part answers one of Defended FedHEP's failures ([chapter 6.4](06_results.md)).

| Part | What it is for | Why it is in the model (the failure it answers) | Evidence it helps |
|---|---|---|---|
| [1] Median reference | what "normal" looks like | the mean is pulled toward a coordinated minority, so backdoor attackers looked more trustworthy than honest clients (AUROC 0.36) | backdoor success, seed 42: 80% with the mean reference → 33% with the median |
| [2] Robust z-score | how unusual a client is, in units that do not drift | Defended FedHEP ranked sign flippers correctly (AUROC 0.97) but never acted: its fixed temperature fades as all cosines shrink. A z-score is unchanged when everything shrinks together | sign and label flippers isolated in round 1 |
| [3] Exclusion at $z < -2.5$ | keep a suspicious update out this round | evidence takes rounds to build, and the damage happens meanwhile | not measured on its own |
| [4] Evidence and permanent isolation | remove persistent offenders, forgive one bad round | the trust ledger never removed anyone; letting isolated clients back in readmitted attackers (backdoor 21% → 39%) | sign flip +8.3, label flip +3.1, backdoor 79% → 21% |
| [5] Regrouping by weights | groups that reflect real similarity after removals | FedHEP's grouping cosine is dominated by BatchNorm counters (6.3), and isolations empty groups | limited the false-isolation cascade (6.5) |

## 4.2 What it changes against Defended FedHEP

| | Defended FedHEP | FedHEP+ |
|---|---|---|
| Reference | mean of unit vectors | coordinate-wise **median** update |
| Turning scores into a decision | softmax at a fixed temperature | **robust z-score** (scale-free) |
| Removing a client | ledger score below 0.25 (never happened) | left out at $z < -2.5$; **isolated for good** at $E < -1.5$ |
| Who is aggregated | everyone, down-weighted | only kept clients (SCCF over them) |
| Groups | random, fixed | **regrouped** by weight similarity |

## 4.3 Variants

Four variants of FedHEP+ were tested on the same seeds. None was kept; each one's formula, the reason it was tried and why it failed are in [chapter 7.1](07_ideas_that_did_not_work.md).

| Variant | Config | What it changes | Outcome |
|---|---|---|---|
| Readmission | `plus.readmit: true` | an isolated client is still scored and returns when its evidence recovers to $-0.75$ | readmitted attackers, not honest clients; backdoor 21% → 39% |
| The cap | `plus.max_excluded: 0.34` | at most a third of the clients may be left out in one round | let attackers back in when reached; backdoor 59–62% |
| Full-population yardstick | `plus.population: all` | median and MAD over every client not banned, isolated ones included | more honest isolations (87 against 46), backdoor 52–54% |
| Whole-update signal, one-sided | `plus.signal: weights` | the cosine of the whole update instead of the root head | stopped the backdoor on one seed but isolated many honest clients; returns in FedHEP-Guard (chapter 5.4) |

Two further settings changed a single number and are listed for completeness: patience 3 (`plus.exclusion_patience`, `plus.isolation_patience`: three bad rounds in a row before acting, chapter 6.6a) and an absolute scale floor (`plus.mad_floor` 0.01 or 0.03, chapter 6.6a). Patience 3 became part of FedHEP-Guard.
