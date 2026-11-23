# 8. Open problems and current experiments

What FedHEP-Guard does not solve yet, why, and what is being tested. Each problem gives its evidence and its status. The model is described in [chapter 6](06_fedhep_guard.md), its results in [chapter 7](07_fedhep_guard_results.md).

## 8.1 What is still open

| # | Problem | Evidence | Status |
|---|---|---|---|
| 1 | **Honest clients isolated under mild skew** | 4.3 per run at 15 clients and 14.7 at 30 (α = 0.5, GPU), against FedHEP+'s 2.7 and 4.3; about half are whole parent groups | cause found (8.2); the shared signal is being tested (8.3) |
| 2 | **A late catch lets a backdoor stay** | one stealth seed over 100 rounds: attackers caught in rounds 15–18, the rewind reaches back only 10 rounds, backdoor grew to 34% (7.3) | repeated with the shared signal in `acer_c12_shared_durability` (8.3) |
| 3 | **Backdoors under strong skew** | α = 0.1: 15–31% get through (GPU); at D2 no model tested stops the plain backdoor (38–47%); FedHEP-Guard not yet run at D2 | GPU benchmark (chapter 9) |
| 4 | **A third of the clients attacking with a backdoor** | 5 of 15: 12% (GPU) and one seed at 20% (Mac); FLAME holds | open |
| 5 | **Removing ALIE attackers costs accuracy** | catching all 9 ALIE attackers leaves the union screen 2.5 points below FedHEP+, which never catches them (GPU); fixing $f$ made it worse (5.3f) | open |
| 6 | **Untested attacks** | IPM and delayed-onset attacks from the team's `oreh` code; colluders that report one class mask | not scheduled |

## 8.2 Why honest clients are isolated

![Honest isolations by setting and signal](../figures/phase6_honest_isolation_summary.png)

- **The gap gate fixed them for the root-head signal:** 1.0 per run at D1 and D2 on the Mac (FedHEP+: 3.0 and 9.0), and 0.7–1.3 in three of four GPU settings.
- **The union screen isolates more under mild skew:** at α = 0.5, 4.3 per run at 15 clients and 14.7 at 30, above FedHEP+.
- **Under strong skew it isolates fewer:** at α = 0.1 and 15 clients, 1.7 against FedHEP+'s 7.7.
- **The D1 mean of 4.3 comes mostly from one seed** (10, 3 and 0 honest clients isolated).

There are two causes, and both act through the whole-update half of signal [1].

**Cause 1: the gap gate opens by chance, more often with more clients.** The gate opens when the largest gap between neighbouring sorted scores is at least 8 times the median gap. With more clients there are more gaps, and the largest of them grows by chance alone. For scores drawn from one normal distribution (no attacker at all):

| Clients | Chance the gate opens with no attacker | Median gap ratio | 95th percentile |
|---|---|---|---|
| 12 | 13% | 4.3 | 11.2 |
| 15 | 18% | 4.9 | 12.1 |
| 20 | 29% | 6.0 | 14.6 |
| 30 | **51%** | 8.1 | 19.2 |
| 50 | **81%** | 12.1 | 27.6 |

The logged gap ratios of the no-attack runs agree. At 30 clients and α = 0.5, the whole-update gate opened in 44% of rounds and the critical-layer gate in 43%, against 32% and 14% at 15 clients. The union acts when either gate is open, so it blames someone in most rounds.

**A larger gate is not the fix.** The obvious repair is a threshold that grows with the number of clients, $\gamma(n)$, for example the 95th percentile of the gap ratio in honest rounds (14 at 15 clients, 21 at 30). The logs show what it would cost. In the rounds where the largest gap split off exactly the backdoor attackers, the gap ratio was only 8.6 (median, 15 clients) and 14.1 (30 clients), inside the honest range. A gate at the honest 95th percentile would have kept about a quarter of those catches. Sign flip, ALIE and Min-Max split off at ratios of 60–130 (median) and are not at risk.

![Gap ratios: honest rounds, backdoor catches and model-poisoning catches](../figures/phase6_gap_gate_tradeoff.png)

So the gap ratio cannot tell an honest outlier from a backdoor attacker: both stand a little apart from the rest.

**A larger scale floor has the same problem** (offline, on the logged scores of the GPU grid; not a run). Raising the floor of the z-score's scale from $0.1\,\lvert\operatorname{med}(s)\rvert$ to $0.2\,\lvert\operatorname{med}(s)\rvert$:
- removes 27–32% of the honest client-rounds that would be flagged with no attack (15 and 30 clients, α = 0.5);
- removes 40–65% of the backdoor attackers' flagged rounds.


**Cause 2: whole parent groups drift away together.** At α = 0.5, about half of the union screen's honest isolations come as a whole parent group: at least 3 honest members of one group isolated in the same round.

![Honest clients isolated a whole parent group at a time](../figures/phase6_group_isolations.png)

- **Which screens do it:** union screen 51% (15 clients) and 56% (30 clients); whole update alone 53% and 73%; FedHEP+ 2–3%; root head 0% (GPU grid, every attack pooled).
- **On the Acer:** FedHEP-Guard's three mass isolations in the 100-round runs (rounds 13, 14 and 16) were exactly parent group 2, mostly group 2, and exactly group 1.
- **The cause.** The whole-update signal is every weight of the update, the personal heads included. The parent head $W_p$ is shared by the members of one group and differs between groups by design.
  - As training goes on, a group's updates drift away from the others together: in the example, the whole-update gap ratio grows from about 3 to about 20 over rounds 9–13, and the group's z-scores fall together until all pass the cut.
  - The critical-layer signal already leaves the personal heads out, and its gap ratio stays low in the same rounds.
- **Why mild skew is worse.** At α = 0.5 the clients' own data differ little, so the group direction dominates the differences between updates.


**What has been tried.** The peer reference judged each client against the clients with similar classes. It made things worse, because most of these isolations are not clients with unusual data (5.4).

## 8.3 Phase 8: the shared signal (running on the Acer)

**What changes.** Only the whole-update half of signal [1]. The new signal scores the update restricted to the coordinates every client shares, the trunk and the root head:

$$v_i^{\mathrm{shared}} = \mathrm{bound}\bigl(\Delta_i\big\vert_{\text{trunk} \,\cup\, W_r}\bigr), \qquad s_i = \cos\bigl(v_i^{\mathrm{shared}},\ \operatorname{med}_k v_k^{\mathrm{shared}}\bigr)$$

It is combined with the critical layers exactly as now: each signal z-scored and gap-gated on its own, and a client's z is the lower of the two.

![The shared signal: what is scored](figures/diagrams/18_shared_signal.png)

**Legend:**

| Symbol | Meaning | Config key, value |
|---|---|---|
| $\Delta_i\big\vert_{\text{trunk} \cup W_r}$ | client $i$'s weight update restricted to the trunk and the root head (BatchNorm buffers, $W_p$ and $W_l$ left out) | `plus.signal: shared` |
| bound | the norm bound of chapter 3, $k = 3$ | `plus.norm_bound` |
| — | union with the critical layers | `plus.signal: shared_critical` |

**Why it should keep the backdoor catches.** The backdoor is learned by the shared trunk and root head, and both stay in the signal. The risk: the parent head also carries part of the backdoor (chapter 5.3e), and on some seeds that part may have been what split the attackers off.

**The experiment** (Acer, predictions P26–P29):

| Sweep | Runs | What it tests | Controls (Acer) |
|---|---|---|---|
| `acer_c11_shared.yaml` (`x11_`) | 21 | shared union screen without rewind: no attack, sign flip, ALIE, Min-Max, backdoor at α 0.5; no attack and backdoor at α 0.1 | `x10_*__r3wc_shp`, `x8_f3__*__r3wc` |
| `acer_c11_control.yaml` | 3 | union screen under Min-Max at α 0.5 | — |
| `acer_c12_shared_durability.yaml` (`x12_`) | 6 | FedHEP-Guard with the shared signal, 100 rounds, both backdoors | `x9__*__r3wc_tr` |

Tests: `tests/test_shared_signal.py`. It checks that only the personal heads are left out, and that a drifting parent group is split off by the whole-update signal but not by the shared one. It also runs the shared union end to end.


## 8.4 What comes next

- **GPU server, when the lab reopens:**
  - the canonical benchmark (chapter 9), with the full FedHEP-Guard under every attack at D1 and D2;
  - the stealth study at scale (`gpu_c7_stealth`);
  - the 50-client part of the grid (P19).
- **Acer:** Phase 8 results (P26–P29). If they hold, the shared union becomes FedHEP-Guard's signal [1], and the benchmark's `guard` model is re-run with it.
- **After that:** the open problems 3–6 above, in that order.
