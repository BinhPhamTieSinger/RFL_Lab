# 8. Open problems and current experiments

What FedHEP-Guard does not solve yet, why, and what is being tested. Each problem gives its evidence and its status. The model is described in [chapter 5](05_fedhep_guard.md), its results in [chapter 6](06_results.md).

## 8.1 What is still open

| # | Problem | Evidence | Status |
|---|---|---|---|
| 1 | **Honest clients isolated, more with more clients** | per run with no attack (GPU): 4.3 and 14.7 at 15 and 30 clients (α = 0.5); 1.7, 7.0 and 22.0 at 15, 30 and 50 clients (α = 0.1). FedHEP+: 1.0–7.7. About half are whole parent groups | two causes found (8.2). The shared signal (Phase 8) removed most whole-group isolations at 15 clients but let the backdoor through on one seed in three, so it was not adopted (7.6); its scale test runs next (8.3). Cause 1 has no fix yet |
| 2 | **A late catch lets a backdoor stay** | one stealth seed over 100 rounds: attackers caught in rounds 15–18, the rewind reaches back only 10 rounds, backdoor grew to 34% (6.10) | repeated with the shared signal in `acer_c12_shared_durability` (8.3) |
| 3 | **Backdoors under strong skew** | α = 0.1: 15–31% get through (GPU); at D2 no model tested stops the plain backdoor (35–48% on both machines; the whole update stops it on one seed of three); FedHEP-Guard not yet run at D2 | GPU benchmark (chapter 6.1) |
| 4 | **A third of the clients attacking with a backdoor** | 5 of 15: 12% (GPU) and one seed at 20% (Mac); 9 of 30: 5.3% (GPU; one seed at 11%); FLAME holds | open |
| 5 | **Removing ALIE attackers costs accuracy** | catching all 9 ALIE attackers leaves the union screen 2.5 points below FedHEP+, which never catches them (GPU); fixing $f$ made it worse (7.4f) | open |
| 6 | **Untested attacks** | IPM and delayed-onset attacks from the team's `oreh` code; colluders that report one class mask | not scheduled |

## 8.2 Why honest clients are isolated

![Honest isolations by setting and signal](../figures/phase6_honest_isolation_summary.png)

- **The gap gate fixed them for the root-head signal:** 1.0 per run at D1 and D2 on the Mac (FedHEP+: 3.0 and 9.0), 1.7 at D2 on the GPU (FedHEP+: 8.7), and 0.7–1.3 in three of the GPU grid's settings. It is less reliable at α = 0.1 with many clients: 5.0 at 30 clients and 6.0 at 50 (all 18 of the 50-client ones on one seed).
- **The union screen isolates more honest clients than FedHEP+ in every GPU grid setting but one,** and the count grows with the number of clients:

| Clients | α = 0.5: FedHEP+ / union screen | α = 0.1: FedHEP+ / union screen |
|---|---|---|
| 15 | 2.7 / 4.3 | 7.7 / **1.7** |
| 30 | 4.3 / 14.7 | 2.3 / 7.0 |
| 50 | running | 1.0 / **22.0** |

- **The one exception, 15 clients at α = 0.1,** is where FedHEP+ is at its worst. It was the basis of an earlier claim that FedHEP-Guard fixes honest isolations under strong skew; the larger settings show it does not.
- **The D1 mean of 4.3 comes mostly from one seed** (10, 3 and 0 honest clients isolated). At 50 clients it is every seed (25, 21 and 20), starting in rounds 3–6.

There are two causes, and both act through the whole-update half of signal [1].

**Cause 1: the gap gate opens by chance, more often with more clients.** The gate opens when the largest gap between neighbouring sorted scores is at least 8 times the median gap. With more clients there are more gaps, and the largest of them grows by chance alone. For scores drawn from one normal distribution (no attacker at all):

| Clients | Chance the gate opens with no attacker | Median gap ratio | 95th percentile |
|---|---|---|---|
| 12 | 13% | 4.3 | 11.2 |
| 15 | 18% | 4.9 | 12.1 |
| 20 | 29% | 6.0 | 14.6 |
| 30 | **51%** | 8.1 | 19.2 |
| 50 | **81%** | 12.1 | 27.6 |

The logged gap ratios of the no-attack runs agree. At 30 clients and α = 0.5, the whole-update gate opened in 44% of rounds and the critical-layer gate in 43%, against 32% and 14% at 15 clients. The union acts when either gate is open, so it blames someone in most rounds. The 50-client runs confirm the prediction of this table: at 81% chance of an open gate per signal, the union screen isolates 22 honest clients per run at α = 0.1.

**A larger gate is not the fix.** The obvious repair is a threshold that grows with the number of clients, $\gamma(n)$, for example the 95th percentile of the gap ratio in honest rounds (14 at 15 clients, 21 at 30). The logs show what it would cost. In the rounds where the largest gap split off exactly the backdoor attackers, the gap ratio was only 8.6 (median, 15 clients) and 16.1 (30 clients), inside the honest range. A gate at the honest 95th percentile would have kept 16% (15 clients) and 27% (30 clients) of those catches. Sign flip, ALIE and Min-Max split off at ratios of 60–130 (median) and are not at risk.

![Gap ratios: honest rounds, backdoor catches and model-poisoning catches](../figures/phase6_gap_gate_tradeoff.png)

So the gap ratio cannot tell an honest outlier from a backdoor attacker: both stand a little apart from the rest.

**A larger scale floor has the same problem** (offline, on the logged scores of the GPU grid; not a run). Raising the floor of the z-score's scale from $0.1\,\lvert\operatorname{med}(s)\rvert$ to $0.2\,\lvert\operatorname{med}(s)\rvert$:
- removes 27–32% of the honest client-rounds that would be flagged with no attack (15 and 30 clients, α = 0.5);
- removes 40–65% of the backdoor attackers' flagged rounds.


**Cause 2: whole parent groups drift away together.** At α = 0.5, about half of the union screen's honest isolations come as a whole parent group: at least 3 honest members of one group isolated in the same round.

![Honest clients isolated a whole parent group at a time](../figures/phase6_group_isolations.png)

- **Which screens do it:** union screen 51% (15 clients) and 56% (30 clients); whole update alone 53% and 74%; FedHEP+ 2–3%; root head 0% (GPU grid, every attack pooled).
- **On the Acer:** FedHEP-Guard's three mass isolations in the 100-round runs (rounds 13, 14 and 16) were exactly parent group 2, mostly group 2, and exactly group 1.
- **The cause.** The whole-update signal is every weight of the update, the personal heads included. The parent head $W_p$ is shared by the members of one group and differs between groups by design.
  - As training goes on, a group's updates drift away from the others together: in the example, the whole-update gap ratio grows from about 3 to about 20 over rounds 9–13, and the group's z-scores fall together until all pass the cut.
  - The critical-layer signal already leaves the personal heads out, and its gap ratio stays low in the same rounds.
- **Why mild skew is worse.** At α = 0.5 the clients' own data differ little, so the group direction dominates the differences between updates.
- **Under strong skew it depends on the number of clients.** At α = 0.1 none of the union screen's honest isolations at 15 and 30 clients were whole groups: they were single clients, which points to cause 1. At 50 clients 44% were whole groups (29 of 66, no attack). So at 50 clients both causes act, and fixing cause 2 alone would remove about half.


**What has been tried.** The peer reference judged each client against the clients with similar classes. It made things worse, because most of these isolations are not clients with unusual data (7.5).

## 8.3 Phase 8 at scale (Acer)

The shared signal, tested at 15 clients, cut honest isolations by three quarters at α = 0.5 but let the backdoor through on one seed in three, so it was not adopted (chapter 7.6). Its scale test asked how much of the honest-isolation problem at 30 and 50 clients is cause 2, which the shared signal removes, and how much is cause 1, which it does not.

| Sweep | Runs | What it tests | Status |
|---|---|---|---|
| `acer_c13_shared_scale.yaml` (`x13_`) | 18 | the shared union screen against the union screen at 30 clients (α 0.5, no attack and 6 backdoor attackers) and 50 clients (α 0.1, no attack) (P30–P32) | 30 clients done (7.6); 50 clients stopped with an out-of-memory error in L0's nearest-neighbour step, fixed, to be rerun |

**At 30 clients with no attack, cause 1 dominates on the Acer:** the union screen isolated 6.0 honest clients per run, only 1.0 of them in whole-group events, and the shared signal isolated 7.3. On the GPU the same setting gave 14.7 per run with about half in groups (8.2), so the split differs by run and machine; both machines agree that removing the parent head is not enough.

## 8.4 What comes next

- **GPU server:**
  - now running: the rest of the 50-client grid (P14, P19), about one day;
  - queued next: the canonical benchmark (chapter 6.1), with the full FedHEP-Guard under every attack at D1 and D2, then the stealth study at scale (`gpu_c7_stealth`).
- **Acer:** Phase 8 at scale (`acer_c13_shared_scale`, P30–P32), to measure how much of the honest-isolation problem at 30 and 50 clients is cause 2.
- **A fix for cause 2 that keeps the backdoor.** The heads carry both the group signature and part of the backdoor signature (7.6), so the next candidate should tell them apart rather than drop the heads. One option: keep the whole-update signal, but isolate a client only if it also stands out on the shared signal at a lower bar (for example $|z| \ge 2$), so that a group drifting only in its heads is spared. To test such rules without new acting runs, the screen should log the scores of both signals on the same updates, as the floor counterfactual did (8.2).
- **Cause 1 needs its own fix.** At 50 clients half the honest isolations are single clients flagged because the gap gate opens by chance. A gate that scales with $n$ costs backdoor catches (above). Candidates to test once Phase 8 has reported: requiring both signals to agree when $n$ is large, or a gate on the gap's size relative to the score scale rather than to the median gap.
- **After that:** the open problems 3–6 above, in that order.
