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

### What the screen's own logs show (offline replay of 196 GPU runs)

Every run of FedHEP-Guard's screen logs, each round, both signals' scores, their median and scale, and their gap ratio. `scripts/replay_screen.py` rebuilds the screen's decisions from these logs; replaying the real rule reproduces the real isolations in 195 of the 196 GPU runs (the other differs in one isolation in round 29). The same logs then show how well the scores separate attackers from honest clients, and what other rules would do.

**Who gets isolated, and how** (every real isolation in the GPU grid's union-screen runs):

| Isolated clients | Number | Flagged by | Sign of z (whole update, critical layers) | Round (median) | Gap ratio when flagged (median) |
|---|---|---|---|---|---|
| Sign flip, ALIE, Min-Max attackers | 721 | both signals in 88% | high–high 64%, low–low 36% | 3 | 131 |
| Backdoor attackers | 162 | the whole update alone in 54%, both in 23% | **low–low 75%** | 6 | 11 |
| Honest clients, no attack | 205 | one signal in 80% | mixed signs in 53% | 8 | 14 |
| Honest clients, attack runs | 1,095 | one signal in 85% | mixed signs in 49% | 12 | 11 |

**Three findings.**
1. **The gap gate does not tell backdoor rounds from honest ones.** In rounds with no attack it opens in 17–19% of rounds at 15 clients, 30–35% at 30 and 40–49% at 50. In backdoor rounds it opens just as often (15–28%, about 40%, 43–55%). In sign-flip, ALIE and Min-Max rounds it opens in 93–100%. So the gate does its job against model poisoning, but backdoor attackers are caught only when it happens to be open while they stand out.
2. **The two-sided fold costs backdoor separation.** Backdoor attackers sit on the low side: their updates point away from the others. Folding the high side in, which the two-sided screen does to catch ALIE and Min-Max, adds the honest clients that are unusually close to the median. Per-round AUROC of backdoor attackers against honest clients (rounds 1–10), whole-update signal:

| | 15 clients | 30 clients | 50 clients |
|---|---|---|---|
| two-sided $\lvert z\rvert$ (what the screen uses) | 0.82 | 0.81 | 0.65 |
| low side only ($-z$) | 0.93 | 0.84 | 0.73 |
| low side, smoothed over rounds (EMA, 0.7) | **0.96** | **0.85** | **0.81** |
| critical layers, two-sided | 0.75 | 0.65 | 0.50 |

![Separability of backdoor attackers from honest clients](../figures/replay_separability.png)

3. **Separation falls with the number of clients,** for every signal: with more honest clients, more of them are as unusual as an attacker. This is the root of the growing honest isolations (cause 1); the gate opening by chance is how it shows.

**Replayed rules.** Each point is one rule: honest clients isolated per run with no attack, and the share of backdoor attackers caught. Attacker catches of stricter rules are lower bounds: a client the real run isolated is not scored afterwards, so a rule that would have caught it a few rounds later is counted as missing it. The model trajectory is also the real run's.

![Replayed decision rules](../figures/replay_rules.png)

| Rule (replayed) | Honest per run, no attack: 15 / 30 / 50 clients | Backdoor attackers caught: 15 / 30 / 50 | ALIE caught |
|---|---|---|---|
| FedHEP-Guard's screen (now) | 3.0 / 10.8 / 23.6 | 70 / 80 / 57% | 100% |
| **high side only when the gap ratio reaches 20** | **2.3 / 8.8 / 10.6** | 68 / 72 / 43% (lower bounds) | 97% |
| high side only at 30 | 2.3 / 8.3 / 9.6 | 68 / 72 / 43% | 94% |
| isolation cut 2.0 instead of 1.5 | 1.3 / 4.0 / 7.6 | 48 / 45 / 13% | 100% |

No rule removes honest isolations without losing some backdoor catches: with these two signals the two groups overlap. The high-side gate is the best trade in the replay. At 50 clients it halves the honest isolations; the drop in backdoor catches is small at 15 clients and partly an artefact of the lower bound. Live runs test it on the Acer (`acer_c14_high_gate.yaml`, P33–P36). The smoothed low-side score separates best of all, which points to the next signal to build: a per-client score accumulated over rounds before any gate is applied, as FLDetector and FLTracer do (`LITERATURE_REVIEW.md`).

## 8.3 Phase 8 at scale (Acer)

The shared signal, tested at 15 clients, cut honest isolations by three quarters at α = 0.5 but let the backdoor through on one seed in three, so it was not adopted (chapter 7.6). Its scale test asked how much of the honest-isolation problem at 30 and 50 clients is cause 2, which the shared signal removes, and how much is cause 1, which it does not.

| Sweep | Runs | What it tests | Status |
|---|---|---|---|
| `acer_c13_shared_scale.yaml` (`x13_`) | 18 | the shared union screen against the union screen at 30 clients (α 0.5, no attack and 6 backdoor attackers) and 50 clients (α 0.1, no attack) (P30–P32) | done (30 clients in 7.6, 50 clients below) |

**At 50 clients (α = 0.1, no attack) both causes act, and cause 1 is the larger.**

| Acer, 50 clients, α 0.1, no attack | Union screen | Shared union screen |
|---|---|---|
| Honest clients isolated, seeds 123 / 42 / 7 | 24 / 21 / 22 | 17 / 11 / 17 |
| Per run | **22.3** (GPU: 22.0) | **15.0** |
| In whole-group events (3 runs) | 27 of 67 | 13 of 45 |
| Single clients (3 runs) | 40 of 67 | 32 of 45 |
| First honest isolation | round 3 | round 3 (seed 42: 12) |
| Accuracy | 48.41 | 49.68 |

The shared signal removed a third of the honest isolations (22.3 → 15.0), half of the whole-group ones and a fifth of the single ones (P31 held: less than 60% removed). Even if every whole-group isolation were removed, 13 honest clients per run would still be isolated one by one. So at scale the main cause is cause 1, the gap gate opening on honest outliers, and the next fix has to target it.

**At 30 clients with no attack, cause 1 dominates on the Acer:** the union screen isolated 6.0 honest clients per run, only 1.0 of them in whole-group events, and the shared signal isolated 7.3. On the GPU the same setting gave 14.7 per run with about half in groups (8.2), so the split differs by run and machine; both machines agree that removing the parent head is not enough.

## 8.4 What comes next

- **GPU server:** the benchmark (6.1) and the rest of the 50-client grid, about 6 more days on 3 GPUs; then the stealth study at scale (`gpu_c7_stealth`).
- **Acer:** the high-side gate (`acer_c14_high_gate.yaml` with `acer_c14_control.yaml`, P33–P36): the union screen with `plus.high_gate: 20` at 15 clients (no attack, backdoor, stealth backdoor, ALIE, Min-Max) and 50 clients (no attack, backdoor).
- **A smoothed per-client score.** The replay's best separator is each client's low-side score averaged over rounds (8.2). A screen that gates on that score, rather than on the round's scores, is the next candidate; the replay can test it first. An observe-only mode (score and log every round, never exclude) would remove the replay's lower-bound limit.
- **Cause 2 without dropping the heads.** The heads carry both the group signature and part of the backdoor signature (7.6). A rule that spares a group drifting only in its heads, by requiring the shared signal to agree at a lower bar, can be replayed once both signals are logged on the same updates.
- **After that:** the open problems 3–6 above, in that order.
