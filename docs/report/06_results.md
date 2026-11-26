# 6. Results

Every result in one chapter, after the models. Section 6.1 is the **benchmark**: the general charts, every model under every condition on one machine. The sections after it are the **diagnostic** results, in the order of the models: they explain why a model behaves as it does, with charts that only some models have or that compare a few models directly.

| Section | What it answers | Machine |
|---|---|---|
| 6.1 The benchmark | which model is best, under which condition and setting | GPU server |
| 6.2 Reproduction | did we rebuild FedHEP and Defended FedHEP correctly? | Mac |
| 6.3 The grouping counter | why FedHEP's parents always copy Global | Mac |
| 6.4 Defended FedHEP under attack | where the published defense fails | Mac |
| 6.5–6.7 FedHEP+ | what it fixed, the cascade, why honest z-scores grow mid-training, where it fails | Mac, Acer |
| 6.8 FedHEP-Guard scorecard | what FedHEP-Guard fixed from FedHEP+ | all three |
| 6.9–6.13 FedHEP-Guard diagnostics | the ladder, backdoor recovery, the adaptive backdoor, a wrong $f$, scale | all three, one per table |
| 6.14 Earlier runs per machine | every model run at D1, per machine | Mac, GPU |

## 6.1 The benchmark: every model, every condition, one machine

The benchmark compares **every model and variant ever tested under the same eight conditions, in every setting ever tested, on one machine**: the lab GPU server. It is the comparison to show when someone asks which model is best. The sections after it diagnose specific problems, often on other machines and with charts that only some models have.

| | |
|---|---|
| Machine | lab GPU server (NVIDIA RTX PRO 4500, CUDA, deterministic kernels) |
| Models | 54: FedAvg, Multi-Krum, FedHEP, Defended FedHEP, FedHEP+, FedHEP-Guard and every variant of them ever tested, plus the established robust rules (Median, Trimmed mean, Bulyan, FLAME, FLTrust) |
| Conditions | no attack, sign flip, label flip, random noise, ALIE, Min-Max, backdoor, stealth backdoor |
| Settings | D1 (15 clients, α = 0.5), D2 (20 clients, α = 0.2), 30 clients α = 0.5, 15 clients α = 0.1, 30 clients α = 0.1, 50 clients α = 0.1, 50 clients α = 0.5, 20 clients α = 0.5, 15 clients α = 0.05 (the paper's) |
| Attackers | 20% of the clients (3, 4, 6 or 10), the same clients for every model in a given seed; robust rules are told the true count |
| Seeds, rounds | 42, 123, 7; 30 rounds |
| Sweep files | `configs/sweeps/bench_0_fedavg_family.yaml` and `bench_1_…` to `bench_9_…`, one per setting, written by `scripts/make_benchmark.py` (11,502 runs; runs the large grid already made are reused). `gpu_c8_gates.yaml` runs the two newest models in D1 and D2 first, under the same run names |
| Charts | `python scripts/plot_benchmark.py outputs` |

FedAvg and Multi-Krum have no FedHEP server, so they cannot run ALIE, Min-Max or the stealth backdoor ("n/a" in the charts). Every FedHEP-family run also records accuracy and backdoor success with the depth-exit guard.

**Which runs fill a cell.** Any GPU-server run whose configuration is the cell's, whatever its name, so the large grid and earlier GPU studies already fill part of the table. Runs from the Mac and the Acer never count: machines are not mixed (chapter 1.3). An empty cell has not been run yet.

**How to read the charts.**
- **Overview (one per setting):** every model (rows) under every condition (columns), in four panels: accuracy; backdoor success; detection F1, the mean over rounds of the F1 of the clients left out each round (1 = exactly the attackers, every round). Left out means isolated for FedHEP+ and FedHEP-Guard, but only skipped for that round for Bulyan, FLAME, FLTrust and Multi-Krum, and, for every model, dropped for sending an update with non-finite numbers (this is why Median and Trimmed mean, which never leave anyone out, can have an F1 above 0); and honest clients isolated per run. Grey cells are not run yet; "n/a" means the model cannot run that attack. Row colours are model families; FedHEP-Guard is in bold.
- **Detail (one per setting and condition, once a model has run it):** accuracy (± one standard deviation); backdoor success with seeds as dots; attackers isolated by round 30, with detection F1 as a diamond; honest clients isolated per run, or, for rules that never isolate, the share of honest clients they leave out per round; and two round-by-round maps, honest clients isolated so far and the share of attackers isolated so far, so it shows **when** each model acts.

[//]: # (BENCH:index START)
#### D1: 15 clients, α = 0.5, 3 attackers (`gg15_a05`)

410 of 426 cells run; 990 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, D1](../figures/bench/overview_gg15_a05.png)

![No attack, D1](../figures/bench/gg15_a05__clean.png)

![Sign flip, D1](../figures/bench/gg15_a05__sign.png)

![Label flip, D1](../figures/bench/gg15_a05__label.png)

![Random noise, D1](../figures/bench/gg15_a05__noise.png)

![ALIE, D1](../figures/bench/gg15_a05__alie.png)

![Min-Max, D1](../figures/bench/gg15_a05__minmax.png)

![Backdoor, D1](../figures/bench/gg15_a05__backdoor.png)

![Stealth backdoor, D1](../figures/bench/gg15_a05__stealth.png)

#### D2: 20 clients, α = 0.2, 4 attackers (`gg20_a02`)

410 of 426 cells run; 1230 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, D2](../figures/bench/overview_gg20_a02.png)

![No attack, D2](../figures/bench/gg20_a02__clean.png)

![Sign flip, D2](../figures/bench/gg20_a02__sign.png)

![Label flip, D2](../figures/bench/gg20_a02__label.png)

![Random noise, D2](../figures/bench/gg20_a02__noise.png)

![ALIE, D2](../figures/bench/gg20_a02__alie.png)

![Min-Max, D2](../figures/bench/gg20_a02__minmax.png)

![Backdoor, D2](../figures/bench/gg20_a02__backdoor.png)

![Stealth backdoor, D2](../figures/bench/gg20_a02__stealth.png)

#### 30 clients, α = 0.5, 6 attackers (`gg30_a05`)

410 of 426 cells run; 542 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, 30 clients, α 0.5](../figures/bench/overview_gg30_a05.png)

Detail charts: [No attack](../figures/bench/gg30_a05__clean.png) · [Sign flip](../figures/bench/gg30_a05__sign.png) · [Label flip](../figures/bench/gg30_a05__label.png) · [Random noise](../figures/bench/gg30_a05__noise.png) · [ALIE](../figures/bench/gg30_a05__alie.png) · [Min-Max](../figures/bench/gg30_a05__minmax.png) · [Backdoor](../figures/bench/gg30_a05__backdoor.png) · [Stealth backdoor](../figures/bench/gg30_a05__stealth.png)

#### 15 clients, α = 0.1, 3 attackers (`gg15_a01`)

51 of 426 cells run; 151 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, 15 clients, α 0.1](../figures/bench/overview_gg15_a01.png)

Detail charts: [No attack](../figures/bench/gg15_a01__clean.png) · [Sign flip](../figures/bench/gg15_a01__sign.png) · [Label flip](../figures/bench/gg15_a01__label.png) · [Random noise](../figures/bench/gg15_a01__noise.png) · [ALIE](../figures/bench/gg15_a01__alie.png) · [Min-Max](../figures/bench/gg15_a01__minmax.png) · [Backdoor](../figures/bench/gg15_a01__backdoor.png)

#### 30 clients, α = 0.1, 6 attackers (`gg30_a01`)

50 of 426 cells run; 150 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, 30 clients, α 0.1](../figures/bench/overview_gg30_a01.png)

Detail charts: [No attack](../figures/bench/gg30_a01__clean.png) · [Sign flip](../figures/bench/gg30_a01__sign.png) · [Label flip](../figures/bench/gg30_a01__label.png) · [Random noise](../figures/bench/gg30_a01__noise.png) · [ALIE](../figures/bench/gg30_a01__alie.png) · [Min-Max](../figures/bench/gg30_a01__minmax.png) · [Backdoor](../figures/bench/gg30_a01__backdoor.png)

#### 50 clients, α = 0.1, 10 attackers (`gg50_a01`)

50 of 426 cells run; 118 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, 50 clients, α 0.1](../figures/bench/overview_gg50_a01.png)

Detail charts: [No attack](../figures/bench/gg50_a01__clean.png) · [Sign flip](../figures/bench/gg50_a01__sign.png) · [Label flip](../figures/bench/gg50_a01__label.png) · [Random noise](../figures/bench/gg50_a01__noise.png) · [ALIE](../figures/bench/gg50_a01__alie.png) · [Min-Max](../figures/bench/gg50_a01__minmax.png) · [Backdoor](../figures/bench/gg50_a01__backdoor.png)

#### 50 clients, α = 0.5, 10 attackers (`gg50_a05`)

18 of 426 cells run; 46 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, 50 clients, α 0.5](../figures/bench/overview_gg50_a05.png)

Detail charts: [No attack](../figures/bench/gg50_a05__clean.png) · [Sign flip](../figures/bench/gg50_a05__sign.png) · [Label flip](../figures/bench/gg50_a05__label.png) · [Random noise](../figures/bench/gg50_a05__noise.png) · [Backdoor](../figures/bench/gg50_a05__backdoor.png)

#### 20 clients, α = 0.5, 4 attackers (`gg20_a05`)

10 of 426 cells run; 30 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, 20 clients, α 0.5](../figures/bench/overview_gg20_a05.png)

Detail charts: [No attack](../figures/bench/gg20_a05__clean.png) · [Sign flip](../figures/bench/gg20_a05__sign.png) · [Label flip](../figures/bench/gg20_a05__label.png) · [Random noise](../figures/bench/gg20_a05__noise.png) · [Backdoor](../figures/bench/gg20_a05__backdoor.png)

#### 15 clients, α = 0.05, 3 attackers (the paper's setting) (`gg15_a005`)

10 of 426 cells run; 30 of 1278 runs (3 seeds per cell; a cell with fewer seeds is preliminary).

![Benchmark overview, 15 clients, α 0.05 (paper)](../figures/bench/overview_gg15_a005.png)

Detail charts: [No attack](../figures/bench/gg15_a005__clean.png) · [Sign flip](../figures/bench/gg15_a005__sign.png) · [Label flip](../figures/bench/gg15_a005__label.png) · [Random noise](../figures/bench/gg15_a005__noise.png) · [Backdoor](../figures/bench/gg15_a005__backdoor.png)

[//]: # (BENCH:index END)

**Results so far (2026-11-26: 3,287 of 11,502 runs).** **D2 is complete for the 52 models queued first** (410 cells, 3 seeds each). D1 has 990 of their 1,230 runs (every cell, 170 with 3 seeds). 30 clients, α 0.5, has every cell but mostly one seed: its numbers below are marked preliminary. The other settings hold the large grid's runs and FedAvg and Multi-Krum. The two models added on 2026-11-26 (gate 6, high side 20; chapter 8.2) have no runs yet. All numbers are from the GPU server.

**D2 (complete).**
- **Accuracy separates the models little at D2.** Every FedHEP-server model with a screen stays between 51 and 55% under every attack, except random noise against R3 with gate 8 or 12 (41.6, 41.2%) and R3 with a rewind (49.6%). ALIE and Min-Max do almost no harm here even to undefended FedHEP (55.0 and 52.4%, against 54.5% with no attack); sign flip (43.0%) and random noise (11.2%) do. FLAME pays 6–8 points everywhere (46.0% with no attack), Multi-Krum trains on too few clients (17.6%), FedAvg collapses (random noise 0.7%).
- **The backdoor can be stopped at D2** (open question 4 below). FLAME holds it on every seed (plain 0 / 1 / 0%, stealth 1 / 3 / 0%) at its accuracy cost; Bulyan on two (0 / 12 / 0%). **FedHEP-Guard holds it on two seeds of three** (plain 39 / 1 / 1%, stealth 41 / 1 / 0%, seeds 42 / 123 / 7); its screen without the rewind on one or two (40 / 1 / 47%, 41 / 1 / 1%). **FedHEP+ with the whole-update signal and no regrouping** also holds it on two (0 / 18 / 0%, 0 / 19 / 0%). Every other FedHEP model is at 30–48%.
- **Why seed 42 gets through FedHEP-Guard.** The four attackers are the four lowest clients on the whole-update signal in rounds 2, 5 and 6 (z between −2 and −7, the lowest honest client −1.9). But the gap gate opens only in round 2: the gap ratio is 3.5–7.5 in the other rounds, below 8, because four attackers spread out make no single large gap. A closed gate gives no evidence, so the streak of 3 never forms; one attacker is isolated in round 9. FedHEP+ with the whole-update signal has no gate and isolates them in rounds 3–4. The offline replay (8.2) tests a lower gate on the low side.
- **The rewind costs 0.8–1.8 points in every condition** (FedHEP-Guard against its screen, 3 seeds: no attack 52.2 against 53.8). It fires whenever someone is isolated, honest or not: 1–4 times per run with no attacker at all. Each time it resets the trunk and root head to before the isolated client was first suspected, so honest isolations cost training rounds. At 30 clients (one seed, preliminary) it fired 8 times with no attack, and accuracy fell to 36.8% against 43.3% for the screen on the same seed.
- **Honest clients isolated per run, no attack:** FedHEP-Guard 5.0, its screen 3.7, FedHEP+ 8.7, FedHEP+ whole update 3.0, R3 1.7.

**D1 (2–3 seeds per cell).**
- FedHEP-Guard: plain backdoor 0.0 / 0.0% (2 seeds); **stealth backdoor 1% on seed 42 but 81% on seed 7**, where neither it nor its screen isolates any attacker. FedHEP+ with the whole update: 0 / 0% and 0 / 1%. FedHEP+: 18 / 5 / 42%, stealth 70 / 76%.
- FedHEP-Guard catches every ALIE and Min-Max attacker; FedHEP+ with the whole update catches none. At D1 this costs little (ALIE does almost no harm). At 30 clients (preliminary) Min-Max takes FedHEP+ with the whole update to 35.1%, against 41.9% for the union screen.
- **Correction to the 2026-11-25 note.** It said the rewind did not fire in the D1 no-attack runs. It fired 0, 4 and 2 times (seeds 7, 42, 123); accuracy was 47.1% against 47.4% for the screen.

![Benchmark overview, D2](../figures/bench/overview_gg20_a02.png)

**What the benchmark will decide.** Set before the results, so the results can refute them (in the style of `PREDICTIONS.md`):
1. **FedHEP-Guard against FedHEP+.** FedHEP-Guard wins under ALIE, Min-Max and both backdoors at D1. At D2 it loses less than 1 point with no attack, where FedHEP+ loses 2.
2. **FedHEP-Guard against FLAME and Bulyan.** It matches their backdoor protection (under 2%) at D1, with 3–6 points more accuracy with no attack.
3. **Honest isolations.** FedHEP-Guard isolates more honest clients than FedHEP+ at 30 and 50 clients (chapter 6.13); the benchmark adds the full model and its Phase 8 variant.
4. **The open question:** whether any model stops the backdoor at D2. None did on the Mac or in the GPU's D2 ladder.

**Outcomes so far (2026-11-26).**
1. **Half refuted.** At D1 FedHEP-Guard wins against both backdoors on success (plain 0 / 0% against 18 / 5 / 42%; stealth on one seed of two), but under ALIE and Min-Max it is 0.8–1.6 points less accurate than FedHEP+, although it catches every attacker and FedHEP+ none: these attacks do little harm at D1. At D2 it loses 2.3 points with no attack against FedHEP, FedHEP+ 1.8 (refuted).
2. **Half held.** Plain backdoor under 2% at D1 (0.0%), but the stealth backdoor reaches 81% on seed 7. With no attack it is 4.0 points more accurate than FLAME and 2.1 more than Bulyan.
3. **Held so far** at 30 clients, α 0.5 (preliminary): FedHEP-Guard 18 honest isolations (one seed), its screen 14.7, FedHEP+ 4.3. The 50-client cells are not run yet.
4. **Answered: yes, on some seeds.** FLAME on every seed (at 46% accuracy), FedHEP-Guard and FedHEP+ with the whole update on two of three.

## 6.2 Reproducing the published models (Phases 1 and 1c)

### FedHEP and the paper's baselines

We ported the authors' code function by function and checked it with 100 parity tests that call their functions directly. The tests cover model and initial weights, skew score, head weights, grouping, parent momentum, one full local round to $10^{-5}$, and the test-time ensemble. Planted one-line errors break the tests.

![Phase 1: our numbers against the authors' code and the paper](../figures/phase1_parity.png)

| Run (Mac, 3 seeds) | Ours | Authors' code (same machine) | Paper |
|---|---|---|---|
| FedAvg, IID | 39.04 ± 1.25 | 38.04 ± 0.94 | 41.46 ± 0.36 |
| FedAvg, α = 0.05 | 29.44 ± 1.33 | 29.07 ± 2.05 | 36.29 |
| **FedHEP, α = 0.05** | **65.51 ± 0.80** | 65.68 ± 0.57 | 65.49 ± 0.61 |
| FedHEP, α = 0.05, bottom-10% | 49.57 ± 1.91 | 50.23 ± 1.50 | 53.92 ± 0.63 |

**Verdict: reproduced.** Every number is within 2 standard errors of the authors' code. FedHEP also matches the paper. The paper's FedAvg numbers are higher than both implementations, a gap between the authors' released code and their paper.

The five baselines the paper compares against (FedProx, SCAFFOLD, Ditto, FedRep, Multi-Krum) were ported with 43 further parity tests. FedAvg and Multi-Krum are used in the benchmark ([chapter 6.1](06_results.md)).


### Defended FedHEP

149 parity tests match the authors' SCCF, trust weights, attacker choice and attack uploads. On CIFAR-100 (D1 data, the paper's attack rule with probability $q$, Mac, 3 seeds):

| Setting | Ours | Paper |
|---|---|---|
| No attack | 48.39 ± 0.72 | 47.88 ± 0.46 |
| Sign flip, $q = 0.2$ | 40.75 ± 4.65 | 40.22 ± 3.24 |
| Random noise, $q = 0.3$ | 44.63 ± 2.07 | 44.53 ± 1.86 |

**Verdict: reproduced**, all within 2 standard errors of the paper.


## 6.3 The grouping signal is mostly a counter

FedHEP decides groups, and whether a parent copies Global, from the cosine between **full** flattened updates. That vector includes BatchNorm's batch counter, which grows by the number of local steps (about 105 per layer per round) and has the same sign for every client. The counter dominates:

| Part of the update | Norm |
|---|---|
| weights | 5.2–6.3 |
| BatchNorm running statistics | 24–25 |
| batch counters | 99–113 |

On CIFAR-100, members' agreement was 0.995–0.999 in every round, so **parents always copied Global and never averaged their members**. The middle tier therefore adds no group-specific information, as run in the authors' code. We kept the faithful port for the reproduction. FedHEP+ later groups on weights only (chapter 4).

![BatchNorm counters dominate the grouping cosine](../figures/batchnorm_counter_dominance.png)

![Parent agreement per round](../figures/fedhep_parent_agreement.png)

## 6.4 Defended FedHEP under attack (Phase 2)

### The fair benchmark

**Question:** how well do the published methods resist attacks when all are measured the same way, including a backdoor the paper did not test?

**Setup:** D1, exactly 3 attackers (the same for every method), 30 rounds, Mac, 3 seeds, 72 runs (`phase2_core.yaml`). Detection is scored every round.

![Phase 2: accuracy under attack](../figures/phase2_accuracy.png)

| Model | No attack | Sign flip | Label flip | Random noise | Backdoor (acc.) | Backdoor success |
|---|---|---|---|---|---|---|
| FedAvg | 38.1 ± 0.8 | 8.5 ± 12.9 | 32.9 ± 1.1 | 1.2 ± 0.3 | 36.2 ± 1.6 | 88.0% |
| Multi-Krum | 23.4 ± 1.6 | 24.6 ± 0.8 | 24.0 ± 1.0 | 24.5 ± 0.9 | 24.8 ± 1.4 | 0.6% |
| FedHEP | 47.8 ± 0.4 | 33.7 ± 2.5 | 43.6 ± 0.4 | 5.4 ± 0.3 | 46.2 ± 0.3 | 79.1% |
| **Defended FedHEP** | **48.4 ± 0.7** | **38.1 ± 1.0** | 43.2 ± 0.4 | **45.8 ± 0.7** | **46.4 ± 0.4** | 78.7% |

![Phase 2: backdoor success](../figures/phase2_backdoor_asr.png)

![Phase 2: detection](../figures/phase2_detection.png)


### What improved, and what still fails

**Improved over FedHEP:**
- **The most accurate method under every attack.** Its biggest win is random noise (45.8% against 5.4%), where the norm bound and the BatchNorm median do the work.
- **Under sign flip it gains 4.4 points** over FedHEP (38.1% against 33.7%).

**Still fails:**
1. **It never removes an attacker.** No client was isolated in any of 24 runs (9 in Phase 1c, 15 attacked runs in Phase 2). Attackers keep a 0.5–0.8 share of the weight every round. A ledger score settles near the average of $\min(1, n t_i)$, so it never reaches 0.25, and sign flip still costs 10.3 points.
2. **It does nothing against the backdoor**: 78.7% success, the same as no defense. Backdoor attackers even look **more** trustworthy than honest clients (AUROC 0.36), because half their data is normal and their root-head updates point with the consensus.
3. **The defense fades during training.** The temperature is fixed while all cosines shrink. When the mean direction gets short ($\lVert r\rVert < 0.2$), [d] is skipped and everyone gets equal weight; this happens in 7–73% of runs in each of the last five rounds. In those rounds the ledger also gives every client, attackers included, full credit, which pushes every score back toward 1. The trust scores often rank attackers correctly (AUROC 0.97 for sign flip) but never act on that ranking.

![Why the defense fades](../figures/defended_failure_diagnosis.png)

**Other models:**
- Multi-Krum blocks every attack only by keeping 3 of 15 clients every round, which leaves accuracy near 24%.
- FedAvg collapses under sign flip and noise.


## 6.5 FedHEP+ results (Phases 3 and 3b)

D1, 3 attackers, Mac, 3 seeds. The comparison rows are the Phase 2 runs with the same seeds and attackers.

![FedHEP+ against Defended FedHEP and the Phase 3b variants, three seeds](../figures/phase3b_confirm.png)

**Reading the Phase 3b charts.** Each bar is one model. The chart labels name FedHEP+'s parts:
- grey "Defended FedHEP" is the baseline;
- blue "z + median + regroup" is **FedHEP+**: [2] robust z, [1] median reference, [5] regrouping;
- the orange, green and yellow bars are two variants tested in the same runs and rejected: readmission and the cap (chapter 7.1).

The detection panel shows the F1 of the clients left out each round: 1 means exactly the attackers.

| Model | No attack | Sign flip | Label flip | Random noise | Backdoor (acc.) | Backdoor success |
|---|---|---|---|---|---|---|
| Defended FedHEP | 48.39 ± 0.72 | 38.08 ± 1.01 | 43.23 ± 0.40 | 45.80 ± 0.70 | 46.42 ± 0.37 | 78.7% |
| **FedHEP+** | 47.92 ± 0.60 | **46.34 ± 0.87** | **46.34 ± 0.87** | 45.41 ± 1.15 | 46.26 ± 0.56 | **20.8%** |
| Change, paired by seed | −0.47 | **+8.26** | **+3.11** | −0.38 | −0.16 | **−57.9 points** |

- The gains under sign flip, label flip and backdoor hold **on every seed**, by far more than the seed spread. This is the project's first confirmed improvement.
- Sign and label flippers are isolated in round 1. That is why both rows are identical: from round 2 on, the same 12 honest clients train alone.
- The median reference is what exposes the backdoor: with the mean reference, backdoor success stays near 80% (seed 42: 33% against 80%).

![FedHEP+ backdoor](../figures/phase3b_backdoor.png)

### The false-isolation cascade

![The false-isolation cascade](../figures/phase3_cascade.png)

**The cascade.** In FedHEP+ honest clients are isolated one after another as training goes on. The first explanation, a yardstick that shrinks with each removal, led to the variants above, and 7.1c refuted it. The actual cause, a scale floor that fades mid-training, is in 6.6a. Regrouping and scoring only the clients still kept limited the cascade but did not remove it.


## 6.6 Where FedHEP+ fails: the follow-up diagnostics

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

## 6.7 What FedHEP+ improved, and what still fails

**Improved over Defended FedHEP (D1):** sign flip +8.3, label flip +3.1, backdoor success 79% → 21%, and the bottom-10% under sign flip 31.2 → 40.8. All of this costs 0.5 points with no attack.

**Still fails:**
1. **Honest but unusual clients look like attackers**: about 9 of 20 isolated at α = 0.2.
2. **Adaptive attackers are not seen**, because the screen is one-sided.
3. **A backdoor that got in stays**, even after its attackers are removed.
4. **More colluders pull the median**: with 5 backdoor attackers, success rises to 56% (chapter 7.3).

The lesson: "far from the typical update" catches crude attacks well, even many of them. But honest clients with unusual data are also far from typical, and clever attackers stay close to it. FedHEP-Guard ([chapter 5](05_fedhep_guard.md)) follows this lesson; the ideas that did not fix it are in [chapter 7](07_ideas_that_did_not_work.md).
## 6.8 FedHEP-Guard: what it fixed from FedHEP+

**FedHEP-Guard itself has been run only 6 times so far:** on the Acer, over 100 rounds, against the plain and the stealth backdoor (`x9__*__r3wc_tr`, 6.10). Every other number in the FedHEP-Guard column below comes from runs of **parts** of the model, marked in each cell:

**Update 2026-11-26: the full model now has benchmark runs** (6.1, GPU, 30 rounds): every condition at D2 (3 seeds) and D1 (2–3 seeds). **[full]** at D2: plain backdoor 39 / 1 / 1%, stealth 41 / 1 / 0%; every sign-flip, ALIE and Min-Max attacker isolated; 5.0 honest isolations per run with no attack (its screen 3.7, FedHEP+ 8.7); 0.8–1.8 points below its screen in every condition, because the rewind also fires after honest isolations. At D1: plain backdoor 0.0 / 0.0%, stealth 1 / 81% (seed 7: no attacker caught). The table below keeps its marks until the benchmark is complete.

| Mark | Model actually run | What it lacks | Runs |
|---|---|---|---|
| **[full]** | FedHEP-Guard: union screen + rewind (`r3wc_tr`) | nothing | Acer, 100 rounds, both backdoors |
| **[screen]** | FedHEP-Guard without the rewind (`r3wc`) | the rewind [6] | GPU grid, Mac stealth study, Acer |
| **[signal]** | R3 with one of the union's two signals (`r3w` or `r3c`) | the other signal, the rewind | GPU D2 ladder |
| **[w+rewind]** | R3, whole-update signal + rewind (`r3w_tr`) | the critical-layer signal | GPU 100 rounds |

**How far the parts stand in for the model.** The rewind changes nothing until the first isolation, because it only acts when someone is isolated. After that it resets the trunk and root head, so the run takes a different path. On the Acer, with the plain backdoor, the full model isolated 18 honest clients in 3 runs against 8 for the screen alone, and was 1.8 points less accurate. With the stealth backdoor it was 10 against 16 and 1.7 points more accurate. So **[screen]** numbers are a close guide for which attackers are caught, and a rough one for honest isolations and accuracy. The full model under every condition and setting comes with the benchmark (6.1).

FedHEP+ left three failures ([chapter 6.7](06_results.md)).

| | FedHEP+ | FedHEP-Guard or its parts (see marks) | Verdict |
|---|---|---|---|
| **1. Honest but unusual clients isolated** (per run, no attack, GPU) | 15 clients: 2.7 (α 0.5), 7.7 (α 0.1). 30 clients: 4.3, 2.3. 50 clients, α 0.1: 1.0. D2: 8.7 | **[screen]** 15 clients: 4.3, 1.7. 30 clients: 14.7, 7.0. 50 clients, α 0.1: 22.0. **[signal]** D2: 1.7 (whole update), 2.3 (critical layers). **[full]** no run without an attack yet; with a backdoor, 3–6 per run over 100 rounds (Acer) | **Not fixed** by the screen: fewer than FedHEP+ only at 15 clients with α = 0.1; in the other four grid settings it isolates more, and more with more clients (22 of 50 at 50 clients). About half come as whole parent groups (8.2). The root-head R3 does fix it (0.7–6.0), but lets the backdoor in. Not yet measured for the full model |
| **2. Adaptive attackers not seen** | ALIE and Min-Max: 0 of 9 attackers isolated each (GPU, D1); 0 of 12 each at D2 (GPU); stealth backdoor 56–75% (Mac) | **[screen]** ALIE and Min-Max 9 of 9 each (GPU, D1); stealth backdoor 0.2–0.5% (Mac, 30 rounds). **[signal]** D2: each signal 12 of 12 under ALIE and Min-Max (GPU). **[full]** stealth backdoor 0.4 / 0.4 / 33.8% over 100 rounds (Acer) | **Fixed by the screen.** The full model has faced only the stealth backdoor, and one seed got through after a late catch. Cost under ALIE: 45.3 against FedHEP+'s 47.8 (GPU, [screen]) |
| **3. Backdoors that persist** | all attackers isolated, yet 18 / 5 / 42% stays (GPU, 30 rounds); 21% (Mac); 13.9% after 100 rounds (GPU) | **[full]** 0.3% on every seed over 100 rounds (Acer). **[screen]** 0.5–0.6% (GPU), 0.8% (Mac), 30 rounds. **[w+rewind]** 2.5% over 100 rounds (GPU) | **Fixed at α = 0.5 when the attackers are caught early** (rounds 3–6), for the full model on the Acer and the screen on every machine. **Not fixed** by the screen at α = 0.1 (15–31%) or with a third of the clients attacking (12% at 15 clients, 5% at 30), or by the full model after a late catch (one stealth seed at 34%). At D2 untested; there every rung except the whole update leaves 38–46% |
| Accuracy with no attack | 48.48 (GPU) | **[screen]** 47.36 (GPU) | about 1 point lower |
| Worst accuracy under sign flip, ALIE and Min-Max (GPU grid: 15 / 30 clients × α 0.5 / 0.1) | 34.7 / 54.0 / 26.6 / 50.7 | **[screen]** 43.3 / 57.8 / 40.2 / 54.4 | **better in all four** |

**In short:** FedHEP-Guard's screen is an improvement on FedHEP+ against attackers, not yet for honest clients; the full model has been tested only against backdoors, on one machine.
- The screen fixes failure 2 in every setting tested (at D2 through each of its two signals), and with the rewind failure 3 at α = 0.5. It holds up best when the model is poisoned.
- The screen does not fix failure 1. With no attack it isolates 22 of 50 honest clients per run at 50 clients, where FedHEP+ isolates 1. An earlier version of this scorecard called failure 1 "fixed under strong skew", from the 15-client setting alone.
- What the full model does without an attack, under ALIE, Min-Max, sign flip, at D2 and at scale is not yet known; the benchmark (6.1) runs it everywhere.

![Honest isolations by setting and signal](../figures/phase6_honest_isolation_summary.png)

![FedHEP-Guard over 100 rounds](../figures/phase6_guard_durability.png)

## 6.9 The ladder (Mac, D1 and D2; GPU server, D2)

D1: 15 clients, α = 0.5, 3 attackers. D2: 20 clients, α = 0.2, 4 attackers. Mac, 3 seeds, 30 rounds. R2 = L0 + two-sided screen + patience; R3 = R2 + gap gate. All rungs here use the root-head signal.

**D1:**

| Model | No attack | Sign flip | ALIE | Min-Max | Backdoor success | Honest isolated per run (clean / sign / ALIE / Min-Max) |
|---|---|---|---|---|---|---|
| Defended FedHEP | **48.39** | 38.08 | 47.49 | 45.19 | 78.7% | 0 / 0 / 0 / 0 |
| FedHEP+ | 47.92 | **46.34** | 47.55 | 44.87 | **20.8%** | 3.0 / 1.7 / 2.0 / 1.0 |
| L0 | 47.56 | 41.65 | **47.56** | 45.47 | 62.5% | 0 / 0 / 0 / 0 |
| R2 | 46.99 | 46.00 | 45.65 | 45.96 | 22.6% | 4.3 / 1.7 / 3.0 / 3.7 |
| R3, gate 8 | 47.36 | 46.14 | 45.74 | 46.02 | 51.4% | 1.0 / 0.7 / 0.7 / 0.7 |
| R3, gate 12 | 47.31 | 46.20 | 45.87 | **46.07** | 61.6% | **0.0 / 0.7 / 0.0 / 0.3** |

![Ladder at D1](../figures/phase6_d1.png)

**D2 (held out):**

| Model | No attack | Sign flip | Min-Max | Backdoor success | Honest isolated, 3 seeds (clean / sign / Min-Max / backdoor) |
|---|---|---|---|---|---|
| FedHEP | **55.59** | 42.51 | 52.10 | 46.9% | 0 / 0 / 0 / 0 |
| Defended FedHEP | 54.70 | 50.62 | **54.16** | 44.3% | 0 / 0 / 0 / 0 |
| FedHEP+ | 52.68 | 52.15 | 53.55 | 43.3% | 27 / 21 / 6 / 18 |
| L0 | 54.23 | 50.52 | 52.96 | **37.7%** | 0 / 0 / 0 / 0 |
| R2 | 52.13 | 51.86 | 52.62 | 44.1% | 38 / 29 / 29 / 33 |
| R3, gate 8 | 54.72 | **54.26** | 53.34 | 45.3% | 3 / 6 / 8 / 7 |
| R3, gate 12 | **54.73** | 54.19 | 52.51 | 45.3% | **2 / 1 / 0 / 6** |

![Ladder at D2](../figures/phase6_d2.png)

**What the ladder shows:**
- **The gap gate solves FedHEP+'s held-out failure.** At D2, honest isolations with no attack fall from 27 (FedHEP+) and 38 (R2) to 2–3. R3 is the best model under sign flip (+3.6 over Defended FedHEP) and within 1 point of undefended FedHEP with no attack.
- **The two-sided screen catches every ALIE and Min-Max attacker**, where the one-sided screen caught none.
- **But the root-head signal with the gate lets the backdoor in** (51–62% at D1). That led to the signal search ([chapter 7.4](07_ideas_that_did_not_work.md)), the union screen [1] and the rewind [6].

**D2 on the GPU server (`gpu_c4_d2`).** The same ladder, plus the two signals that make up FedHEP-Guard's union screen, each on its own: the whole update and the critical layers. The union screen itself and the full model were not in this batch; they come with the benchmark (chapter 6.1). CUDA, 3 seeds, 30 rounds.

| Model | No attack | Sign flip | ALIE | Min-Max | Backdoor success, seeds 123 / 42 / 7 | Honest isolated per run, no attack | Attackers caught, ALIE / Min-Max (of 12 each) |
|---|---|---|---|---|---|---|---|
| FedHEP | **54.48** | 42.99 | 55.09 | 52.42 | 47 / 46 / 46% | 0 | 0 / 0 |
| Defended FedHEP | 53.81 | 50.28 | **55.37** | 54.39 | 48 / 42 / 41% | 0 | 0 / 4 |
| FedHEP+ | 52.73 | 52.04 | 55.18 | 53.97 | 46 / 38 / 44% | 8.7 | 0 / 0 |
| L0 | 53.77 | 49.71 | 55.18 | 53.32 | 44 / 36 / 35% | 0 | 0 / 0 |
| R3 gate 8, root head | 54.20 | 53.41 | 54.36 | 53.45 | 47 / 41 / 47% | 1.7 | 9 / 12 |
| R3 gate 12, root head | 54.20 | **53.75** | 55.09 | 52.50 | 47 / 41 / 47% | 1.0 | 2 / 8 |
| R3, whole update | 53.93 | 53.66 | 54.27 | **54.44** | 47 / 40 / **4.7%** | 1.7 | **12 / 12** |
| R3, critical layers | 54.13 | 53.57 | 53.85 | 54.11 | 47 / 41 / 47% | 2.3 | **12 / 12** |

![Ladder at D2, GPU server](../figures/phase6_gpu_d2.png)

- **The GPU repeats the Mac's D2 result for honest clients.** The gap gate brings FedHEP+'s 8.7 honest isolations per run down to 1.0–1.7, and the two whole-model signals stay at 1.7–2.3. At D2 the whole-model signals do not bring back the honest isolations they cause at 30 and 50 clients (6.13).
- **Adaptive attackers are caught at D2 too.** Both signals of the union screen catch all 12 ALIE and all 12 Min-Max attackers. FedHEP+ catches none.
- **The backdoor is not solved at D2.** Every model stays at 35–48%, except the whole update on one seed (4.7%). At α = 0.2 with 4 attackers, the backdoor attackers look like honest clients to every signal tested. Whether the union screen does better at D2 is the open question for the benchmark.
- **Accuracy hardly moves under attack at D2.** ALIE leaves every model at 54–55%, within 0.6 points of its no-attack accuracy, because strong skew makes the personal heads carry most of the prediction. Only sign flip hurts undefended FedHEP (−11.5 points).

## 6.10 Recovering from a backdoor (Mac, Acer and GPU server)

**30 rounds, attackers plant the backdoor in rounds 1–10, then train honestly (Mac, D1).** Backdoor success at round 30, seeds 123 / 42 / 7:

| Variant | Backdoor success | Attackers caught (of 9) | Accuracy |
|---|---|---|---|
| R3, root head (with or without rewind) | 19 / 3 / 47% | 1 | 46.7–48.1 |
| R3, whole update | 0.5 / 0.5 / 0.9% | 9 (rounds 3–6) | 46.10 |
| R3, whole update + trunk-and-root rewind | 0.2 / 0.4 / 0.3% | 9 | 45.83 |

**100 rounds, attackers plant the backdoor until round 50, then train honestly (Acer, D1).** Backdoor success at round 100:

| Model | Seeds 123 / 42 / 7 | Mean | With exit guard | Accuracy | Honest isolated (3 seeds) |
|---|---|---|---|---|---|
| FedHEP | 69 / 74 / 75 | 72.4 | 38.6 | 43.55 | 0 |
| L0 | 64 / 70 / 71 | 68.3 | 37.6 | 44.38 | 0 |
| FedHEP+ | 2 / 5 / 2 | 3.1 | 2.8 | 43.68 | 24 |
| R3, root head | 23 / 71 / 21 | 38.4 | 22.9 | 43.63 | 1 |
| R3, root head + rewind | 8 / 28 / 15 | 17.1 | 11.2 | 42.54 | 2 |
| R3, whole update | 1 / 17 / 1 | 6.2 | 4.5 | 42.19 | 18 |
| **R3, whole update + rewind** | **0 / 1 / 0** | **0.6** | 0.6 | 41.81 | 12 |

![Rewind, per seed](../figures/phase6_rewind.png)

![Backdoor over 100 rounds](../figures/phase6_durability.png)

- **A backdoor nobody catches stays at 64–75%** for 50 rounds after the attack ends.
- **The rewind matters when the catch is late.** On seed 42 the whole-update screen isolated the attackers after part of the backdoor had reached the trunk, and it grew to 17% after they stopped. With the rewind, the same seed stayed at 1%. It costs 0.4 points on average.
- **Why seed 42 is different.** Seed 42 draws attackers 0, 9 and 11. It is the hard seed for every signal and machine: 70–71% for the root head and the whole update on the GPU too (chapter 7.4).
  - **The attackers stay hidden for 15 rounds.** With the root head, their z stays between −2.2 and +1.1 for the first 15 rounds, and no round's gap ratio reaches 8 before round 16. On seeds 123 and 7 their z falls below −2 by rounds 6–10, and they are isolated in rounds 13–14. Their data sizes (540–705 images) are ordinary (honest clients: 470–873).
  - **When they are caught:** without the rewind, attackers 9 and 11 are isolated only in round 40 and attacker 0 never. By then the backdoor is at 70%, and nothing removes it after the attack stops: the flat 71% line.
  - **With the rewind:** this is a separate run, and XPU runs are not bit-for-bit repeatable, so the two runs part ways early. Here the attackers are caught in rounds 21–24 and the model is rewound to rounds 19–22. Success drops from 13% to 8%, but the rewind reaches back only as far as the attackers' evidence trail (at most 10 rounds). The poisoning from rounds 1–18 stays in, and it grows back to 28% by round 100, as a backdoor does once it is in (chapter 6.6e).
  - **The whole-update screen catches them earlier,** in rounds 12–17. With the rewind, seed 42 then stays at 1%.
- **Honest isolations are the price of the whole-update signal.** Over 100 rounds the root-head screens isolate 1–2 honest clients in 3 runs, the whole-update screens 12–18, or 4–6 per run. FedHEP-Guard's union screen contains the whole-update signal, so it pays the same price (6.13, 8.1). The gap gate removed FedHEP+'s honest isolations for the root-head signal (6.9). It has not done so for the signals that catch the backdoor.


**The same experiments on the GPU server (`gpu_c5_recovery`, `gpu_c6_durability`).** These repeat the Mac's recovery runs and the Acer's 100-round runs with the two single signals; the union screen was not in these batches.

*30 rounds, backdoor in every round (D1, 3 attackers).* Backdoor success at round 30, seeds 123 / 42 / 7:

| Signal | No rewind (`g1__`) | Rewind trunk | Rewind trunk + root head | Rewind everything |
|---|---|---|---|---|
| R3, root head | 32 / 70 / 68% | 21 / 64 / 51% | 18 / 63 / 47% | 18 / 63 / 47% |
| R3, whole update | 0.6 / 71 / 0.6% | 0.5 / 71 / 0.4% | 0.4 / 71 / 0.5% | 0.2 / 71 / 0.4% |

- **For the whole update the rewind changes nothing in 30 rounds.** It catches the attackers in rounds 3–4, before anything has to be undone, or not at all (seed 42).
- **For the root head the rewind lowers success from 57% to 42–45%** on average, but not below 15%: the attackers are caught in rounds 12–21, and part of the backdoor is older than the rewind's reach (prediction P8, not met here either).
- **With the attack stopped after round 10**, the whole update ends at 0.2 / 2.7 / 0.4%, and the root head at 27 / 2.7 / 56% with any rewind. On seed 42 the backdoor faded on its own, with nobody caught.

*100 rounds, attackers until round 50 (D1, 3 attackers).* Backdoor success at round 100:

| Model | Seeds 123 / 42 / 7 | Mean | With exit guard | Accuracy | Honest isolated (3 runs) | Attackers caught in round |
|---|---|---|---|---|---|---|
| FedHEP | 68 / 72 / 78 | 72.9 | 40.4 | 43.93 | 0 | never |
| L0 | 64 / 64 / 75 | 67.7 | 37.1 | 44.55 | 0 | never |
| FedHEP+ | 1.6 / 11 / 29 | 13.9 | 8.9 | **44.63** | 25 | 8–14 |
| R3, root head | 7 / 71 / 49 | 42.3 | 23.8 | 43.33 | 1 | 12–20; one never |
| R3, root head + rewind | 4 / 70 / 34 | 36.1 | 22.2 | 43.54 | 0 | 12–20; one never |
| R3, whole update | 0.5 / 52 / 0.7 | 17.8 | 9.2 | 42.87 | 15 | 3–4; seed 42: one in round 42, two never |
| **R3, whole update + rewind** | **0.6 / 6.5 / 0.5** | **2.5** | **2.2** | 42.19 | 11 | 3–4; seed 42: all in round 20 |

![Backdoor over 100 rounds, GPU server](../figures/phase6_gpu_durability.png)

![Rewind per seed, GPU server](../figures/phase6_gpu_rewind.png)

- **The GPU confirms the Acer's ranking.** A backdoor nobody catches stays at 64–78%. The whole update with the rewind is lowest (2.5%; 0.6% on the Acer), and the rewind matters most on the seed where the catch is late.
- **Seed 42 needs care.** GPU runs are repeatable, so the whole-update runs with and without rewind are identical until the first rewind, which an honest isolation triggered. After that they part ways: the rewind run caught all three attackers in round 20, the other run caught one, in round 42. The rewind changed the run's path; whether it reliably leads to an earlier catch cannot be told from one seed.
- **FedHEP+ holds the backdoor to 13.9% here, against 3.1% on the Acer.** It catches every attacker, but in rounds 8–14, later than the whole update, and pays 25 honest isolations in 3 runs.
- The union screen and the full FedHEP-Guard have not yet run over 100 rounds on the GPU (only on the Acer, below).

**FedHEP-Guard itself over 100 rounds (Acer, `acer_c9_final_durability`).** These are the first runs of the full model (union screen + trunk-and-root rewind), in the same setting: 3 attackers until round 50. Backdoor success at round 100, seeds 123 / 42 / 7:

| Model | Plain backdoor | Stealth backdoor | With exit guard (plain / stealth, mean) | Accuracy (plain / stealth) | Honest isolated (3 runs, plain / stealth) |
|---|---|---|---|---|---|
| FedHEP (no defense) | 67 / 75 / 77% | 67 / 73 / 77% | 41.7 / 37.6% | 43.06 / 43.51 | 0 / 0 |
| Union screen, no rewind | 0.3 / 0.4 / 0.6% | 0.5 / 1.6 / 0.4% | 0.4 / 0.8% | 43.47 / 42.30 | 8 / 16 |
| **FedHEP-Guard** | **0.3 / 0.3 / 0.3%** | 0.4 / 0.4 / **33.8%** | 0.3 / 6.3% | 41.66 / 43.96 | 18 / 10 |

![FedHEP-Guard over 100 rounds, per seed](../figures/phase6_guard_durability.png)

- **The plain backdoor is held on every seed for 100 rounds,** including seed 42, which beat the root-head and whole-update screens (above). The union screen isolates seed 42's attackers in rounds 3–4, against round 40 for the root head and rounds 12–17 for the whole update.
- **One stealth seed got through FedHEP-Guard.**
  - On seed 7 the attackers were isolated late, in rounds 15–18. The rewinds reached back only to rounds 15–16, so the poisoning from rounds 1–14 stayed in. It grew from 22% at round 50 to 33.8% at round 100 (17.9% with the exit guard).
  - In the run without rewind, the same seed's attackers were caught in round 6 and success stayed at 0.4%. XPU runs are not repeatable bit for bit, so the two runs part ways early.
  - In the FedHEP-Guard run, an honest pair was isolated in round 7 and the model rewound to round 5. Whether that delayed the catch cannot be told from one run.
  - **What it shows:** the rewind cannot undo poisoning older than its 10-round reach when the catch is late.
- **Accuracy.** With the plain backdoor, FedHEP-Guard is 1.8 points below the union screen without rewind and 18 honest isolations against 8. With the stealth backdoor it is 1.7 points above, with 10 against 16. These differences track the honest isolations more than the rewind: rewinds triggered by honest isolations discarded only 1–7 rounds per run.
- **Most of these honest isolations are whole parent groups.** Mass isolations of 4–5 honest clients happened in rounds 13–16; the cause is in chapter 8.2.
## 6.11 An adaptive backdoor (Mac, D1)

The stealth backdoor ([chapter 1.4](01_federated_learning.md)) hides inside the honest spread. Backdoor success at round 30, seeds 123 / 42 / 7:

| Model | Plain, 3 attackers | Stealth, 3 attackers | Plain, 5 attackers | Stealth, 5 attackers |
|---|---|---|---|---|
| FedHEP | 74 / 80 / 84 | 71 / 77 / 83 | 75 / 82 / 84 | 73 / 81 / 84 |
| Defended FedHEP | 74 / 80 / 82 | 73 / 80 / 81 | 74 / 81 / 82 | 73 / 81 / 82 |
| L0 | 54 / 66 / 68 | 61 / 70 / 75 | 69 / 77 / 79 | 69 / 79 / 80 |
| FedHEP+ | 5 / 29 / 29 | 56 / 75 / 65 | 40 / 64 / 65 | 71 / 79 / 80 |
| R3, whole update | 0.5 / 0.5 / 0.9 | 0.2 / **61** / 0.9 | 20 / 4.4 / 3.8 | 2.9 / 9.5 / 3.0 |
| **R3, union screen** | 0.5 / 0.5 / 1.4 | **0.2 / 0.5 / 0.4** | 20 / 4.0 / 0.7 | 2.9 / 9.5 / 1.7 |
| FLAME | 0.6 / 0.4 / 0.5 | 1.5 / 0.4 / 0.4 | 0.4 / 0.6 / 0.4 | 1.7 / 3.0 / 1.8 |

![Stealth backdoor](../figures/phase6_stealth.png)

- Hiding costs the attacker almost nothing: undefended, it still reaches 71–83%.
- It defeats single-view screens: FedHEP+, and the whole-update screen on one seed.
- **The union screen catches every stealth attacker on every seed.** The price is more honest isolations (14 over 3 runs, against 9) and 1.2 points against undefended FedHEP.
- **With 5 attackers (a third of the clients)** both R3 screens catch all 15 attackers, plain or stealth, but on one seed the plain backdoor gets in first (20%). FLAME holds under 1% at about 3 points of accuracy.

## 6.12 A wrong guess of the attacker count (Acer, D1)

The defender assumes $f_0$ = 1, 3 (true) or 6 while 3 attack. Accuracy, or backdoor success where stated:

| Attack | L0, $f_0$ = 1 / 3 / 6 | R3 union, $f_0$ = 1 / 3 / 6 |
|---|---|---|
| None | 48.10 / 47.43 / 47.56 | 46.07 / 46.82 / 45.64 |
| Sign flip | **35.73** / 40.73 / 41.89 | 44.35 / 45.28 / 43.77 |
| ALIE | 47.75 / 47.53 / 46.19 | 44.50 / 44.09 / 44.93 |
| Backdoor success | 70–78% / 60–69% / 58–62% | 0.5–0.6% / 0.4–0.5% / 0.4–1.5% |

![Mis-specified f](../figures/phase6_fmis.png)

The screen makes R3 almost indifferent to $f_0$: every attacker is caught under every attack and every guess. L0 depends on the guess under sign flip.

## 6.13 At scale (GPU server)

The large grid (`gpu_grid_n{15,30,50}`, 1,872 runs) repeats 8 models at 15, 30 and 50 clients, α = 0.5 and 0.1, with 10, 20 and 30% attackers, 3 seeds, 30 rounds. On 2026-11-23, 1,308 runs were finished: the 15- and 30-client parts complete, and the 50-client, α = 0.1 part with no attack and with 5 sign-flip attackers. The rest of the 50-client part is running. Its FedHEP-Guard column is the **screen without the rewind** (R3 + union screen); the rewind arrives with the canonical benchmark.

**The 15-client, α = 0.5, 3-attacker cell repeats GPU batch 1** under new names, and its whole-update runs give the same backdoor numbers (0.6 / 71.4 / 0.6%): the GPU server reproduces itself across jobs.

**Summary per setting.** Columns:
- *Worst under poisoning*: the lowest accuracy over sign flip, ALIE and Min-Max at all three attacker shares.
- *Backdoor*: success at 10 / 20 / 30% attackers.
- *Honest isolated*: per run, with no attack.

| Setting | Model | No attack | Worst under poisoning | Backdoor | Honest isolated |
|---|---|---|---|---|---|
| 15 clients, α = 0.5 | FedHEP | 49.19 | 24.1 | 70 / 79 / 80% | 0 |
| | Defended FedHEP | 48.68 | 32.1 | 72 / 79 / 80% | 0 |
| | FedHEP+ | 48.48 | 34.7 | 2.0 / 22 / 58% | 2.7 |
| | L0 | 48.21 | 37.0 | 40 / 64 / 76% | 0 |
| | R3 (root head) | 47.89 | **44.6** | 14 / 57 / 71% | 1.3 |
| | **FedHEP-Guard screen** | 47.36 | 43.3 | **0.4 / 0.6 / 12%** | 4.3 |
| | FLAME | 43.12 | 38.8 | 0.4 / 0.5 / 0.4% | 0 |
| 15 clients, α = 0.1 | FedHEP | 61.09 | 29.6 | 20 / 30 / 32% | 0 |
| | Defended FedHEP | 60.76 | 46.4 | 21 / 25 / 26% | 0 |
| | FedHEP+ | 59.93 | 54.0 | 22 / 29 / 29% | 7.7 |
| | L0 | 60.32 | 52.5 | 19 / 23 / 23% | 0 |
| | R3 (root head) | 60.43 | 29.9 | 18 / 28 / 32% | 1.3 |
| | **FedHEP-Guard screen** | 60.21 | **57.8** | 15 / 20 / 31% | 1.7 |
| | FLAME | 49.76 | 46.5 | **0.8 / 1.0 / 3.4%** | 0 |
| 30 clients, α = 0.5 | FedHEP | 45.22 | 22.8 | 63 / 73 / 74% | 0 |
| | Defended FedHEP | 44.28 | 25.7 | 62 / 72 / 74% | 0 |
| | FedHEP+ | 44.41 | 26.6 | 22 / 58 / 67% | 4.3 |
| | L0 | 44.69 | 36.6 | 17 / 61 / 68% | 0 |
| | R3 (root head) | 44.34 | 39.5 | 34 / 61 / 69% | 0.7 |
| | **FedHEP-Guard screen** | 43.30 | **40.2** | **0.5 / 0.9 / 5.3%** | **14.7** |
| | FLAME | 38.29 | 30.1 | 0.5 / 0.5 / 0.7% | 0 |
| 30 clients, α = 0.1 | FedHEP | 56.13 | 41.0 | 16 / 23 / 27% | 0 |
| | Defended FedHEP | 55.62 | 47.5 | 13 / 21 / 24% | 0 |
| | FedHEP+ | 55.64 | 50.7 | 16 / 21 / 25% | 2.3 |
| | L0 | 55.11 | 49.0 | 6.0 / 19 / 23% | 0 |
| | R3 (root head) | 55.29 | 40.3 | 15 / 23 / 25% | 5.0 |
| | **FedHEP-Guard screen** | 55.36 | **54.4** | 1.0 / 17 / 25% | 7.0 |
| | FLAME | 45.01 | 39.3 | **0.6 / 1.9 / 3.5%** | 0 |

**50 clients, α = 0.1 (partial).** Only the no-attack and 5-attacker sign-flip cells are complete, so there is no worst case or backdoor column yet:

| Model | No attack | Sign flip, 5 attackers | Honest isolated per run, no attack |
|---|---|---|---|
| FedHEP | **50.64** | 46.52 | 0 |
| Defended FedHEP | 49.37 | 47.87 | 0 |
| FedHEP+ | 49.68 | **48.10** | 1.0 |
| L0 | 49.28 | 47.27 | 0 |
| R3 (root head) | 49.54 | 47.80 | 6.0 |
| R3, whole update | 49.41 | 47.82 | 16.7 |
| **FedHEP-Guard screen** | 49.08 | 47.42 | **22.0** |
| FLAME | 39.92 | 39.39 | 0 |

**What the grid shows:**
- **The most robust model against model poisoning.** FedHEP-Guard's screen has the best worst-case accuracy in three of four settings. In the fourth it is 1.3 points behind R3. It never collapses: its worst case stays within 1–4 points of its no-attack accuracy. R3 with the root-head signal falls to 29.9 and 40.3 at α = 0.1.
- **The best backdoor defense short of FLAME at α = 0.5.** With 10 and 20% attackers it holds the backdoor under 1%. With 30% it lets through 12% at 15 clients and 5.3% at 30 clients (1.2 / 11 / 3.5% by seed; all 9 attackers caught by rounds 3–8). FLAME stops every backdoor in every setting, but costs 6–11 points with no attack and has a lower worst case.
- **At α = 0.1 the backdoor is not solved.** The screen lets through up to 31%; only FLAME holds. Strong skew makes honest clients as different from each other as attackers are.
- **Honest isolations grow with the number of clients, under both skews.** Per run with no attack: 4.3 at 15 clients and 14.7 at 30 (α = 0.5); 1.7, 7.0 and **22.0** at 15, 30 and 50 clients (α = 0.1). At 50 clients that is 44% of the clients, against FedHEP+'s 1.0. Accuracy hardly suffers (49.08 against 49.68), because isolated clients keep training on their own and keep their personal heads. It is still wrong to blame them, and it would make the detection numbers of a paper meaningless. The charts below show when it happens.

**In which rounds honest clients are isolated.** Cumulative honest clients isolated per round, mean of 3 runs, at the middle attacker share. Where the green line hides the orange one, the whole-update screen and the union screen isolated the same clients.

![Honest isolations per round, 15 clients](../figures/phase6_isolation_rounds_n15.png)

![Honest isolations per round, 30 clients](../figures/phase6_isolation_rounds_n30.png)

Why honest isolations grow with the number of clients, and why a larger gate would not fix it, is in chapter 8.2.

## 6.14 Earlier runs: every model tested, per machine

Before the benchmark (6.1) is complete, the runs already made give one chart per attack and machine, including the Mac's and the Acer's runs that the benchmark does not use. Each chart holds every model ever run in setting D1 on that machine. Machines are never mixed (chapter 1.3), so the Mac and the GPU server each get their own set.

**What is included:**
- Setting D1: 15 clients, α = 0.5, 10,000 training images, 30 rounds. Under attack, exactly 3 attackers, placed at random, with no boost and no stop.
- Only models with all three seeds (42, 123, 7). The bar is the mean, the line ±1 standard deviation.
- **Models are identified by their training configuration**, so a model run in two studies appears once. Evaluation-only switches such as the exit guard do not count.
- FedAvg and Multi-Krum use the authors' 3 local epochs (E = 3); every FedHEP-family model uses 5.
- **Colours are model families:** grey baselines, dark grey the published FedHEP models, blue the established robust rules (Phase 5), orange FedHEP+ and its variants (Phases 3–4), green L0, R2, R3 and their variants (Phase 6), yellow FedHEP-Guard's union screen.
- The Acer's 30-round D1 runs hold only L0 and the union screen with different $f$ (chapter 6.12), so it has no charts here.

The charts are made by `python scripts/plot_all_models.py outputs`, which also writes the numbers to `docs/results/all_models.json`.

### Mac (MPS)

FedHEP-Guard itself (union screen **and** rewind) has not run on the Mac; it ran on the Acer over 100 rounds (6.10), and on the GPU it comes with the benchmark (6.1). The charts therefore mark its closest relative, **FedHEP-Guard without rewind** (the union screen, `r3wc`), in bold. "R3 gate 8, root head" is the Phase 6 rung FedHEP-Guard is built on, not FedHEP-Guard (chapter 6.9, 5.8).

| Attack | Chart | Models | Best accuracy | FedHEP-Guard without rewind | Note |
|---|---|---|---|---|---|
| No attack | `figures/all_models_mac_clean.png` | 31 | Defended FedHEP 48.4 | not run on the Mac | the FedHEP family sits within 1.5 points; FLAME 42.8, Multi-Krum 23.4 |
| Sign flip | `figures/all_models_mac_sign_flip.png` | 23 | FedHEP+ (full population) 46.4; R3 gate 8 and 12, root head, 46.1–46.2 | not run on the Mac | every screen holds; Defended FedHEP 38.1, FedHEP 33.7, FedAvg 8.5 |
| Label flip | `figures/all_models_mac_label_flip.png` | 17 | FedHEP+ 46.3 | not run on the Mac | the FedHEP+ family takes the top 7 places |
| Random noise | `figures/all_models_mac_random_noise.png` | 17 | Median 46.6 | not run on the Mac | FedHEP 5.4 and FedAvg 1.2 collapse |
| ALIE | `figures/all_models_mac_alie.png` | 14 | FedHEP 47.8, L0 47.6, FedHEP+ 47.6 | not run on the Mac | the two-sided screens lose 1.7–3.1 points (the ALIE puzzle, 7.4f); FLAME 40.8 |
| Min-Max | `figures/all_models_mac_min_max.png` | 11 | R3 gate 8 and 12, root head, 46.0–46.1 | not run on the Mac | Bulyan 37.5 |
| Backdoor | `figures/all_models_mac_backdoor.png` | 31 | R3 gate 12, root head, 47.0 (success 62%) | 46.1, rank 21 of 31; success **0.8%** | success under 1%: whole-update screen (with or without rewind), union screen, Bulyan, FLAME, Multi-Krum |
| Stealth backdoor | `figures/all_models_mac_stealth_backdoor.png` | 7 | L0 46.9 (success 69%) | 45.2, rank 6 of 7; success **0.4%** | success under 1%: union screen and FLAME (0.8%) only |

![Every model, no attack, Mac](../figures/all_models_mac_clean.png)

![Every model, sign flip, Mac](../figures/all_models_mac_sign_flip.png)

![Every model, label flip, Mac](../figures/all_models_mac_label_flip.png)

![Every model, random noise, Mac](../figures/all_models_mac_random_noise.png)

![Every model, ALIE, Mac](../figures/all_models_mac_alie.png)

![Every model, Min-Max, Mac](../figures/all_models_mac_min_max.png)

![Every model, backdoor, Mac](../figures/all_models_mac_backdoor.png)

![Every model, stealth backdoor, Mac](../figures/all_models_mac_stealth_backdoor.png)

### GPU server (CUDA)

| Attack | Chart | Models | Best accuracy | FedHEP-Guard without rewind | Note |
|---|---|---|---|---|---|
| No attack | `figures/all_models_gpu_clean.png` | 16 | FedHEP 49.2 | 47.4, rank 15 of 16 | R3 and its variants 47.4–48.1; FLAME 43.1 |
| Sign flip | `figures/all_models_gpu_sign_flip.png` | 13 | FedHEP+ 46.3; R3 gate 8, root head, 46.1 | 45.8, rank 9 of 13 | L0 41.6, Defended FedHEP 38.8, FedHEP 33.8 |
| ALIE | `figures/all_models_gpu_alie.png` | 16 | FedHEP+ 47.8 | 45.3, rank 14 of 16 | FLAME 41.7 |
| Min-Max | `figures/all_models_gpu_min_max.png` | 12 | R3 gate 8, sign alignment, 46.4 | 45.6, rank 8 of 12 | FLAME 43.1 |
| Backdoor | `figures/all_models_gpu_backdoor.png` | 18 | Defended FedHEP 47.5 (success 79%) | 44.8, rank 15 of 18; success **0.6%** | success under 1%: union screen and FLAME (0.5%) only; the six rewind variants (`c5__`) stay at 24–45% |

![Every model, no attack, GPU](../figures/all_models_gpu_clean.png)

![Every model, sign flip, GPU](../figures/all_models_gpu_sign_flip.png)

![Every model, ALIE, GPU](../figures/all_models_gpu_alie.png)

![Every model, Min-Max, GPU](../figures/all_models_gpu_min_max.png)

![Every model, backdoor, GPU](../figures/all_models_gpu_backdoor.png)

**How to read them together.** No single model is best everywhere:
- The undefended and lightly defended models are the most accurate with no attack and under ALIE, and collapse under sign flip, noise or the backdoor.
- The screens give up 1–2 points with no attack and hold under every attack.
- Only two models stay under 1% backdoor success on both machines, and against the stealth backdoor on the Mac: FedHEP-Guard's union screen and FLAME. The union screen is about 3 points more accurate than FLAME under the backdoor on both machines (46.1 against 43.1 on the Mac, 44.8 against 41.9 on the GPU).

The benchmark (6.1) replaces these charts with one machine, one set of attackers and every model, in every setting.
