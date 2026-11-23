# 3. Defended FedHEP: the published defense

Defended FedHEP is FedHEP with the authors' defense. This chapter describes what the defense adds, changes and removes, with its formulas and the reason for each part. Its reproduction is in chapter 6.2, and how it fares under attack, including a backdoor the paper did not test, in chapter 6.4.

## 3.1 What Defended FedHEP adds to FedHEP

Defended FedHEP keeps FedHEP's model and local training. It replaces **every aggregation**, at the parents and at Global, with **SCCF** (soft-cosine consensus filtering), and adds a **trust ledger** that can isolate a client for good.

![Defended FedHEP: aggregation, changes against FedHEP](figures/diagrams/04_defended_round.png)

**[a] Norm bound.** Each update is scaled down to at most $k$ times the lower-quartile norm of the round's updates ($k = 2$ at parents, $3$ at Global):

$$\mathrm{bound}(\Delta_i) = \Delta_i \cdot \min\Bigl(1, \frac{k\,Q_{0.25}(\lVert\Delta\rVert)}{\lVert\Delta_i\rVert}\Bigr)$$

**[b] Signal.** Trust looks only at the **root head's weight rows for the classes client $i$ holds**: $v_i = \mathrm{rows}_{Y_i}(W_r)$ of the bounded update. The trunk, the other heads and BatchNorm play no part.

**[c] Reference.** The mean of the unit vectors, $r = \frac1n\sum_i v_i/\lVert v_i\rVert$. If $\lVert r\rVert < 0.2$ there is no consensus: step [d] is skipped entirely (no cosines, no hard rejection, no softmax) and every sender gets the same weight $t_i = 1/n$. [e] then takes the plain mean of the bounded updates, and [f] gives every client full credit, $\min(1, n\cdot\tfrac1n) = 1$.

**[d] Trust weight.** Each sender's weight is a softmax of its cosine to the reference, at temperature $T$:

$$t_i = \frac{\exp\bigl(\cos(v_i, r)/T\bigr)}{\sum_j \exp\bigl(\cos(v_j, r)/T\bigr)}$$

$T = 0.5$, multiplied by 0.9 when the cosines vary and by 1.05 otherwise. At Global, a sender with $\cos(v_i, r) < -0.1$ is rejected outright ($t_i = 0$); if every sender is rejected, all get $1/n$ again.

**[e] SCCF step.** The aggregate is the trust-weighted sum of the bounded updates. BatchNorm buffers take the coordinate-wise median instead:

$$\Delta = \sum_i t_i\,\mathrm{bound}(\Delta_i)$$

**[f] Trust ledger.** After every aggregation (parents and Global), each client's score is updated from its share of the weight. A score below 0.25 isolates the client for good, in both tiers:

$$q_i \leftarrow 0.85\,q_i + 0.15\,\min(1,\ n\,t_i), \qquad q_i < 0.25 \Rightarrow i \text{ isolated}$$

**Groups** are random and fixed for the whole run. The defended code never calls FedHEP's similarity grouping.

**Legend:**

| Symbol | Meaning |
|---|---|
| $Q_{0.25}(\lVert\Delta\rVert)$ | lower quartile of the round's update norms |
| $k$ | norm-bound multiplier (2 at parents, 3 at Global) |
| $v_i$ | client $i$'s signal: root-head rows of its classes |
| $r$ | reference direction (mean of unit signals) |
| $t_i$ | SCCF trust weight of client $i$ in this aggregation |
| $T$ | softmax temperature |
| $q_i$ | ledger score of client $i$, starting at 1 |
| $n$ | number of senders in this aggregation ($n\,t_i = 1$ is an equal share) |

### Why each part is in Defended FedHEP

| Part | What it is for | Why it is in the model | What we measured |
|---|---|---|---|
| [a] Norm bound | caps each update's length | one huge update (noise, gradient ascent, a boosted backdoor) would otherwise dominate any weighted sum | random noise: FedHEP 5.4% → Defended FedHEP 45.8% |
| [b] Root-head signal | what trust is computed on | the root head's rows for a client's own classes are small, cheap to compare and class-specific | it misses backdoors, which live mostly in the trunk (chapter 5.4) |
| [c] Mean-direction reference | what "normal" looks like | the direction most honest clients share | a coordinated minority pulls it: backdoor attackers looked *more* trustworthy than honest clients (AUROC 0.36) |
| [d] Softmax trust, hard reject | down-weight dissenters smoothly; drop clearly reversed updates | a soft weight keeps honest but unusual clients; a hard cut removes updates pointing backwards | sign flip +4.4 over FedHEP, but the weights flatten late in training (6.4) |
| [e] BatchNorm median | robust BatchNorm statistics | running means and variances are not checked by the cosine signal and are easy to poison | part of the noise robustness |
| [f] Trust ledger | memory across rounds, and removal | an attacker should be removed, not only down-weighted every round | never isolated anyone in 24 runs |
| Random fixed groups | groups that do not depend on the updates | the authors' code does not say why; our reading is that attackers then cannot steer which group they join | — |

## 3.2 What it changes against FedHEP

| | FedHEP | Defended FedHEP |
|---|---|---|
| Global aggregation | FedAvg mean | SCCF (norm bound, root-head trust, softmax weights) |
| Parent aggregation | copy Global or average, with momentum | SCCF |
| Groups | similarity k-means, regrouped | random, fixed |
| BatchNorm buffers | averaged | coordinate median |
| Removing a client | never | trust ledger: $q_i < 0.25$ isolates for good |
