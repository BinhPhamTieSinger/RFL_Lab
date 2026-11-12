Contents

1. [Summary](#summary)
2. [Architecture](#architecture)
3. [Client round](#client-round)
4. [Server round](#server-round)
5. [Data flow](#data-flow)
6. [Prediction](#inference)
7. [Paper vs code](#settings)
8. [Reported results](#results)
9. [Code map](#code-map)
10. [Regenerate tables](#reproduce)
11. [Caveats](#caveats)

AAMAS 2027 submission #2335 · reproduction guide

# FedHEP Reproduction Map

The FedHEP paper explains the idea but leaves out most of the settings that decide the numbers. The authors' repository has them. This page lays out how FedHEP actually runs in that code: the model, one training round, the data pipeline and the prediction rule. It also gives every setting side by side with the paper, and the exact commands that regenerate each table.

**Paper** final PDF, 9 pages

**Draft** earlier 8-page version

**Code** [nam200718/Topology-aware-FDL](https://github.com/nam200718/Topology-aware-FDL) at 001b4f6, plus its fedhep_supplementary_material.zip

### The code is the only complete spec

Rounds, batch size, learning-rate schedule, weight decay, data split, clustering trigger and ensemble rule are all fixed in code and absent from both PDFs.

### Table 2 FedHEP runs without SCCF or trust tracking

The personalization runs use `defense_mode: none`. The SCCF + trust defense only runs as "Defended FedHEP" in the attack table.

### FedHEP trains 5 local epochs, baselines 3

FedHEP's two stages (3 collaborative epochs, then 2 local-head epochs) both happen inside every round.

### Three Table 2 cells use a 20-round run

Ditto at α = 0.1 and 0.05 and FedHEP at α = 0.5 come from a run tagged "2nd run (higher accuracy)" with 20 rounds. Every other cell uses 25.

1 · Model

## One backbone, three heads

Every client holds a single half-width ResNet-9. Its 256-number feature vector feeds three linear heads that differ only in who shares them and which loss trains them.

Layer widths and losses as defined in `src/core/model.py` (`MultiHeadResNet9`) and `src/core/updater.py`. Heads carry a bias term: 3 × 25,700 parameters, 1,720,652 in total for CIFAR-100. The paper's "1.65M" is the backbone alone. The dashed path is the "dimensional capacity" anchor. For C = 100, the Local head sees features whose gradient back into the backbone is scaled by 0.1.

The Local head's class-balanced loss weights each observed class by 1/√(its count), normalized to mean 1. Neither PDF mentions this.

2 · One round on a client

## Five local epochs in two stages

In the code, the "two-stage schedule" the paper mentions happens inside each round. The first three epochs train the shared parts. The last two freeze the backbone and train only the private head.

The optimizer is SGD with Nesterov momentum 0.9 and weight decay 1e-4, batch size 32, re-created each round. The learning rate follows a cosine decay across rounds from 0.05 (FEMNIST: 0.02) to 0.001. The Root and Parent weights λ come from the paper's Bernstein formula. The skew ratio is r = (e^H − 1)/(C − 1) and the anchor is a = max(1/(2K), |Y_i|/C). Source: `_update_hep` in `src/core/updater.py` and `get_current_lr` in `src/core/base_engine.py`.

3 · One round on the server

## Cluster, aggregate twice, broadcast

Clients upload their whole model. The server decides cluster membership from the updates, builds one Parent head per cluster and one global Root model, then sends each client the global model plus its cluster's Parent head.

Source: `run_round` in `src/core/hierarchical_ensemble_engine.py` (Table 2) and `src/core/unified_defended_engine.py` (Table 3). The paper describes clients sketching their own Parent-head update before upload. In the code, clients send their full model and the server does the projection, so the sketch does not reduce what the server sees.

4 · Data pipeline

## Separate train and test pools, one shared split

Training and test images come from different CIFAR-100 splits. Both are cut by the same per-class Dirichlet proportions, so each client's test set has the same class mix as its training set.

Source: `get_cifar100` and `partition_data` in `src/data/dataset.py`. Images are only normalized; there is no crop, flip or other augmentation. In the IID regime the images are shuffled and dealt out equally instead. There is no minimum client size, so at α = 0.05 a client can end up with very few images. FEMNIST uses EMNIST-ByClass, 15,000 train and 3,000 test, with the same Dirichlet split rather than per-writer data.

5 · Evaluation

## How a FedHEP prediction is made

Accuracy is measured per client on its own test set and averaged. FedHEP is scored with the model each client holds after its local training in the last round. Global baselines are scored with the aggregated global model.

Source: `evaluate_ensemble`, `compute_head_weights` and `_safr_blend_weights` in `src/core/hierarchical_ensemble_engine.py`. The paper's fixed temperatures (1.0 / 0.8 / 0.5) are only a fallback. Bottom-10% is the mean of the lowest ⌈0.1·N⌉ client accuracies. Ditto is scored with each client's personal model, FedRep with each client's private head, and the other baselines open-set with the global model.

6 · Settings

## Paper vs code, setting by setting

Use the Code column to regenerate the paper's numbers. The last column shows what my earlier reimplementation assumed from the PDF alone, which explains why its FedAvg run reached 26.5% instead of 41.5%.

Code differs code contradicts the paper text Paper silent only the code gives it Matches

| Setting | Status | Paper (final) | Draft | Official code | My earlier guess |
| --- | --- | --- | --- | --- | --- |
| Clients, Table 2 CIFAR-100 | Matches | 15 | 10 | 15 | 15 |
| Data size | Paper silent | "13k samples" | 10,000 train / 3,000 test | 10,000 from train split + 3,000 from test split, partitioned separately (FEMNIST 15,000 / 3,000) | 13,000 pooled, 80/20 per client |
| Rounds, Table 2 | Paper silent | one of 25, 30, 40 | one of 25, 30, 40 | 25 (three cells: 20). Orchestrator default is 20, so pass `--rounds 25` | 30 |
| Local epochs | Code differs | E = 3 | E = 3 | Baselines 3. FedHEP 5 (3 collaborative + 2 Local-head) | 3 |
| Batch size | Paper silent | 32 or 64 | 32 or 64 | 32 | 64 |
| Learning rate | Paper silent | 0.02–0.05 | 0.02–0.05 | CIFAR-100 0.05, FEMNIST 0.02, cosine decay over rounds to 0.001 | 0.05 constant |
| Optimizer | Code differs | SGD, momentum 0.9, wd 5e-4 | same | SGD, Nesterov momentum 0.9, wd 1e-4 | SGD, momentum 0.9, wd 5e-4 |
| Augmentation | Paper silent | not stated | not stated | None, normalization only | random crop + flip |
| ResNet-9 | Paper silent | 1.65M params, "e.g. d = 512" | 1.65M params | Base width 32, average pooling, heads with bias, d = 256 | width 32, max pooling, no bias |
| Skew ratio r | Matches | (e^H − 1)/(C − 1) | 1 − H / ln C | (e^H − 1)/(C − 1) | (e^H − 1)/(C − 1) |
| Two-stage schedule | Code differs | "Root/Parent training, then Local-only" | not stated | Inside every round: epochs 1–3 then 4–5, for C ≥ 100 | rounds 1–25 then 26–30 |
| Ensemble weights | Paper silent | start at α, "refined online" | not stated | Per-batch update mixed 50/50 with prior π; at test time softmax(−entropy − loss + log α); Root only if r ≥ 0.85 | fixed at α |
| Head temperatures | Code differs | 1.0 / 0.8 / 0.5 | not stated | Matched to logit spread each batch; 1.0 / 0.8 / 0.6 only as fallback | 1.0 / 0.8 / 0.5 |
| Clustering | Code differs | clients sketch ΔW_p; spherical k-means | sketches, m = 256 | Server projects full-model Δ to 256-d; k-means, 10 iterations; warm-up and misalignment triggers | client sketch of ΔW_p |
| Clusters K | Paper silent | not stated | not stated | 3 | 3 |
| SCCF + trust in Table 2 | Code differs | described as part of FedHEP | described as part of FedHEP | Off (`defense_mode: none`). On only for "Defended FedHEP" | always on |
| SCCF temperature | Paper silent | not stated | not stated | 0.5, adaptive (decay factor 0.95, floor 0.1); hard-reject cos \< −0.1; equal weights if consensus norm \< 0.2 | 0.1 |
| Trust tracking | Matches | β 0.8 or 0.85, threshold 0.2 or 0.25 | not stated | β 0.85, threshold 0.25 | 0.85 / 0.25 |
| Ditto λ | Matches | 0.05 | 0.05 | 0.05 in the baseline registry (some YAML files use 0.1) | 0.05 |
| Multi-Krum | Paper silent | not stated | not stated | Keeps 3 of 15 clients each round | keeps 9 of 15 |
| FedRep head epochs | Paper silent | not stated | not stated | 1 head epoch, then body (3 passes total) | 10 head + 3 body |
| Gaussian attack σ | Paper silent | not stated | not stated | 2.0, added to the model weights | 1.0 |
| Attack table setup | Code differs | α = 0.5, 3 seeds | α = 0.05 | α = 0.5, 100 rounds; recorded std is 0.0, so effectively one seed | α = 0.5, 30 rounds |
| Ablation table | Matches | N = 10 (noted) | not stated | N = 10, 25 rounds | N = 10 |

7 · Reported results

## What the regenerated tables should match

These are the authors' own recorded numbers from `outputs/master_experimental_data.json`, the file the paper's tables are built from. Hover a column to read every method at that regime.

A hollow ring marks a cell taken from the 20-round "2nd run (higher accuracy)" instead of the 25-round main run.

### Attack robustness, CIFAR-100, α = 0.5

The solid blue line is Defended FedHEP, the version with SCCF and trust tracking. The dashed blue line is the same FedHEP used in Table 2. The paper's Table 3 does not show it, but the authors recorded it.

8 · Code map

## What runs when you regenerate Table 2

1

`scripts/run_aamas_suite.py --job 1`

Picks dataset defaults (`CIFAR100_DEFAULTS` or `FEMNIST_DEFAULTS`) and applies `--rounds`, `--seeds`, `--methods`, `--regimes`.

2

`src/baselines/run_all_baselines.py · run_personalization_suite`

Loops method × regime × seed and skips runs whose `metrics.json` already exists.

3

`src/baselines/experiment_configs.py · create_personalization_config`

Turns a method ID (`fedavg`, `ditto`, `topo` = FedHEP, …) into a full config; the defaults in `src/config.py · ClientConfig` fill everything else.

4

`src/baselines/factory.py`

Chooses the engine: `CentralizedEngine` (FedAvg, FedProx, Multi-Krum, Ditto, FedRep), `ScaffoldEngine`, `HierarchicalEnsembleEngine` (FedHEP) or `UnifiedDefendedEngine` (Defended FedHEP).

5

`src/core/base_engine.py`

Loads and subsamples the data, partitions train and test with the same seed, builds the model, sets the cosine learning-rate schedule.

6

`src/core/updater.py · PyTorchLocalUpdater.update`

Local training per client: `_update_hep`, `_update_ditto`, `_update_split_head` (FedRep) or `_update_standard`. Attacks are applied here too.

7

`engine.run_round → aggregation → evaluate_ensemble`

Server step from section 3, then per-client accuracy and bottom-10% written to `outputs/baselines/…/metrics.json`.

8

`scripts/compute_multiseed_statistics.py · scripts/generate_all_tables.py`

Mean ± std over seeds and the LaTeX tables in `outputs/tables/`. `scripts/full_audit_test.py` checks paper numbers against the master data file.

**Topology-aware-FDL/** ├── **scripts/**run_aamas_suite.py *# entry point for Jobs 1–5 (only in the zip)* ├── **configs/** *# YAML experiment files (only in the zip)* ├── **outputs/**master_experimental_data.json *# authors' recorded results (only in the zip)* ├── **src/** │ ├── **config.py** *# every default: ClientConfig holds \~70 hyperparameters* │ ├── **data/**dataset.py *# loaders + Dirichlet partition (only in the repo)* │ ├── **core/**model.py *# ResNet-9, MultiHeadResNet9, MobileNetV3-Small* │ ├── **core/**updater.py *# all local training rules, attacks* │ ├── **core/**hierarchical_ensemble_engine.py *# FedHEP server + evaluation* │ ├── **core/**unified_defended_engine.py *# Defended FedHEP (SCCF, clip, trust)* │ ├── **core/**aggregator.py *# FedAvg, soft-cosine trust, trimmed mean* │ ├── **baselines/** *# method registry, SCAFFOLD, FedProx, Multi-Krum* │ └── **topologies/**hierarchical.py *# random projection + k-means clustering* └── **paper/** *# LaTeX source of the paper*

9 · Regenerate

## Commands for each table

The GitHub repo and the supplementary zip each miss pieces: the repo lacks the run scripts, configs and recorded results, and the zip lacks `src/data/`. Unzipping with `-n` adds the missing files without overwriting the newer repo code. I ran the smoke test below on the merged copy and it completed (1.5 minutes on a 2-core CPU).

### Set up once

Python 3.10+, about 15 MB of code

```
git clone https://github.com/nam200718/Topology-aware-FDL
cd Topology-aware-FDL
unzip -n fedhep_supplementary_material.zip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_aamas_suite.py --job 1 --smoke-test --dataset synthetic
```

- CIFAR-100 downloads into `./data` on first use. If you already downloaded it, add `--data-dir` with that folder.

### Table 2B · CIFAR-100 personalization

7 methods × 5 regimes × 3 seeds

```
python scripts/run_aamas_suite.py --job 1 --rounds 25
```

```
cp -r outputs/baselines/personalization_cifar100 outputs/personalization_cifar100_r25
python scripts/run_aamas_suite.py --job 1 --rounds 20 --methods ditto topo --regimes severe extreme moderate --force
```

- The second block regenerates the three 20-round cells (Ditto severe and extreme, FedHEP moderate). Output folders are named by method, regime and seed only, so it overwrites those 25-round results. The first line keeps a copy.
- The authors' recorded elapsed time per cell (all 3 seeds) is about 34 minutes for FedAvg and 65 minutes for FedHEP on their GPU.

### Table 2A · FEMNIST personalization

lr 0.02, 15,000 / 3,000 images

```
python scripts/run_aamas_suite.py --job 1 --dataset femnist --rounds 25 --force
```

- Uses torchvision's EMNIST-ByClass split (about 535 MB download).
- `--force` is needed because the job manifest marks Job 1 as done after the CIFAR-100 run.

### Table 3 · Attack robustness

α = 0.5, 4 attacks × 5 rates

```
python scripts/run_aamas_suite.py --job 2 --rounds 100 --methods fedavg multikrum ditto topo_defended --seeds 42
```

- The recorded runs use 100 rounds and show std 0.0, which points to a single seed. Add `topo` to also get the undefended FedHEP line.

### Tables 4–6 · Scaling, MobileNetV3, ablations

N = 50 · MobileNetV3 · N = 10

```
python scripts/run_aamas_suite.py --job 3 --rounds 40
python scripts/run_aamas_suite.py --job 4 --rounds 30
python scripts/run_aamas_suite.py --job 5 --clients 10 --rounds 25
python scripts/run_aamas_suite.py --job finalize
```

- Jobs 3 and 4 default to 20 and 15 rounds; the recorded results used 40 (at 20% participation) and 30, hence the flags. The memory and latency numbers in Table 5 come from CUDA profiling and cannot be measured the same way on a Mac.
- `finalize` rebuilds the LaTeX tables from whatever has finished.

10 · Caveats

## What to keep in mind

1. **Mixed runs in Table 2.** Three CIFAR-100 cells come from a 20-round run labelled "higher accuracy", next to 25-round runs for every other cell. A faithful reproduction should report both settings for those cells.
2. **Unequal local compute.** FedHEP trains 5 local epochs per round; the baselines train 3.
3. **Unequal evaluation.** FedHEP is scored closed-set with each client's locally trained model. Global baselines are scored open-set with the aggregated model.
4. **The method in the paper is not the method in Table 2.** SCCF and trust tracking appear only in Defended FedHEP. Several mechanisms that do run in Table 2 are not in the paper: the dimensional-capacity anchor, class-balanced Local loss, learned blend weights, the IID shortcut, adaptive clustering triggers and the cluster sync rule.
5. **Clustering privacy.** Clients send their full model and the server projects it, so the sketch gives no privacy at the upload step.
6. **Seeds and hardware.** Results on Apple MPS will not match CUDA bit for bit. The code seeds data splits and models, so differences should be small, not systematic.