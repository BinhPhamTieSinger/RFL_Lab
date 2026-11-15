# Operations guide

How to set up a machine, run experiments, move results between the Mac and the Acer, and publish. Commands are meant to be pasted as they are. Paths assume the repository is at `~/Documents/AI_researching/Robust_Federated_Learning/RFL_Lab` on the Mac and `~/Documents/RFL_Lab` on the Acer.

## 1. The two machines

| | Mac | Acer |
|---|---|---|
| Hardware | MacBook Pro, Apple M5, 32 GB | Acer Swift, Core Ultra 5 226V, Intel Arc 130V, 16 GB |
| PyTorch device | `mps` | `xpu` |
| Speed | about 25 s per round | about 25 s per round |
| Role | source of truth for the repository; pushes to GitHub | second runner; results are pulled to the Mac |
| Reach | local | `ssh tiesinger@100.117.221.19` (Tailscale) |

**Rule:** a comparison is only valid within one machine. The same seed gives slightly different numbers on Apple and Intel GPUs, so every sweep file states which machine it belongs to (the machine whose earlier runs it is compared against). Each run records its device in `metrics.json`.

## 2. One-time setup

### Mac
```bash
cd ~/Documents/AI_researching/Robust_Federated_Learning
git clone https://github.com/BinhPhamTieSinger/RFL_Lab.git   # already done
cd RFL_Lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
bash scripts/check.sh                                          # all gates must pass
```
CIFAR-100 is read from `../Paper_Reproduction/data` (sibling of `RFL_Lab`).

### Acer
```bash
sudo apt install -y git rsync tmux python3-venv
mkdir -p ~/Documents && cd ~/Documents
git clone https://github.com/BinhPhamTieSinger/RFL_Lab.git
python3 -m venv ~/venvs/rfl && source ~/venvs/rfl/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
cd RFL_Lab && pip install -e ".[dev]"
python -c "import torch; print(torch.xpu.is_available())"     # must print True
```
Copy the data from the Mac (178 MB, once):
```bash
# on the Mac
scp -r ~/Documents/AI_researching/Robust_Federated_Learning/Paper_Reproduction/data \
    tiesinger@100.117.221.19:~/Documents/Paper_Reproduction/
```
Benchmark: `time python scripts/run.py configs/cifar100_fedhep_defended.yaml --set run.rounds=2 run.eval_every=2 name=bench run.output_dir=outputs_bench` should take about one minute.

Make `(rfl)` activate automatically on the Acer: `echo 'source ~/venvs/rfl/bin/activate' >> ~/.bashrc`.

### Lab GPU (NVIDIA A1000)
```bash
nvidia-smi                                   # note the "CUDA Version" in the top right
git clone https://github.com/BinhPhamTieSinger/RFL_Lab.git && cd RFL_Lab
python3 -m venv .venv && source .venv/bin/activate
# pick the wheel matching the driver's CUDA version (cu121, cu124 or cu126); cu124 shown:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Copy the CIFAR-100 folder next to the repository (`../Paper_Reproduction/data`), then time one round:
`time python scripts/run.py configs/cifar100_fedhep_defended.yaml --set run.rounds=2 run.eval_every=2 name=bench run.output_dir=outputs_bench`.
The A1000 is a third machine: its results are compared only with each other, so every A1000 sweep carries its own baselines (`a1000_*.yaml`).

## 3. Running experiments

### One configuration
```bash
python scripts/run.py configs/cifar100_fedhep.yaml --seeds 42 123 7 --set data.alpha=0.2 name=my_test
```
Writes `outputs/my_test_s42/`, `_s123/`, `_s7/`, each with `config.yaml` and `metrics.json`.

### A sweep (a grid of methods, attacks and seeds)
```bash
python scripts/sweep.py configs/sweeps/phase6_ladder_d1.yaml --dry-run   # list the runs, run nothing
python scripts/sweep.py configs/sweeps/phase6_ladder_d1.yaml             # run them
```
A run that already has `metrics.json` is skipped, so an interrupted sweep resumes where it stopped: press Ctrl-C whenever you like and start the same command again. At most one run (about 12 minutes) is lost.

### A queue of sweeps
Mac (keeps the machine awake):
```bash
cd ~/Documents/AI_researching/Robust_Federated_Learning/RFL_Lab && source .venv/bin/activate
caffeinate -i bash -c 'for s in phase6_ladder_d1 phase6_ladder_d2; do python scripts/sweep.py configs/sweeps/$s.yaml || break; done'
```
Acer (inside tmux, so it survives closing the SSH window):
```bash
tmux new -s rfl          # or: tmux attach -t rfl
cd ~/Documents/RFL_Lab
for s in followup_backdoor_heads followup_n30_sign; do python scripts/sweep.py configs/sweeps/$s.yaml || break; done
```
Detach with Ctrl-b then d; `tmux attach -t rfl` brings it back. Check progress from anywhere: `ls ~/Documents/RFL_Lab/outputs | wc -l`.

### Which sweep runs where
The current assignment is always in the latest "Run plan" entry of [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); the header comment of every sweep file names its machine.

## 4. Moving results and code between machines

Everything flows Acer → Mac → GitHub → Acer.

### Pull the Acer's finished runs to the Mac
```bash
cd ~/Documents/AI_researching/Robust_Federated_Learning/RFL_Lab
bash scripts/sync_acer.sh
```
This copies `outputs/` from the Acer with `rsync --ignore-existing`: a run already on the Mac is never overwritten, and a run still in progress arrives without `metrics.json` and is completed on the next sync. Run it as often as you like.

### Commit and push from the Mac
```bash
git add -A
git commit -m "Results from the Mac and the Acer"
git push
```
`outputs/` is tracked on purpose: every number in the documents is backed by a run in the repository.

### Update the Acer's code
```bash
cd ~/Documents/RFL_Lab
git pull --ff-only
```
If `git pull` refuses because the Acer has local changes you do not need: `git fetch origin && git reset --hard origin/main` (runs in `outputs/` are untracked there and are not touched). Do this between sweeps, not during one: a sweep already running keeps its old code until it is restarted.

## 5. Reading results

```bash
python scripts/summarize.py outputs --filter a02__        # mean ± sd over seeds for matching runs
python scripts/plot_results.py outputs                    # Phase 1-5 charts into docs/figures/
python scripts/plot_followup.py outputs
python scripts/plot_adaptive.py outputs outputs_followup_full outputs_followup_holdout
python scripts/plot_phase6.py outputs
```
`metrics.json` holds, per run: `history` (accuracy, bottom-10% accuracy and backdoor success at each evaluation, per head for the FedHEP family), `detection` (flagged clients per round, precision, recall, F1), `diagnostics` (FedHEP+ only: scores, z, scale, isolations per round), `resources` (seconds per round, peak process and device memory) and `device`.

## 6. Before any result is recorded

```bash
bash scripts/check.sh
```
Runs ruff, mypy, import-linter, the test suite and the code-graph check. All must pass. New code comes with a test; a new result comes with a log entry that names its runs.

## 7. Writing a new sweep file

```yaml
# What it tests, why, which machine, how many runs, about how long.   <- always
seeds: [42, 123, 7]
common:                      # applied to every run
  data.alpha: 0.5
  run.rounds: 30
axes:                        # the grid: one run per combination, named <axis1>__<axis2>
  - my__method_a: {config: configs/cifar100_fedhep_defended.yaml, method: fedhep_plus, plus.reference: median}
    my__method_b: {config: configs/cifar100_fedhep_defended.yaml, method: fedhep_robust, robust.rule: arc_nnm_tm}
  - clean: {}
    sign3: {attack.kind: sign_flip, attack.count: 3}
```
Check it with `--dry-run` before starting. Keep each file self-contained or name the baselines it relies on, and keep it on the machine those baselines ran on.

## 8. Common problems

| Symptom | Cause and fix |
|---|---|
| `git` on the Mac says `index.lock` exists | A previous git command was interrupted. `rm .git/index.lock` once you are sure no git process is running. |
| `ssh: connect to host 100.117.221.19: Network is unreachable` | Tailscale is off on one side, or the Acer is asleep. |
| Sweep much slower than 12 min per run on the Acer | Heat throttling. Check `sensors`; raise the laptop or lower the ambient temperature. |
| `torch.xpu.is_available()` is `False` after a reboot | Intel GPU driver not loaded; `sudo apt install intel-opencl-icd intel-level-zero-gpu level-zero` and reboot. |
| Two different numbers for the same run name | The two machines both ran it. Keep the one on the machine its sweep file names; delete the other. |
| Resumed sweep reruns a finished run | Its `metrics.json` is missing (the run was interrupted during the final write). Delete that run's folder and resume. |
