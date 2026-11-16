# Operations guide

How to set up a machine, run experiments, move results between the Mac, the Acer and the lab GPU server, and publish. Commands are meant to be pasted as they are. Paths assume the repository is at `~/Documents/AI_researching/Robust_Federated_Learning/RFL_Lab` on the Mac and `~/Documents/RFL_Lab` on the Acer.

## 1. The machines

| | Mac | Acer | Lab GPU server |
|---|---|---|---|
| Hardware | MacBook Pro, Apple M5, 32 GB | Acer Swift, Core Ultra 5 226V, Intel Arc 130V, 16 GB | 4 x NVIDIA RTX PRO 4500 (32 GB each), Slurm |
| PyTorch device | `mps` | `xpu` | `cuda` |
| Speed | about 25 s per round | about 25 s per round | measured by the pilot |
| Role | source of truth for the repository; pushes to GitHub | second runner; results are pulled to the Mac | main runner from now on; pushes its results to GitHub |
| Reach | local | `ssh tiesinger@100.117.221.19` (Tailscale) | your SSH session only; code and results through GitHub |

**Rule:** a comparison is only valid within one machine. The same seed gives slightly different numbers on Apple, Intel and NVIDIA GPUs, so every sweep file states which machine it belongs to (the machine whose earlier runs it is compared against). Each run records its device in `metrics.json`.

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

### Lab GPU server (4 x NVIDIA RTX PRO 4500)
Slurm schedules the GPUs; code runs inside the lab's PyTorch container (Apptainer) with a small virtual environment on top (`qsetup`, `qshell`, `qbatch`; the lab's guide explains them). Only you can reach the server; code arrives and results leave through GitHub (`git pull`, `git push`).

**One-time setup.** In your SSH session (replace `<project>` with the project directory you were given):
```bash
cd /scratch/projects/<project>
ls -A                                   # should be empty, or hold only what qsetup made
# empty directory:
git clone https://github.com/BinhPhamTieSinger/RFL_Lab.git .
# not empty (qsetup already ran): the repository's files win, qsetup's folders stay
#   git init -b main && git remote add origin https://github.com/BinhPhamTieSinger/RFL_Lab.git
#   git fetch origin && git checkout -f -B main origin/main
git config user.name "BinhPhamTieSinger" && git config user.email "<the address you use on GitHub>"
PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 qsetup   # wait for "Python environment prepared."; the long timeout survives slow package downloads
git status --short                      # if qsetup replaced requirements.txt: git checkout requirements.txt && qsetup
```
`requirements.txt` lists only small packages: PyTorch comes from the container and must not be installed with pip.

**CIFAR-100.** On the server prompt (not inside a job; compute nodes may have no internet):
```bash
mkdir -p data && cd data
until curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -o cifar-100-python.tar.gz \
      https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz; do sleep 5; done   # resumes after each cut
md5sum cifar-100-python.tar.gz          # eb9058c3a382ffc7106e4002c42a8d85
tar xzf cifar-100-python.tar.gz && ls cifar-100-python && cd ..   # meta  test  train
```
If the download never completes, copy the archive from a computer that reaches the server: `scp cifar-100-python.tar.gz <user>@<server>:/scratch/projects/<project>/data/`, then run the `md5sum` and `tar` lines.

Every command below passes `--set data.root=data`.

**Check the environment** (about 5 minutes):
```bash
qshell --gpus 1 --cpus 4 --mem 16G --time 00:30:00
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python -m pytest -q tests/test_benchmark_harness.py      # must pass
exit
```

**Pilot: how many runs fit on one GPU** (three 1-GPU jobs at once, about 15 minutes):
```bash
for p in 1 2 4; do
  qbatch --gpus 1 --cpus 8 --mem 32G --time 01:00:00 scripts/gpu_sweep.py configs/sweeps/gpu_pilot.yaml \
      --procs $p --set data.root=data run.output_dir=outputs_pilot/p$p
done
squeue -u "$USER"                       # PD = waiting, R = running
tail -n 3 logs/qbatch-*.out             # each job ends with "finished in X min"
```
Then push the pilot (below). Choose `--procs` from it: the most runs per hour while peak GPU memory stays under about 28 GB of the 32 GB.

**The campaign, in order** (each line one job; submit the next when the previous has finished, or list several files in one job):
```bash
qbatch --gpus 4 --cpus 16 --mem 96G --time 12:00:00 scripts/gpu_sweep.py \
    configs/sweeps/gpu_d1_signals.yaml configs/sweeps/gpu_c2_alie.yaml --procs 3 --set data.root=data
qbatch --gpus 4 --cpus 16 --mem 96G --time 12:00:00 scripts/gpu_sweep.py \
    configs/sweeps/gpu_c4_d2.yaml configs/sweeps/gpu_c5_recovery.yaml --procs 3 --set data.root=data
qbatch --gpus 4 --cpus 16 --mem 96G --time 24:00:00 scripts/gpu_sweep.py \
    configs/sweeps/gpu_c6_durability.yaml --procs 3 --set data.root=data
```

**A sweep on all four GPUs:**
```bash
qbatch --gpus 4 --cpus 16 --mem 96G --time 12:00:00 scripts/gpu_sweep.py \
    configs/sweeps/gpu_d1_signals.yaml --procs 3 --set data.root=data
```
Several sweep files can follow each other in one job (list them in order). Resubmitting the same command after a time limit or a crash resumes: finished runs are skipped. Do not run two jobs on the same sweep file at once. For the one-week option on two GPUs: `--gpus 2 --time 7-00:00:00` (check the longest time the lab allows).

**Watch and stop:** `squeue -u "$USER"`, `tail -f logs/qbatch-<job>.out`, worker logs in `logs/gpu_sweep_<job>_w*.log`, `ls outputs | grep -c g1__` for finished runs, `scancel <job>` to stop.

**Push results** (any time, also while a job runs; only finished runs have `metrics.json`):
```bash
git add outputs/ outputs_pilot/ outputs_env/
git commit -m "GPU server results"
git pull --rebase && git push
```
`outputs_env/` holds one file per job: commit, container image, PyTorch, CUDA and GPU names, for the paper's reproducibility section. Then on the Mac: `git pull`.

**Update the server's code** with `git pull --rebase` when no job is running: a job reloads the code at the start of each sweep file, so pulling mid-job can mix two code versions in one batch.

The GPU server is a third machine: its results are compared only with each other, so every sweep run there carries its own baselines (`gpu_*.yaml`).

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

The visualizer's Experiment history lists only runs named in `visualizer/frontend/research/history-manifest.json`. After committing new runs, add the completed ones (the Netlify build log also names any it finds missing):
```bash
python3 - <<'PY'
import json, glob, os
path = "visualizer/frontend/research/history-manifest.json"
listed = {e["source"] for e in json.load(open(path))}
done = sorted(os.path.dirname(m) for m in glob.glob("outputs*/*/metrics.json")
              if not m.startswith(("outputs_smoke", "outputs_bench")) and json.load(open(m)).get("final"))
entries = [{"source": s} for s in sorted(listed | set(done))]
json.dump(entries, open(path, "w"), indent=2); open(path, "a").write("\n")
print(len(set(done) - listed), "runs added")
PY
```
Note: `git add -A` also commits whatever the visualizer's working copy holds; when someone is editing `visualizer/`, add `outputs*/` and the manifest by path instead.

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
