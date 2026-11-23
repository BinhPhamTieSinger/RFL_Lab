# Operations guide

How to set up a machine, run experiments, move results between the Mac, the Acer and the lab GPU server, and publish. Commands are meant to be pasted as they are. Paths assume the repository is at `~/Documents/AI_researching/Robust_Federated_Learning/RFL_Lab` on the Mac and `~/Documents/RFL_Lab` on the Acer.

## 1. The machines

| | Mac | Acer | Lab GPU server |
|---|---|---|---|
| Hardware | MacBook Pro, Apple M5, 32 GB | Acer Swift, Core Ultra 5 226V, Intel Arc 130V, 16 GB | 4 x NVIDIA RTX PRO 4500 (32 GB each), Slurm |
| PyTorch device | `mps` | `xpu` | `cuda` |
| Speed | about 25 s per round | about 25 s per round | measured by the pilot |
| Role | analysis, hypotheses and sweep design (no sweeps since 2026-11-19); the only writer of `main` | runner; pushes results to branch `acer` | main runner; pushes results to branch `pc` |
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

**The large grid.** The pilot showed that one run already fills a GPU (`--procs 1`; about 4.5 min per 30-round run, about 50 runs per hour on four GPUs). One job, in this order: the large grid first (1,872 runs, about 40 hours, longer if the 50-client runs are slower), then the mechanism studies:
```bash
qbatch --gpus 4 --cpus 16 --mem 96G --time 3-00:00:00 scripts/gpu_sweep.py \
    configs/sweeps/gpu_grid_n15.yaml configs/sweeps/gpu_grid_n30.yaml configs/sweeps/gpu_grid_n50.yaml \
    configs/sweeps/gpu_c4_d2.yaml configs/sweeps/gpu_c2_alie.yaml configs/sweeps/gpu_c5_recovery.yaml \
    configs/sweeps/gpu_c6_durability.yaml --procs 1 --set data.root=data
```
If the lab refuses the time limit, use the longest it allows and resubmit the same command when the job ends: finished runs are skipped. Progress: `ls outputs | grep -c '^gg'` (of 1,872). The grid files are written by `scripts/make_gpu_grid.py` (client counts, attacker counts, ALIE z per n and m); edit that script, not the YAML.

**The benchmark** (report 6.1): 52 models × 8 conditions × 9 settings × 3 seeds = 11,070 runs, of which about 600 already exist in the large grid and are skipped. At about 50 runs per hour on four GPUs that is about 9 days, so it is split into one sweep file per setting, in priority order: `bench_0` FedAvg and Multi-Krum in every setting (270 runs), `bench_1` D1 and `bench_2` D2 (1,200 runs each, about a day each), then the grid's settings, then the follow-up and paper settings. After the job above has ended:
```bash
bash scripts/runner_sync.sh pc --update
qbatch --gpus 4 --cpus 16 --mem 96G --time 3-00:00:00 scripts/gpu_sweep.py \
    configs/sweeps/bench_0_fedavg_family.yaml configs/sweeps/bench_1_gg15_a05.yaml configs/sweeps/bench_2_gg20_a02.yaml \
    configs/sweeps/bench_3_gg30_a05.yaml configs/sweeps/bench_4_gg15_a01.yaml configs/sweeps/bench_5_gg30_a01.yaml \
    configs/sweeps/bench_6_gg50_a01.yaml configs/sweeps/bench_7_gg50_a05.yaml configs/sweeps/bench_8_gg20_a05.yaml \
    configs/sweeps/bench_9_gg15_a005.yaml --procs 1 --set data.root=data
```
Resubmit the same command each time the 3-day limit ends; finished runs are skipped. The sweep files are written by `scripts/make_benchmark.py` (models, settings, conditions); edit that script, not the YAML. On the Mac, `python scripts/plot_benchmark.py outputs` redraws every benchmark chart and the chart index of report 6.1, from whatever GPU runs exist. After the benchmark: `configs/sweeps/gpu_c7_stealth.yaml`.

**A sweep on all four GPUs:**
```bash
qbatch --gpus 4 --cpus 16 --mem 96G --time 12:00:00 scripts/gpu_sweep.py \
    configs/sweeps/gpu_d1_signals.yaml --procs 1 --set data.root=data
```
Several sweep files can follow each other in one job (list them in order). Resubmitting the same command after a time limit or a crash resumes: finished runs are skipped. Do not run two jobs on the same sweep file at once. For the one-week option on two GPUs: `--gpus 2 --time 7-00:00:00` (check the longest time the lab allows).

**Watch and stop:** `squeue -u "$USER"`, `tail -f logs/qbatch-<job>.out`, worker logs in `logs/gpu_sweep_<job>_w*.log`, `ls outputs | grep -c '^gg'` for finished grid runs, `scancel <job>` to stop.

**Push results** (any time, also while a job runs; only finished runs have `metrics.json`): `bash scripts/runner_sync.sh pc` (section 4). `outputs_env/` holds one file per job: commit, container image, PyTorch, CUDA and GPU names, for the paper's reproducibility section.

**Take new code and sweep files** when no job is running: `bash scripts/runner_sync.sh pc --update`. A job reloads the code at the start of each sweep file, so updating mid-job can mix two code versions in one batch.

**Device parity** (does the device change the numbers, and by how much?). Run `configs/sweeps/device_parity.yaml` (12 runs of 30 rounds) on every machine, each into its own folder, then compare:
```bash
# GPU server: twice, so its own run-to-run noise is measured too (about 30 minutes on 4 GPUs)
qbatch --gpus 4 --cpus 16 --mem 64G --time 03:00:00 scripts/gpu_sweep.py configs/sweeps/device_parity.yaml \
    --procs 1 --set data.root=data run.output_dir=outputs_parity/cuda
qbatch --gpus 4 --cpus 16 --mem 64G --time 03:00:00 scripts/gpu_sweep.py configs/sweeps/device_parity.yaml \
    --procs 1 --set data.root=data run.output_dir=outputs_parity/cuda_b
# Mac (about 3 hours) and Acer (about 3 hours), after their current queues
python scripts/sweep.py configs/sweeps/device_parity.yaml --set run.output_dir=outputs_parity/mps
python scripts/sweep.py configs/sweeps/device_parity.yaml --set run.output_dir=outputs_parity/xpu
# anywhere, once the folders are pulled together
python scripts/compare_devices.py outputs_parity
```

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
The Mac no longer runs sweeps (2026-11-19); it analyses, writes code and pushes. Its earlier runs stay valid for Mac-only comparisons.

Acer (inside tmux, so it survives closing the SSH window). The current files are Phase 8: `acer_c11_shared.yaml` (21 runs), `acer_c11_control.yaml` (3) and `acer_c12_shared_durability.yaml` (6 runs of 100 rounds), about 14 hours in all:
```bash
tmux new -s rfl11        # or: tmux attach -t rfl11
cd ~/Documents/RFL_Lab && source ~/venvs/rfl/bin/activate
bash scripts/runner_sync.sh acer --update
python -m pytest -q tests/test_shared_signal.py
python scripts/sweep.py configs/sweeps/acer_c11_shared.yaml && \
python scripts/sweep.py configs/sweeps/acer_c11_control.yaml && \
python scripts/sweep.py configs/sweeps/acer_c12_shared_durability.yaml
```
When those have finished, the next Acer file is Phase 8 at scale, `acer_c13_shared_scale.yaml` (18 runs, about 7 hours; predictions P30–P32). It needs no new code:
```bash
tmux new -s rfl13        # or: tmux attach -t rfl13
cd ~/Documents/RFL_Lab && source ~/venvs/rfl/bin/activate
bash scripts/runner_sync.sh acer --update
python scripts/sweep.py configs/sweeps/acer_c13_shared_scale.yaml
```

Detach with Ctrl-b then d; `tmux attach -t rfl11` brings it back. Check progress from anywhere: `ls ~/Documents/RFL_Lab/outputs | wc -l`. If `tmux ls` says "no server running", the Acer restarted (`uptime`); start the same sweep command again. Push results any time with `bash scripts/runner_sync.sh acer`.

GPU server: see "Lab GPU server" above; the next file after the large grid is `configs/sweeps/gpu_c7_stealth.yaml`.

### Which sweep runs where
The current assignment is always in the latest "Run plan" entry of [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md); the header comment of every sweep file names its machine.

## 4. Moving results and code between machines

Since 2026-11-19 the Mac designs experiments and analyses results, and the Acer and the GPU server run them. Each machine has its own branch on GitHub:

| Branch | Written by | Holds |
|---|---|---|
| `main` | the Mac only | code, sweep files, docs, every merged result; Netlify deploys it |
| `acer` | the Acer only | `main` plus the Acer's result commits |
| `pc` | the GPU server only | `main` plus the GPU server's result commits |

```
Mac (main) ──push──> GitHub main ──(runner_sync.sh --update)──> acer, pc     new code and sweep files
Acer (acer), GPU server (pc) ──(runner_sync.sh)──> GitHub acer, pc ──(hub_collect.sh)──> Mac (main)   results
```

A runner never pulls code in the middle of a sweep; pushing results does not touch code, so it is safe at any time. Runners rebase their branch onto `main`, so the Mac's merges never conflict, and result commits the Mac has already merged drop out of the runner's branch on its next rebase.

### On a runner: push results, take new sweeps
```bash
bash scripts/runner_sync.sh acer            # the Acer; the GPU server uses "pc". Any time, also during a sweep
bash scripts/runner_sync.sh acer --update   # between sweeps: also rebase onto main for new code and sweep files
```
The script commits `outputs/`, `outputs_parity/`, `outputs_env/` and `outputs_pilot/`, never code; uncommitted code edits (a colleague's) are set aside during the rebase and put back.

**First time on a runner** (the script arrives with the first rebase):
```bash
git branch -M acer                          # "pc" on the GPU server
git add outputs/ outputs_parity/ outputs_env/ 2>/dev/null; git commit -m "acer results"
git fetch origin && git rebase --autostash origin/main
git push -u origin acer
```
The Acer needs a GitHub token to push (`git config --global credential.helper store`, then enter the user name and a fine-grained token with write access to this repository at the first push).

### On the Mac: collect, analyse, publish
```bash
bash scripts/hub_collect.sh        # fetch, merge origin/acer and origin/pc, list new runs in the visualizer
# ... analyse, write docs and figures, add new sweep files ...
git add <the files you changed> && git commit -m "..."
git push origin main               # Netlify deploys; the runners pick it up with --update
```
Add files by path, not with `git add -A`: the Mac's working copy may hold a colleague's uncommitted edits. `scripts/update_history_manifest.py` (called by `hub_collect.sh`) lists finished runs in the visualizer's Experiment history.

`scripts/sync_acer.sh` (rsync from the Acer over Tailscale) still works as a fallback when the Acer cannot push.

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
