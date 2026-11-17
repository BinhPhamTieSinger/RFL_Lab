"""Run one or more sweeps across every GPU of a Slurm job, several runs per GPU at once.

Submit with qbatch on the lab GPU server (docs/OPERATIONS.md, "Lab GPU server"):

  qbatch --gpus 4 --cpus 16 --mem 64G --time 12:00:00 scripts/gpu_sweep.py \
      configs/sweeps/gpu_d1_signals.yaml --procs 3 --set data.root=data

Each of the GPUs x procs workers runs `scripts/sweep.py <spec> --shard w/W` for every spec in turn, so the
workers split each grid without overlap. Finished runs are skipped, so resubmitting the same command resumes. Run
one job per sweep file at a time: two jobs sharding the same file would run the same runs twice.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gpus() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return visible.split(",")
    import torch

    return [str(i) for i in range(torch.cuda.device_count())] or ["cpu"]


def git_commit() -> str:
    """The checked-out commit, read from .git directly: the lab's container has no git executable."""
    git = ROOT / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return "unknown"


def record_environment(path: Path) -> None:
    """What the paper's reproducibility section needs: code version, container, PyTorch, GPUs."""
    import torch

    lines = [
        f"commit: {git_commit()}",
        f"slurm job: {os.environ.get('SLURM_JOB_ID', 'none')}",
        f"python: {sys.version.split()[0]} ({sys.executable})",
        f"torch: {torch.__version__}, CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}",
        *(f"gpu {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())),
    ]
    settings = ROOT / "quickstart.toml"
    if settings.exists():
        lines += ["quickstart.toml:", settings.read_text()]
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:5]), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--procs", type=int, default=1, help="runs at once on each GPU")
    ap.add_argument("--set", nargs="*", default=[], help="passed to every run, e.g. data.root=data")
    a = ap.parse_args()

    devices = gpus()
    workers = len(devices) * a.procs
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    # The lab's container hides Slurm's variables, so add the process id: jobs started in the same second must not
    # share log files.
    job = os.environ.get("SLURM_JOB_ID") or f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    (ROOT / "outputs_env").mkdir(exist_ok=True)
    record_environment(ROOT / "outputs_env" / f"gpu_{job}.txt")  # tracked: pushed with the results
    print(f"{len(devices)} GPU(s) x {a.procs} = {workers} workers, {cpus} CPUs; "
          f"logs in logs/gpu_sweep_{job}_w*.log", flush=True)

    codes = [0] * workers

    def work(w: int) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": devices[w % len(devices)],
               "OMP_NUM_THREADS": str(max(1, cpus // workers)), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        with open(logs / f"gpu_sweep_{job}_w{w}.log", "a") as log:
            for spec in a.specs:
                cmd = [sys.executable, "-u", str(ROOT / "scripts/sweep.py"), spec, "--shard", f"{w}/{workers}"]
                if a.set:
                    cmd += ["--set", *a.set]
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                codes[w] = codes[w] or code

    start = time.time()
    threads = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"finished in {(time.time() - start) / 60:.1f} min; worker exit codes {codes}", flush=True)
    sys.exit(1 if any(codes) else 0)


if __name__ == "__main__":
    main()
