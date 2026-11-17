"""Compare the device-parity runs (configs/sweeps/device_parity.yaml) across devices.

  python scripts/compare_devices.py outputs_parity

Each subfolder of the argument holds one device's runs (e.g. outputs_parity/mps, outputs_parity/cuda,
outputs_parity/cuda_b for a repeat). Prints, for each pair of folders, the per-run difference in final pooled
accuracy, its mean absolute value, the largest one, and whether the method ranking within each attack and seed is
the same; first for repeats on one device (the noise floor), then across devices.
"""
import itertools
import json
import sys
from pathlib import Path


def load(folder: Path) -> dict[str, float]:
    return {p.parent.name: json.loads(p.read_text())["final"]["pooled"]
            for p in folder.glob("*/metrics.json") if json.loads(p.read_text()).get("final")}


def ranking(runs: dict[str, float], attack: str, seed: str) -> list[str]:
    rows = {k.split("__")[1]: v for k, v in runs.items() if k.endswith(f"__{attack}_s{seed}")}
    return sorted(rows, key=rows.get, reverse=True)


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs_parity")
    devices = {d.name: load(d) for d in sorted(root.iterdir()) if d.is_dir()}
    devices = {k: v for k, v in devices.items() if v}
    if len(devices) < 2:
        sys.exit(f"need at least two device folders with finished runs under {root}")
    for a, b in itertools.combinations(devices, 2):
        common = sorted(set(devices[a]) & set(devices[b]))
        if not common:
            continue
        diffs = {n: devices[b][n] - devices[a][n] for n in common}
        cells = {(n.split("__")[2].rsplit("_s", 1)[0], n.rsplit("_s", 1)[1]) for n in common}
        same = sum(ranking(devices[a], atk, s) == ranking(devices[b], atk, s) for atk, s in cells)
        kind = "repeat" if a.split("_")[0] == b.split("_")[0] else "devices"
        mean = sum(map(abs, diffs.values())) / len(diffs)
        print(f"\n{a} vs {b} ({kind}): {len(common)} runs, mean |diff| {mean:.2f}, "
              f"max |diff| {max(map(abs, diffs.values())):.2f} points; "
              f"ranking identical in {same} of {len(cells)} cells")
        for n in common:
            print(f"  {n:32s} {devices[a][n]:6.2f} {devices[b][n]:6.2f} {diffs[n]:+6.2f}")


if __name__ == "__main__":
    main()
