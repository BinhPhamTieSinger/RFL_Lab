"""Static figures of the report (docs/report/figures/), drawn from the model's structure rather than from runs.

  python scripts/plot_report_figures.py

guard_coverage.png: which parts of the multi-head ResNet-9 each FedHEP-Guard mechanism reads or changes. The flow
charts come from scripts/draw_report_diagrams.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from matplotlib.patches import Rectangle

_spec = importlib.util.spec_from_file_location("plot_results", Path(__file__).with_name("plot_results.py"))
assert _spec and _spec.loader
pr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pr)
plt = pr.plt
OUT = Path(__file__).resolve().parents[1] / "docs" / "report" / "figures"

PARTS = ["prep", "layer1", "res1", "layer2", "layer3", "res2", "BatchNorm\nstatistics", "root head",
         "parent head", "local head"]
# mechanism -> {part index: "r" (reads) or "w" (changes)}. Clients take the trunk, BatchNorm statistics and root
# head from Global, the parent head from their parent, and keep their own local head.
ALL_W = {i: "r" for i in range(6)} | {7: "r", 8: "r", 9: "r"}
MECH = [
    ("FedHEP+ screen (root-head signal)", {7: "r"}),
    ("[1a] whole-update signal", ALL_W),
    ("[1b] critical-layer signal (k most disputed)", {i: "r" for i in range(6)} | {7: "r"}),
    ("[5] L0 at the Global step", {i: "w" for i in range(8)}),
    ("parents: SCCF over kept members", {8: "w"}),
    ("[6] rewind (trunk and root head)", {i: "w" for i in range(7)} | {7: "w"}),
    ("[7] exit guard (prediction only)", {2: "r", 3: "r", 4: "r"}),
    ("ACLM class mask (from FedHEP)", {8: "r", 9: "r"}),
]


def coverage() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, len(PARTS))
    ax.set_ylim(0, len(MECH) + 1.2)
    ax.axis("off")
    for j, part in enumerate(PARTS):
        shade = "#e8eef9" if j < 7 else "#fdefc8"
        ax.add_patch(Rectangle((j + 0.04, len(MECH) + 0.15), 0.92, 0.9, color=shade))
        ax.text(j + 0.5, len(MECH) + 0.6, part, ha="center", va="center", fontsize=8.5)
    for i, (name, cells) in enumerate(MECH):
        y = len(MECH) - 1 - i
        ax.text(-0.1, y + 0.45, name, ha="right", va="center", fontsize=9)
        for j in range(len(PARTS)):
            mode = cells.get(j)
            color = {"r": "#9fd9c0", "w": "#0b7a52", None: "#f2f1ee"}[mode]
            ax.add_patch(Rectangle((j + 0.04, y + 0.08), 0.92, 0.8, color=color))
            if mode:
                ax.text(j + 0.5, y + 0.48, "reads" if mode == "r" else "changes", ha="center", va="center",
                        fontsize=7, color=pr.INK if mode == "r" else "white")
    ax.text(3.5, len(MECH) + 1.15, "shared trunk", ha="center", fontsize=9, color=pr.INK2)
    ax.text(8.5, len(MECH) + 1.15, "classifier heads", ha="center", fontsize=9, color=pr.INK2)
    ax.set_title("Where each mechanism acts in the multi-head ResNet-9 (what each client receives from it)",
                 loc="left", pad=22)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "guard_coverage.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote docs/report/figures/guard_coverage.png")


if __name__ == "__main__":
    pr.style()
    coverage()
