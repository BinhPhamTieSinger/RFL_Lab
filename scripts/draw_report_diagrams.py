"""Flow diagrams of the report (docs/report/figures/diagrams/*.png), with every label typeset in LaTeX.

  python scripts/draw_report_diagrams.py            # writes docs/report/diagrams/*.mmd and renders them (mmdc)

Labels are lists of lines. In a line, $...$ is inline math and everything else is text; a line that starts with
$$ is a displayed formula. Mermaid typesets the whole label with KaTeX (mermaid 11; `<` must be written \\lt).
Colours mark what a model changes against the previous one: green new, yellow changed, red dashed removed, grey
unchanged."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "report" / "diagrams"
OUT = ROOT / "docs" / "report" / "figures" / "diagrams"

STYLE = """  classDef same fill:#f2f1ee,stroke:#9a9893,color:#0b0b0b
  classDef new fill:#d5f2e6,stroke:#1baf7a,stroke-width:2px,color:#0b0b0b
  classDef changed fill:#fdefc8,stroke:#eda100,stroke-width:2px,color:#0b0b0b
  classDef gone fill:#fde3d9,stroke:#eb6834,stroke-dasharray:4 3,color:#0b0b0b
  classDef trunk fill:#e8eef9,stroke:#2a78d6,color:#0b0b0b
  classDef head fill:#fdefc8,stroke:#eda100,color:#0b0b0b
  classDef k fill:#d5f2e6,stroke:#1baf7a,color:#0b0b0b
  classDef x fill:#fde3d9,stroke:#eb6834,color:#0b0b0b
  classDef w fill:#fdefc8,stroke:#eda100,color:#0b0b0b"""

_TEXT_ESC = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "<": r"\textless{}",
             ">": r"\textgreater{}"}


def _text(s: str) -> str:
    return r"\text{" + "".join(_TEXT_ESC.get(ch, ch) for ch in s) + "}" if s else ""


def _math(s: str) -> str:
    return s.replace("<", r"\lt ").replace(">", r"\gt ")


def _line(line: str) -> str:
    if line.startswith("$$"):
        return r"\displaystyle " + _math(line.strip("$"))
    parts = re.split(r"\$([^$]+)\$", line)
    return "".join(_math(p) if i % 2 else _text(p) for i, p in enumerate(parts))


def tex(*lines: str) -> str:
    """A node or edge label: the lines centred one under another, typeset by KaTeX."""
    return '"$$' + r"\begin{array}{c}" + r" \cr ".join(_line(x) for x in lines) + r"\end{array}$$" + '"'


def node(name: str, lines: tuple[str, ...] | list[str], cls: str, shape: str = "box") -> str:
    open_, close = {"box": ("[", "]"), "decision": ("{", "}")}[shape]
    return f"  {name}{open_}{tex(*lines)}{close}:::{cls}"


def edge(a: str, b: str, *label: str, dotted: bool = False) -> str:
    arrow = "-.->" if dotted else "-->"
    return f"  {a} {arrow}|{tex(*label)}| {b}" if label else f"  {a} {arrow} {b}"


def legend(*rows: tuple[str, str]) -> str:
    lines = ['  subgraph LEG[" "]', "    direction LR"]
    lines += [f"    l{i}[{tex(text)}]:::{cls}" for i, (text, cls) in enumerate(rows)]
    lines += ["  end", "  style LEG fill:#fcfcfb,stroke:#e4e3df"]
    return "\n".join(lines)


CHANGE_LEGEND = legend(("unchanged", "same"), ("new in this model", "new"), ("changed in this model", "changed"),
                       ("removed in this model", "gone"))


def fedavg() -> str:
    return "\n".join([
        "flowchart LR",
        node("S", ["Server", "global model $w_t$"], "same"),
        node("C", ["Client $i$ copies $w_t$ and trains", "$E$ epochs of SGD on its data $D_i$"], "same"),
        node("U", ["upload the update", "$$\\Delta_i = w_t^i - w_t$$"], "same"),
        node("A", ["Server: weighted mean", "$$w_{t+1} = w_t + \\sum_i \\frac{n_i}{N}\\,\\Delta_i$$"], "same"),
        node("M", ["Malicious client $a$", "uploads any vector $\\Delta_a$"], "gone"),
        edge("S", "C", "send $w_t$"), edge("C", "U"), edge("U", "A"), edge("A", "S", "$w_{t+1}$"),
        edge("M", "A", "poisoned update", dotted=True),
        STYLE])


def fedhep_model() -> str:
    trunk = [("PREP", "prep, conv 3 to 32"), ("L1", "layer1, conv 32 to 64, pool"), ("R1", "res1, 2 convs + skip"),
             ("L2", "layer2, conv 64 to 128, pool"), ("L3", "layer3, conv 128 to 256, pool"),
             ("R2", "res2, 2 convs + skip"), ("GAP", "average pool: features $h(x) \\in \\mathbb{R}^{256}$")]
    lines = ["flowchart TD", node("X", ["image $x$", "32 by 32 by 3"], "same"),
             '  subgraph TRUNK["Shared trunk"]']
    lines += ["  " + node(n, [t], "trunk") for n, t in trunk]
    lines += ["  end"]
    lines += [edge("X", "PREP")] + [edge(a, b) for (a, _), (b, _) in zip(trunk, trunk[1:], strict=False)]
    lines += [
        node("HR", ["root head $W_r$", "shared by all clients"], "head"),
        node("HP", ["parent head $W_p$", "shared within a group"], "head"),
        node("HL", ["local head $W_l$", "private, never shared"], "head"),
        node("ENS", ["ensemble with class mask (ACLM)",
                     "$$p(y \\mid x) = \\mathrm{softmax}\\Big(\\sum_{h} \\pi_h\\, \\tilde z_h(x)\\Big)$$"], "same"),
        edge("GAP", "HR", "$z_r$"), edge("GAP", "HP", "$z_p$"), edge("GAP", "HL", "$z_l$"),
        edge("HR", "ENS"), edge("HP", "ENS"), edge("HL", "ENS"),
        "  style TRUNK fill:#fcfcfb,stroke:#2a78d6", STYLE]
    return "\n".join(lines)


def fedhep_round() -> str:
    return "\n".join([
        "flowchart TD",
        node("G", ["Global: trunk and root head $W_r$", "parent heads $W_p^{(g)}$, one per group"], "same"),
        node("C", ["Client $i$ trains $E = 5$ epochs on three losses",
                   "$$\\mathcal{L}_i = m_r\\,\\mathrm{CE}(z_r) + m_p\\,\\mathrm{CE}(\\tilde z_p) + "
                   "m_l\\,\\mathrm{CE}(\\tilde z_l)$$", "weights $m_h$ from the label-skew score $r_i$"], "changed"),
        node("LH", ["local head $W_l$ stays on the client"], "new"),
        node("UP", ["upload $\\Delta_i$"], "same"),
        node("GA", ["Global: FedAvg over all clients", "$$w \\leftarrow w + \\sum_i \\frac{n_i}{N}\\,\\Delta_i$$"],
             "same"),
        node("PA", ["Parent $g$ copies Global if its members agree",
                    "$$\\overline{\\cos}(\\Delta_i, \\Delta_j)_{i,j \\in g} > 0.5$$",
                    "otherwise it averages them; adaptive momentum"], "new"),
        node("GR", ["Grouping: k-means on random projections of $\\Delta_i$", "regroup when groups drift"], "new"),
        node("PR", ["Prediction: ensemble of the three heads",
                    "$$p(y \\mid x) = \\mathrm{softmax}\\Big(\\sum_h \\pi_h\\, \\tilde z_h(x)\\Big)$$"], "new"),
        edge("G", "C", "trunk, $W_r$, own group's $W_p$, own $W_l$"),
        edge("C", "UP"), edge("C", "LH"), edge("UP", "GA"), edge("UP", "PA"), edge("UP", "GR"), edge("GR", "PA"),
        edge("GA", "G"), edge("PA", "G"), edge("C", "PR"),
        CHANGE_LEGEND, STYLE])


def defended_round() -> str:
    return "\n".join([
        "flowchart TD",
        node("UP", ["updates $\\Delta_i$ from clients trained as in FedHEP"], "same"),
        node("NB", ["[a] norm bound",
                    "$$\\Delta_i \\leftarrow \\Delta_i \\cdot \\min\\Big(1,\\ "
                    "\\frac{k\\,Q_{0.25}(\\|\\Delta\\|)}{\\|\\Delta_i\\|}\\Big)$$",
                    "$k = 2$ at parents, $k = 3$ at Global"], "new"),
        node("SIG", ["[b] signal: root-head rows of client $i$'s classes",
                     "$$v_i = \\mathrm{rows}_{Y_i}(\\Delta W_r)$$"], "new"),
        node("REF", ["[c] reference: mean of unit vectors", "$$r = \\frac{1}{n}\\sum_i \\frac{v_i}{\\|v_i\\|}$$"],
             "new"),
        node("CON", ["consensus?", "$$\\|r\\| \\ge 0.2\\ ?$$"], "new", "decision"),
        node("EQ", ["no consensus: [d] skipped", "$$t_i = 1/n \\text{ for every sender}$$"], "new"),
        node("TW", ["[d] trust weight", "$$t_i = \\mathrm{softmax}_i\\big(\\cos(v_i, r) / T\\big)$$",
                    "Global: $t_i = 0$ if $\\cos(v_i, r) < -0.1$",
                    "(all rejected: $t_i = 1/n$)"], "new"),
        node("AGG", ["[e] SCCF step", "$$\\Delta = \\sum_i t_i\\,\\Delta_i$$",
                     "BatchNorm buffers: coordinate median"], "changed"),
        node("TL", ["[f] trust ledger", "$$q_i \\leftarrow 0.85\\,q_i + 0.15\\,\\min(1,\\ n\\,t_i)$$",
                    "isolated for good if $q_i < 0.25$"], "new"),
        node("PAR", ["parents and Global both aggregate with SCCF"], "changed"),
        node("GRP", ["groups: random, fixed for the whole run"], "changed"),
        node("OLD", ["FedAvg mean; similarity grouping"], "gone"),
        edge("UP", "NB"), edge("NB", "SIG"), edge("SIG", "REF"), edge("REF", "CON"), edge("CON", "TW", "yes"),
        edge("CON", "EQ", "no"), edge("TW", "AGG"), edge("EQ", "AGG"), edge("TW", "TL"), edge("EQ", "TL"),
        edge("AGG", "PAR"), edge("GRP", "PAR"),
        CHANGE_LEGEND, STYLE])


def plus_round() -> str:
    return "\n".join([
        "flowchart TD",
        node("UP", ["updates $\\Delta_i$ of clients not isolated"], "same"),
        node("S", ["[1] score against the coordinate-wise median",
                   "$$s_i = \\cos(v_i, \\tilde r), \\quad \\tilde r_j = \\mathrm{med}_k\\, v_{k,j}$$"], "changed"),
        node("Z", ["[2] robust z-score",
                   "$$z_i = \\frac{s_i - \\mathrm{med}(s)}{\\max\\big(1.4826\\,\\mathrm{MAD}(s),\\ "
                   "0.1\\,|\\mathrm{med}(s)|\\big)}$$"], "new"),
        node("EX", ["[3]", "$$z_i < -2.5\\ ?$$"], "new", "decision"),
        node("OUT", ["left out this round"], "new"),
        node("KEEP", ["kept"], "same"),
        node("EV", ["[4] evidence", "$$E_i \\leftarrow 0.7\\,E_i + 0.3\\,z_i$$"], "new"),
        node("ISO", ["$$E_i < -1.5\\ ?$$"], "new", "decision"),
        node("PERM", ["isolated for good"], "new"),
        node("RG", ["[5] regroup kept clients by weight similarity", "BatchNorm buffers left out"], "changed"),
        node("AG", ["parents and Global: SCCF over kept clients only"], "same"),
        node("TL", ["trust ledger"], "gone"),
        edge("UP", "S"), edge("S", "Z"), edge("Z", "EX"), edge("EX", "OUT", "yes"), edge("EX", "KEEP", "no"),
        edge("Z", "EV"), edge("EV", "ISO"), edge("ISO", "PERM", "yes"), edge("KEEP", "RG"), edge("RG", "AG"),
        CHANGE_LEGEND, STYLE])


def drill_down() -> str:
    return "\n".join([
        "flowchart TD",
        node("P", ["FedHEP+ screening, unchanged"], "same"),
        node("G", ["each parent $g$ with at least 3 eligible members"], "new"),
        node("GS", ["parent score", "$$c_g = \\cos(\\bar v_g, \\tilde r)$$"], "new"),
        node("NULL", ["permutation test against 64 random groups of the same size",
                      "$$z_g = \\frac{c_g - \\mathrm{med}(\\mathrm{null})}{1.4826\\,\\mathrm{MAD}(\\mathrm{null})}$$"],
             "new"),
        node("F", ["$$z_g < -2\\ ?$$"], "new", "decision"),
        node("V1", ["v1, Phase 4: stricter member cut this round", "$$z_i < -1$$"], "new"),
        node("V2", ["v2, Phase 4b: alignment with the other members",
                    "$$a_i = \\cos\\Big(u_i,\\ \\frac{1}{|g| - 1}\\sum_{j \\neq i} u_j\\Big), \\quad "
                    "u_i = v_i - \\tilde r$$"], "new"),
        node("E2", ["evidence update uses", "$$\\min\\big(z_i,\\ -z_i^{\\mathrm{align}}\\big)$$"], "new"),
        edge("P", "G"), edge("G", "GS"), edge("GS", "NULL"), edge("NULL", "F"), edge("F", "V1", "v1"),
        edge("F", "V2", "v2"), edge("V2", "E2"),
        CHANGE_LEGEND, STYLE])


def phase5_rules() -> str:
    return "\n".join([
        "flowchart LR",
        node("UP", ["updates $\\Delta_i$ from clients trained as in FedHEP"], "same"),
        node("PAR", ["parents: SCCF over the clients the rule accepts"], "same"),
        node("RULE", ["Global step replaced by one rule",
                      "Median: $\\mathrm{med}_i\\,\\Delta_{i,j}$ per coordinate",
                      "$$\\text{Trimmed mean: } \\mathrm{TM}_f(X)_j = \\frac{1}{n - 2f}\\sum_{i=f+1}^{n-f} "
                      "X_{(i),j}$$",
                      "Bulyan: Krum selection, then a trimmed mean",
                      "FLAME: HDBSCAN cluster, clip to median norm, add noise",
                      "FLTrust: $\\mathrm{ReLU}(\\cos(\\Delta_i, \\Delta_0))$ with server root data"], "changed"),
        node("GLOB", ["Global model"], "same"),
        node("TL", ["trust ledger"], "gone"),
        edge("UP", "PAR"), edge("UP", "RULE"), edge("RULE", "GLOB"),
        CHANGE_LEGEND, STYLE])


def guard_round() -> str:
    return "\n".join([
        "flowchart TD",
        node("C", ["clients train as in FedHEP", "three heads, $E = 5$ epochs"], "same"),
        node("UP", ["updates $\\Delta_i$ of clients not isolated"], "same"),
        node("SW", ["[1a] whole-update signal",
                    "$$s_i^{\\mathrm{w}} = \\cos\\big(\\Delta_i^{\\setminus \\mathrm{BN}},\\ "
                    "\\mathrm{med}_k\\,\\Delta_k^{\\setminus \\mathrm{BN}}\\big)$$"], "new"),
        node("SC", ["[1b] critical-layer signal",
                    "$$s_i^{\\mathrm{c}} = \\frac{1}{k}\\sum_{\\ell \\in L^\\star} "
                    "\\cos\\big(\\Delta_i^{(\\ell)}, \\tilde r^{(\\ell)}\\big)$$",
                    "$L^\\star$: the $k = 5$ most disputed layers"], "new"),
        node("ZW", ["[2] robust z, two-sided", "$$z_i^{\\mathrm{w}} \\leftarrow -|z_i^{\\mathrm{w}}|$$"], "changed"),
        node("ZC", ["[2] robust z, two-sided", "$$z_i^{\\mathrm{c}} \\leftarrow -|z_i^{\\mathrm{c}}|$$"], "changed"),
        node("GW", ["[3] gap gate", "$$g = \\frac{\\max_k \\delta_k}{\\mathrm{med}_k\\,\\delta_k} \\ge 8\\ ?$$"],
             "new", "decision"),
        node("GC", ["[3] gap gate", "$$g = \\frac{\\max_k \\delta_k}{\\mathrm{med}_k\\,\\delta_k} \\ge 8\\ ?$$"],
             "new", "decision"),
        node("U", ["[1c] union of the two screens",
                   "$$z_i = \\min\\big(z_i^{\\mathrm{w}},\\ z_i^{\\mathrm{c}}\\big)$$"], "new"),
        node("EX", ["[4] $z_i < -2.5$", "in 3 rounds in a row?"], "changed", "decision"),
        node("OUT", ["left out this round"], "same"),
        node("KEEP", ["kept clients"], "same"),
        node("EV", ["evidence", "$$E_i \\leftarrow 0.7\\,E_i + 0.3\\,z_i$$"], "same"),
        node("ISO", ["[4] $E_i < -1.5$ and $z_i < -1.5$", "in 3 rounds in a row?"], "changed", "decision"),
        node("PERM", ["isolated for good"], "same"),
        node("RW", ["[6] rewind trunk and root head",
                    "$$w \\leftarrow M \\odot w_{t^\\star} + (1 - M) \\odot w$$",
                    "$t^\\star$: before the first suspicious round"], "new"),
        node("RG", ["regroup by weight similarity"], "same"),
        node("PAR", ["parents: SCCF over kept members"], "same"),
        node("L0", ["[5] Global step: L0, replaces SCCF",
                    "$$\\Delta = \\mathrm{TM}_f\\Big(\\mathrm{NNM}_f\\big(\\mathrm{ARC}_f(\\Delta_1, \\dots, "
                    "\\Delta_n)\\big)\\Big)$$"], "new"),
        node("GLOB", ["Global model"], "same"),
        node("INF", ["[7] optional, at prediction: depth-exit guard",
                     "$$s(x) = 1 - \\tfrac{1}{3}\\sum_d e_d[\\hat y] > \\tau_i \\ \\Rightarrow\\ \\text{exits vote}$$"],
             "new"),
        edge("C", "UP"), edge("UP", "SW"), edge("UP", "SC"), edge("SW", "ZW"), edge("SC", "ZC"),
        edge("ZW", "GW"), edge("ZC", "GC"),
        edge("GW", "U", "no: $z \\leftarrow \\max(z, 0)$"), edge("GW", "U", "yes"),
        edge("GC", "U", "no: $z \\leftarrow \\max(z, 0)$"), edge("GC", "U", "yes"),
        edge("U", "EX"), edge("EX", "OUT", "yes"), edge("EX", "KEEP", "no"),
        edge("U", "EV"), edge("EV", "ISO"), edge("ISO", "PERM", "yes"), edge("PERM", "RW"),
        edge("KEEP", "RG"), edge("RG", "PAR"), edge("KEEP", "L0"), edge("L0", "GLOB"), edge("RW", "GLOB"),
        edge("GLOB", "INF"),
        CHANGE_LEGEND, STYLE])


def research_path() -> str:
    return "\n".join([
        "flowchart LR",
        node("P1", ["Phase 1", "reproduce FedAvg, FedHEP,", "Defended FedHEP", "matches the paper"], "k"),
        node("P2", ["Phase 2", "fair benchmark: attacks,", "backdoor, detection", "Defended never isolates;",
                    "backdoor 79%"], "k"),
        node("P3", ["Phase 3", "FedHEP+: robust z screen,", "permanent isolation, regroup",
                    "sign flip +8.3; backdoor 21%"], "k"),
        node("P4", ["Phase 4", "drill-down by parent", "failed: members alike"], "x"),
        node("P5", ["Phase 5", "classic rules in FedHEP", "none better overall"], "x"),
        node("FU", ["Follow-up diagnostics", "FedHEP+ fails at $\\alpha = 0.2$,", "misses ALIE and Min-Max,",
                    "backdoor persists"], "w"),
        node("P6", ["Phase 6", "FedHEP-Guard: L0, two-sided,", "gap gate, union screen, rewind"], "k"),
        node("B", ["Canonical benchmark", "every model, one machine"], "k"),
        edge("P1", "P2"), edge("P2", "P3"), edge("P3", "P4"), edge("P3", "P5"), edge("P4", "FU"),
        edge("P5", "FU"), edge("FU", "P6"), edge("P6", "B"),
        legend(("result carried forward", "k"), ("idea rejected", "x"), ("diagnosis of a failure", "w")),
        STYLE])


DIAGRAMS = {
    "01_fedavg_round": fedavg, "02_fedhep_model": fedhep_model, "03_fedhep_round": fedhep_round,
    "04_defended_round": defended_round, "05_fedhep_plus_round": plus_round, "06_drill_down": drill_down,
    "07_phase5_rules": phase5_rules, "08_fedhep_guard_round": guard_round, "10_research_path": research_path,
}


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in DIAGRAMS.items():
        source = SRC / f"{name}.mmd"
        source.write_text(build() + "\n")
        subprocess.run(["mmdc", "-i", str(source), "-o", str(OUT / f"{name}.png"), "-b", "#fcfcfb", "-s", "3",
                        "-w", "1600"], check=True, capture_output=True)
        print(f"wrote docs/report/figures/diagrams/{name}.png")


if __name__ == "__main__":
    main()
