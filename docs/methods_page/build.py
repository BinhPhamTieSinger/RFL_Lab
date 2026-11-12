# ruff: noqa: E501  (the page content is HTML and LaTeX text; long lines are intended)
"""Builds the RFL Lab methods page (one self-contained HTML file) from docs/diagrams/*.mmd and docs/figures/.

  python docs/methods_page/build.py . docs/methods_page/rfl_lab_methods.html
"""
import base64
import html
import sys
from pathlib import Path

REPO = Path(sys.argv[1])
OUT = Path(sys.argv[2])


def mmd(name: str) -> str:
    src = (REPO / "docs" / "diagrams" / f"{name}.mmd").read_text()
    return f'<div class="diagram"><pre class="mermaid">{html.escape(src)}</pre></div>'


def img(name: str, alt: str) -> str:
    data = base64.b64encode((REPO / "docs" / "figures" / name).read_bytes()).decode()
    return f'<figure class="chart"><img src="data:image/png;base64,{data}" alt="{alt}"><figcaption>{alt}</figcaption></figure>'


SECTIONS = [
    ("setting", "Setting and model", f"""
<p>15 clients share CIFAR-100 (10,000 training and 3,000 test images), split by a Dirichlet(α) label partition:
α&nbsp;=&nbsp;0.5 is moderately non-IID, α&nbsp;=&nbsp;0.05 very non-IID. Training runs 30 rounds of 5 local epochs,
with 3 groups called <em>parents</em>. Unless stated, exactly 3 of the 15 clients attack; the authors' rule
instead makes each client an attacker with probability <i>q</i>.</p>
<p>Every FedHEP variant uses one ResNet-9 trunk with three classifier heads. Which aggregation a head goes
through is what makes FedHEP hierarchical.</p>
{mmd("1_model")}
<div class="formula">Label skew of client <i>i</i>: \\[ r_i = \\frac{{e^{{H(y_i)}} - 1}}{{C - 1}} \\]
1 for uniform labels, 0 for a single class (<i>H</i> = label entropy, <i>C</i> = 100).</div>
"""),
    ("fedhep", "FedHEP: one round", f"""
<p>Each client starts from the Global model, its group's parent head and its own local head, trains, and sends
its model up two paths: to Global (everyone) and to its parent (its group).</p>
{mmd("2_fedhep_round")}
<div class="formula">Head loss weights (binomial schedule), with \\(a = \\max(\\tfrac{{1}}{{2K}}, |Y_i|/C)\\):
\\[ \\lambda_{{local}} = (1-r)^2, \\qquad \\lambda_{{parent}} = 2r(1-r), \\qquad \\lambda_{{root}} = a + (1-a)\\,r^2 \\]
Prediction mixes the heads' logits <i>z<sub>h</sub></i>:
\\[ p = \\mathrm{{softmax}}\\Big(\\sum_h \\omega_h z_h\\Big), \\qquad
\\omega = \\mathrm{{softmax}}\\big((-H_h - \\beta\\,\\ell_h + \\log \\alpha_h)/T\\big) \\]
with <i>H<sub>h</sub></i> the head's prediction entropy, <i>ℓ<sub>h</sub></i> its recent training loss and
<i>α</i> a learned prior. Clients with near-uniform labels use the root head only; classes a client never saw are
masked out.</div>
<p class="note">Found in Phase 1: the grouping cosine is computed on the full model state, where BatchNorm's batch
counters dominate, so in practice every parent copies Global every round.</p>
"""),
    ("defended", "Defended FedHEP (the authors' defense)", f"""
<p>Every aggregation, at the parents and at Global, becomes <strong>SCCF</strong> (soft-cosine consensus
filtering), with a trust ledger on top. Groups are random and fixed.</p>
{mmd("3_defended_round")}
<div class="formula">
\\[ \\tilde\\Delta_i = \\Delta_i \\cdot \\min\\Big(1, \\frac{{k\\,Q_1(\\|\\Delta\\|)}}{{\\|\\Delta_i\\|}}\\Big), \\qquad
\\bar u = \\frac1N\\sum_j \\frac{{\\tilde\\Delta_j}}{{\\|\\tilde\\Delta_j\\|}}, \\qquad
w_i = \\mathrm{{softmax}}_i\\Big(\\frac{{\\cos(\\tilde\\Delta_i, \\bar u)}}{{T}}\\Big) \\]
\\[ \\text{{step}} = \\sum_i w_i \\tilde\\Delta_i, \\qquad
\\text{{score}}_i \\leftarrow 0.85\\,\\text{{score}}_i + 0.15\\min(1, N w_i), \\qquad \\text{{isolate if score}}_i < 0.25 \\]
<i>k</i>&nbsp;=&nbsp;2 at parents and 3 at Global; <i>T</i>&nbsp;=&nbsp;0.5, tightened ×0.9 when cosines vary and
relaxed ×1.05 otherwise. Cosines use only the root-head rows of the classes a client holds. If
\\(\\|\\bar u\\| < 0.2\\) (no consensus), everyone gets \\(w_i = 1/N\\).</div>
<p class="note">Phase 2: the ledger never isolates anyone. Attackers keep 0.5–0.8 of an equal share, so their
score never falls below 0.25, and the weights flatten late in training.</p>
"""),
    ("plus", "FedHEP+ (Phase 3): what changes", f"""
<p>SCCF aggregation stays. The trust ledger is replaced by <strong>screening</strong>, which decides who enters both
tiers, and groups are re-formed by the similarity of the weights alone, with BatchNorm buffers left out.</p>
{mmd("4_fedhep_plus_round")}
<div class="formula">With \\(s_i = \\cos(\\tilde\\Delta_i, \\operatorname{{median}}_j \\tilde\\Delta_j)\\) on the root-head rows:
\\[ z_i = \\frac{{s_i - \\operatorname{{med}}(s)}}{{\\max\\big(1.4826\\cdot\\mathrm{{MAD}}(s),\\; 0.1\\,|\\operatorname{{med}}(s)|\\big)}},
\\qquad \\mathrm{{MAD}}(s) = \\operatorname{{med}}_j |s_j - \\operatorname{{med}}(s)| \\]
\\[ \\text{{leave out this round if }} z_i < -2.5, \\qquad E_i \\leftarrow 0.7\\,E_i + 0.3\\,z_i, \\qquad
\\text{{isolate for good if }} E_i < -1.5 \\]</div>
<ul class="points">
<li><strong>Why 1.4826·MAD:</strong> it estimates a standard deviation from the middle of the data, so a few
attackers cannot inflate it. A plain standard deviation grows with them and hides them.</li>
<li><strong>Why a z-score:</strong> it is unchanged when every cosine shrinks together late in training. That
shrinking is what made Defended FedHEP's fixed temperature and fixed 0.25 cut fade.</li>
<li><strong>The floor</strong> 0.1·|median| stops a very tight consensus from turning tiny differences into huge z.
Fewer than 5 senders are not screened.</li>
<li><strong>Evidence:</strong> one bad round is not enough. Two rounds at z&nbsp;=&nbsp;−3 give E&nbsp;=&nbsp;−1.53.</li>
</ul>
<h3>Variants tested</h3>
<table><thead><tr><th>Variant</th><th>Median, MAD and reference taken over</th><th>Result</th></tr></thead><tbody>
<tr><td>kept (used)</td><td>clients not isolated</td><td>best: sign flip +8.3, label +3.1, backdoor 79% → 21%</td></tr>
<tr><td>full population (3c)</td><td>every client not banned, isolated included:
\\(z_i = (s_i - \\operatorname{{med}}_{{\\mathcal A}} s)/\\max(1.4826\\,\\mathrm{{MAD}}_{{\\mathcal A}} s,\\ 0.1|\\operatorname{{med}}_{{\\mathcal A}} s|)\\)</td>
<td>more false isolations; let the last backdoor attacker stay in</td></tr>
<tr><td>readmission (3b)</td><td>kept; isolated clients return once E&nbsp;&gt;&nbsp;−0.75</td><td>attackers came back late in training</td></tr>
<tr><td>cap (3b)</td><td>kept; at most a third left out per round</td><td>backdoor attackers put back</td></tr>
</tbody></table>
"""),
    ("followup", "Follow-up screening (2026-11-12)", f"""
<p>Two options, both off by default, plus per-round logging of the raw scores, median, MAD and which term set the
scale.</p>
{mmd("5_followup_screening")}
<div class="formula">\\[ \\text{{scale}} = \\max\\big(1.4826\\cdot\\mathrm{{MAD}}(s),\\; 0.1\\,|\\operatorname{{med}}(s)|,\\; f_{{abs}}\\big) \\]
with the absolute floor \\(f_{{abs}} \\in \\{{0, 0.01, 0.03\\}}\\). <em>Patience k</em>: isolate only when
\\(E_i < -1.5\\) and \\(z_i < -1.5\\) in each of the last <i>k</i> rounds.</div>
"""),
    ("why", "Why honest z-scores get extreme", f"""
<p>The professor asked why the MAD shrinks and z drops. The historical runs did not log raw scores, but two
logged quantities identify the mechanism across all 15 FedHEP+ runs:</p>
<ul class="points">
<li><strong>Which term set the scale.</strong> Without the floor, the MAD of a round's z-scores is exactly
1/1.4826&nbsp;=&nbsp;0.674. A smaller value means the floor set the scale that round.</li>
<li><strong>The spread of cosines.</strong> SCCF's weights are softmax(c/T), so T·ln&nbsp;w recovers each cosine up to a
constant. This is a proxy: SCCF uses a mean-direction reference rather than the median.</li>
</ul>
{img("why_z_grows.png", "FedHEP+, 5 settings × 3 seeds: what sets z's scale, cosine spread, and when false isolations happen")}
<p><strong>The MAD does not shrink; the consensus does.</strong> The floor sets the scale in rounds 1–13. As each
client's update follows its own data, the spread of cosines doubles and the median cosine falls. In rounds 14–18
the floor stops binding in every run, and false isolations peak in exactly those rounds. The fix to test is a
floor that does not fall with the median.</p>
{mmd("7_why_z_grows")}
"""),
    ("drill", "Drill-down (Phases 4 and 4b)", f"""
{mmd("6_drill_down")}
<div class="formula">Parent score \\(g = \\cos(\\tfrac{{1}}{{|P|}}\\sum_{{i\\in P}}\\Delta_i,\\ \\text{{ref}})\\), compared with 64
random groups of the same size: \\(z_P = (g - \\operatorname{{med}}\\,\\text{{null}}) / \\max(1.4826\\,\\mathrm{{MAD}}\\,\\text{{null}}, 0.01|\\operatorname{{med}}\\,\\text{{null}}|)\\).
<br>v2 member check: \\(r_i = \\text{{signal}}_i - \\text{{ref}}\\), \\(a_i = \\cos\\big(r_i, \\tfrac{{1}}{{|P|-1}}\\sum_{{j\\in P, j\\ne i}} r_j\\big)\\),
standardised against clients outside the parent.</div>
<p class="note">Neither version was adopted. Honest members of a FedHEP parent are genuinely alike, so the member
check also blames honest groups.</p>
"""),
    ("rules", "Comparison methods", f"""
<p><strong>No defense:</strong> FedAvg (one ResNet-9, weighted average of all uploads) and undefended FedHEP.
<strong>Multi-Krum</strong> (authors' setting): score each update by the summed squared distance to its
<i>n − f − 2</i> nearest others, and average the 3 best. <strong>Phase 5</strong> puts each established rule in
FedHEP's Global step, on weight updates; the parents run SCCF over the clients the rule accepts.</p>
{mmd("8_phase5_rules")}
<div class="formula">
\\[ \\textbf{{Trimmed mean:}}\\ \\ \\text{{step}}_k = \\frac{{1}}{{n-2b}}\\sum_{{j=b+1}}^{{n-b}} \\Delta_{{(j),k}}\\ \\ (b = 3) \\qquad
\\textbf{{FLTrust:}}\\ \\ \\text{{step}} = \\frac{{\\sum_i \\mathrm{{ReLU}}(\\cos(\\Delta_i, g_0))\\,\\frac{{\\|g_0\\|}}{{\\|\\Delta_i\\|}}\\Delta_i}}{{\\sum_i \\mathrm{{ReLU}}(\\cos(\\Delta_i, g_0))}} \\]
\\[ \\textbf{{FLAME:}}\\ \\ S = \\operatorname{{med}}_i \\|\\Delta_i\\|, \\quad
\\text{{step}} = \\frac{{1}}{{|A|}}\\sum_{{i\\in A}} \\Delta_i \\min\\Big(1, \\frac{{S}}{{\\|\\Delta_i\\|}}\\Big) + \\mathcal{{N}}(0, (0.001\\,S)^2) \\]
\\[ \\textbf{{Bulyan:}}\\ \\ n \\ge 4f+3;\\ \\text{{pick }} \\theta = n - 2f \\text{{ by repeated Krum, then per coordinate average the }} \\beta = \\theta - 2f \\text{{ nearest the median}} \\]
</div>
{img("phase5_three_seeds.png", "Phase 5, three seeds: accuracy and backdoor success for every method")}
"""),
    ("attacks", "Attacks", """
<div class="formula">With \\(w\\) the round-start Global model and \\(\\Delta_i\\) the attacker's honestly trained update:
\\[ \\text{sign flip: } w - 1.5\\,\\Delta_i \\qquad \\text{gradient ascent: } w - 5\\,\\Delta_i \\qquad
\\text{random noise: } w + \\mathcal{N}(0, 2^2) \\]
\\[ \\text{ALIE: } w + \\mu - z\\,\\sigma, \\quad z = \\Phi^{-1}\\!\\Big(\\frac{n-s}{n}\\Big),\\ s = \\Big\\lfloor\\frac n2 + 1\\Big\\rfloor - m
\\quad (z = 0.43 \\text{ for } n = 15, m = 3) \\]
\\[ \\text{Min-Max: } w + \\mu - \\gamma\\,\\sigma, \\quad \\gamma = \\max\\Big\\{\\gamma : \\max_j \\|\\mu - \\gamma\\sigma - \\Delta_j\\| \\le \\max_{j,k}\\|\\Delta_j - \\Delta_k\\|\\Big\\} \\]
</div>
<ul class="points">
<li><strong>Label flip:</strong> train on label C − 1 − y instead of y.</li>
<li><strong>Backdoor (BadNets):</strong> a 3×3 white corner patch on 50% of the attacker's images, relabelled as
class 0. "Boosted" scales the update ×3.</li>
<li><strong>ALIE:</strong> μ and σ are the per-weight mean and standard deviation of the attackers' own honest
updates, chosen to sit inside the honest spread.</li>
<li><strong>Min-Max:</strong> uses the honest updates' statistics and goes as far as possible while never looking
further from anyone than two honest clients do from each other.</li>
</ul>
"""),
    ("metrics", "Metrics", """
<div class="formula">Per round, with flagged set <i>F</i> (left out or isolated), attackers <i>A</i> and honest clients <i>H</i>:
\\[ \\text{precision} = \\frac{|F\\cap A|}{|F|}, \\qquad \\text{recall} = \\frac{|F\\cap A|}{|A|}, \\qquad
\\text{false-isolation rate} = \\frac{|F\\cap H|}{|H|} \\]</div>
<p>Accuracy is pooled over all honest clients' own test images, each predicted by its own personalised ensemble.
Wrongly isolated honest clients are included; attackers are not scored. Backdoor success is the share of
triggered test images, not originally of class 0, that the model labels class 0.</p>
"""),
]


def main() -> None:
    toc = "".join(f'<a href="#{sid}">{title}</a>' for sid, title, _ in SECTIONS)
    body = "".join(f'<section id="{sid}"><h2>{title}</h2>{content}</section>' for sid, title, content in SECTIONS)
    template = (Path(__file__).with_name("template.html")).read_text()
    OUT.write_text(template.replace("{{TOC}}", toc).replace("{{BODY}}", body))


main()
