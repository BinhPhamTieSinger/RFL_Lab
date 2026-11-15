import math
from pathlib import Path

from rfl.experiment.config import load_config, to_dict
from rfl.experiment.registry import METHODS

LAB = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "data.root": str(LAB.parent / "Paper_Reproduction/data"),
    "data.train_subset": 600,
    "data.test_subset": 300,
    "data.num_clients": 6,
    "train.local_epochs": 1,
    "run.rounds": 1,
    "run.eval_every": 1,
    "run.device": "cpu",
}
OPTIONS = {
    "data.dataset": ["cifar100", "synthetic"],
    "run.device": ["cpu", "auto", "mps", "cuda", "xpu"],
    "attack.kind": [
        None,
        "label_flip",
        "sign_flip",
        "gradient_ascent",
        "random_noise",
        "backdoor",
        "alie",
        "min_max",
    ],
    "attack.placement": ["random", "concentrated", "spread"],
    "attack.knowledge": ["colluders", "omniscient"],
    "plus.signal": ["root_head", "weights"],
    "plus.reference": ["mean_direction", "median"],
    "plus.population": ["kept", "all"],
    "plus.drill_mode": ["exclude", "evidence"],
    "plus.global_rule": ["sccf", "arc_nnm_tm"],
    "fedhep.similarity_on": ["full", "weights"],
    "robust.rule": ["median", "trimmed_mean", "bulyan", "flame", "fltrust", "arc_nnm_tm"],
    "robust.flame_on": ["models", "updates"],
}
BASELINES = {
    "fedprox": ["fedprox_mu"],
    "ditto": ["ditto_lambda"],
    "multikrum": ["krum_selected"],
    "fedrep": ["fedrep_head_epochs"],
}


def schema(method):
    if method not in METHODS:
        raise ValueError("Unknown method")
    cfg = to_dict(load_config(overrides=DEFAULTS))
    sections = ["data", "train", "run", "attack"]
    if method.startswith("fedhep"):
        sections.append("fedhep")
    if method == "fedhep_plus":
        sections.append("plus")
    if method == "fedhep_robust":
        sections.append("robust")
    if method in BASELINES:
        sections.append("baseline")
    out = []
    for section in sections:
        for key, value in cfg[section].items():
            path = f"{section}.{key}"
            if path in ("data.root", "run.output_dir"):
                continue
            if method == "scaffold" and path in ("train.momentum", "train.nesterov"):
                continue
            if path == "fedhep.similarity_on" and method != "fedhep":
                continue
            if section == "baseline" and key not in BASELINES[method]:
                continue
            if not method.startswith("fedhep") and path in ("attack.placement", "attack.stop_round"):
                continue
            opts = OPTIONS.get(path)
            if path == "attack.kind" and not method.startswith("fedhep"):
                opts = opts[:-2]
            out.append(
                {
                    "key": path,
                    "default": value,
                    "options": opts,
                    "type": "boolean"
                    if isinstance(value, bool)
                    else "number"
                    if isinstance(value, (int, float)) or value is None and path not in OPTIONS
                    else "string",
                }
            )
    return out


def validate(method, overrides):
    allowed = {f["key"] for f in schema(method)}
    if set(overrides) - allowed:
        raise ValueError(f"Unsupported fields: {sorted(set(overrides) - allowed)}")
    settings = {
        **DEFAULTS,
        **overrides,
        "method": method,
        "model": "resnet9_multihead" if method.startswith("fedhep") else "resnet9",
    }
    for f in schema(method):
        v = settings.get(f["key"], f["default"])
        if f["options"] is not None and v not in f["options"]:
            raise ValueError(f"Invalid {f['key']}")
        if f["type"] == "boolean" and not isinstance(v, bool):
            raise ValueError(f"Invalid {f['key']}")
        if f["type"] == "number" and v is not None:
            if isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v):
                raise ValueError(f"Invalid {f['key']}")
            if isinstance(f["default"], int) and not isinstance(v, int):
                raise ValueError(f"Integer required for {f['key']}")
            if v < 0:
                raise ValueError(f"Negative {f['key']}")
    c = load_config(overrides=settings)
    for key, v, low, high in [
        ("clients", c.data.num_clients, 2, 100),
        ("rounds", c.run.rounds, 1, 1000),
        ("epochs", c.train.local_epochs, 1, 100),
        ("batch size", c.train.batch_size, 1, 4096),
        ("train subset", c.data.train_subset, 1, 50000),
        ("test subset", c.data.test_subset, 1, 10000),
        ("eval interval", c.run.eval_every, 1, 1000),
        ("parents", c.fedhep.num_groups, 1, c.data.num_clients),
    ]:
        if v is None or not low <= v <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
    if c.data.alpha is not None and c.data.alpha <= 0:
        raise ValueError("alpha must be positive or null for IID")
    if c.train.lr <= 0 or not 0 <= c.train.momentum < 1:
        raise ValueError("Invalid optimizer settings")
    if c.train.nesterov and c.train.momentum == 0:
        raise ValueError("Nesterov requires momentum")
    if not 0 <= c.attack.rate <= 1 or not 0 <= c.attack.backdoor_fraction <= 1:
        raise ValueError("Invalid attack fraction")
    if c.attack.count is not None and not 0 <= c.attack.count < c.data.num_clients:
        raise ValueError("Attack count must leave an honest client")
    if not 0 <= c.plus.evidence_decay < 1 or not 0 <= c.plus.max_excluded <= 1:
        raise ValueError("Invalid evidence decay or cap")
    if min(c.plus.exclusion_patience, c.plus.isolation_patience) < 1:
        raise ValueError("Patience must be positive")
    if not 0 <= c.robust.momentum < 1:
        raise ValueError("Invalid worker momentum")
    if method == "fedhep_robust":
        f, n = c.robust.assumed_attackers, c.data.num_clients
        if c.robust.rule == "bulyan" and n < 4 * f + 3:
            raise ValueError("Bulyan requires n >= 4f + 3")
        if c.robust.rule in ("trimmed_mean", "arc_nnm_tm") and n <= 2 * f:
            raise ValueError("Rule requires n > 2f")
    if c.attack.count is not None and not isinstance(c.attack.count, int):
        raise ValueError("Attack count must be an integer")
    if c.run.seed > 2**32 - 1:
        raise ValueError("Seed must fit a NumPy uint32")
    if c.attack.kind and c.attack.count is None and c.attack.rate == 1:
        raise ValueError("Leave at least one honest client for evaluation")
    if c.data.test_subset < c.data.num_clients:
        raise ValueError("Test subset must contain at least as many examples as clients")
    if c.attack.placement == "concentrated" and c.fedhep.num_groups == 1:
        raise ValueError("Concentrated placement requires at least two parents")
    if c.attack.backdoor_target >= (100 if c.data.dataset == "cifar100" else 10):
        raise ValueError("Backdoor target exceeds dataset class range")
    if c.plus.global_rule == "arc_nnm_tm" and c.data.num_clients <= 2 * c.plus.global_f:
        raise ValueError("ARC/NNM trimmed mean requires n > 2f")
    return c


FORMULAS = [
    {
        "name": "Screening scale",
        "method": "fedhep_plus",
        "latex": (
            r"z_i=\frac{s_i-\operatorname{median}(s)}"
            r"{\max(1.4826\operatorname{MAD}(s),\rho|\operatorname{median}(s)|,a)}"
        ),
        "explanation": (
            "s is the cosine score. rho is relative_floor and a is mad_floor. Fewer "
            "than five reference senders produce zero z. Runtime inputs appear in round diagnostics."
        ),
        "source": "defense/screening.py",
    },
    {
        "name": "Temporal evidence",
        "method": "fedhep_plus",
        "latex": r"E_{i,t}=d E_{i,t-1}+(1-d)z_{i,t}",
        "explanation": (
            "d is evidence_decay. Isolation also depends on evidence_cut, persistence, "
            "readmission and configured gates."
        ),
        "source": "defense/screening.py",
    },
    {
        "name": "Sample-weighted mean",
        "method": "fedavg",
        "latex": r"\Delta=\frac{\sum_i n_i\Delta_i}{\sum_i n_i}",
        "explanation": (
            "n is client training sample count. Applied by FedAvg. Other aggregation rules have different weights."
        ),
        "source": "aggregation/fedavg.py",
    },
    {
        "name": "Label distribution",
        "method": "fedhep",
        "latex": r"r_{skew}=\frac{\exp(-\sum_c p_c\log p_c)-1}{C-1}",
        "explanation": (
            "p is the true training-label frequency. Smaller values mean more "
            "concentrated labels. Client memory records the actual result."
        ),
        "source": "client/fedhep.py",
    },
    {
        "name": "Learning rate",
        "method": "all",
        "latex": r"\eta_r=\eta_{min}+\frac{\eta_0-\eta_{min}}2(1+\cos(\pi(r-1)/(R-1)))",
        "explanation": (
            "r is the round and R the configured number of rounds. For R=1 the implementation returns eta_0."
        ),
        "source": "client/sgd.py",
    },
]

FORMULAS.extend(
    [
        {
            "name": "Proximal objective",
            "method": "fedprox",
            "latex": r"\mathcal L=\mathrm{CE}(f_w(x),y)+\frac\mu2\|w-w_{received}\|_2^2",
            "explanation": (
                "The anchor contains model parameters. Weight decay and Nesterov remain optimizer settings."
            ),
            "source": "client/fedprox.py",
        },
        {
            "name": "Personal objective",
            "method": "ditto",
            "latex": r"\mathcal L_p=\mathrm{CE}(f_{w_p}(x),y)+\frac\lambda2\|w_p-w_{received}\|_2^2",
            "explanation": (
                "Ditto first trains and uploads the shared model, then trains its personal model "
                "against the received anchor."
            ),
            "source": "client/ditto.py",
        },
        {
            "name": "Control variate update",
            "method": "scaffold",
            "latex": r"g_i\leftarrow g_i+c-c_i,\quad \Delta c_i=-c+\frac{w_{start}-w_{end}}{K\eta}",
            "explanation": (
                "K counts actual optimizer steps. The code uses max(1,K) and a small-step guard. Momentum is zero."
            ),
            "source": "client/scaffold.py",
        },
        {
            "name": "Private classifier upload",
            "method": "fedrep",
            "latex": r"u_{head}=0,\quad \Delta=u-w_{start}",
            "explanation": (
                "The full flat state is uploaded with classifier coordinates zeroed. Personal "
                "evaluation uses the trained head."
            ),
            "source": "client/fedrep.py",
        },
        {
            "name": "Krum selection",
            "method": "multikrum",
            "latex": r"s_i=\sum_{j\in N_i}\|\Delta_i-\Delta_j\|_2^2",
            "explanation": (
                "N_i contains the nearest other updates with k clamped by krum_scores. The m lowest "
                "scores are averaged. At most two senders are simply averaged."
            ),
            "source": "aggregation/multikrum.py",
        },
        {
            "name": "Three-head objective",
            "method": "fedhep",
            "latex": r"\mathcal L=m_r\mathrm{CE}_r+m_p\mathrm{CE}_p+m_l\mathrm{CE}_{l,w}",
            "explanation": (
                "Parent and local logits mask absent classes. Local CE uses inverse-square-root class "
                "weights. Head weights are normalized binomial_head_weights."
            ),
            "source": "client/fedhep.py",
        },
        {
            "name": "SCCF aggregation",
            "method": "fedhep",
            "latex": r"\Delta_{parameters}=\sum_i t_i\,\mathrm{bound}(\Delta_i),\quad t=\mathrm{softmax}(s/T)",
            "explanation": (
                "Applies to SCCF calls in defended variants. T adapts to valid-score variance. Weak "
                "consensus and all-rejected cases use uniform trust. Buffers use coordinate median."
            ),
            "source": "defense/soft_cosine.py",
        },
        {
            "name": "Robust aggregation rules",
            "method": "fedhep_robust",
            "latex": r"\mathrm{TM}_f(X)_j=\frac{1}{n-2f}\sum_{i=f+1}^{n-f}X_{(i),j}",
            "explanation": (
                "Shown for the trimmed-mean rule. ARC/NNM applies clipping and nearest-neighbour "
                "mixing first. Median, Bulyan, FLAME and FLTrust are separately implemented in the "
                "linked source. The runtime event names the selected rule."
            ),
            "source": "defense/classic.py",
        },
    ]
)

FORMULAS.append(
    {
        "name": "Ensemble mixing update",
        "method": "fedhep",
        "latex": (
            r"w=\mathrm{softmax}(\alpha),\quad p=\mathrm{softmax}(\sum_h w_h z_h),"
            r"\quad g_h=\sum_{b,c}\frac{p_{bc}-y_{bc}}{B}\frac{z_{h,bc}}{\max(\mathrm{mean}|z_l|,10^{-4})},"
            r"\quad \alpha'=\log(\beta\mathrm{softmax}(\alpha-\eta a g)+(1-\beta)\pi+10^{-8})"
        ),
        "explanation": (
            "Implemented _alpha_step. Head order is local, parent, root. z contains pre-optimizer logits. "
            "B is batch size, y is one-hot labels, pi is the heterogeneity prior, a is alpha_lr_scale "
            "and beta is prior_mix. First-batch inputs and outputs are captured. "
            "The source does not name this operation ASML."
        ),
        "source": "client/fedhep.py",
    }
)

FORMULAS.extend(
    [
        {
            "name": "ACLM · active-class logit masking",
            "method": "fedhep",
            "latex": (
                r"\widetilde z_{h,c}=\begin{cases}z_{h,c}&c\in Y_i\\-10^9&c\notin Y_i\end{cases},"
                r"\quad h\in\{parent,local\}"
            ),
            "explanation": (
                "The true client labels define Y_i. Parent and local cross-entropy receive masked logits. "
                "The root loss uses every class. The adapter captures actual pre-mask and post-mask logits "
                "for the first sample of the first batch."
            ),
            "source": "client/fedhep.py",
        },
        {
            "name": "TTT · temporal trust tracking",
            "method": "fedhep_defended",
            "latex": r"q_i' = 0.85 q_i + 0.15\min(1,n t_i),\quad q_i'<0.25\Rightarrow i\in I",
            "explanation": (
                "Defended FedHEP starts q_i at 1. Each parent and Global aggregation updates the ledger "
                "using its sender count n and SCCF weight t_i. Isolation persists. FedHEP+ replaces this "
                "ledger with its screening evidence rule."
            ),
            "source": "defense/trust_ledger.py",
        },
    ]
)
