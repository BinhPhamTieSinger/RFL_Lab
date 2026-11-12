"""Audit existing logs and plot raw MAD where available; never infer MAD from z.
Run from RFL_Lab: python scripts/diagnose_screening.py outputs --filter p3b__zm
Label skew here is offline ground truth, never used by the defense.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rfl.data.cifar100 import seeded_subsets  # noqa: E402
from rfl.data.partition import make_partitioner  # noqa: E402


def label_stats(cfg, metrics):
    if metrics.get('client_label_stats'):
        return {x['client']: x for x in metrics['client_label_stats']}
    if cfg['data']['dataset'] != 'cifar100':
        return {}
    root = Path(cfg['data']['root'])
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    with (root / 'cifar-100-python/train').open('rb') as f:
        labels = np.asarray(pickle.load(f, encoding='bytes')[b'fine_labels'])
    d, seed = cfg['data'], metrics['seed']
    tr, _ = seeded_subsets(len(labels), 10000, d['train_subset'], d['test_subset'], seed)
    parts = make_partitioner(d['alpha']).split(labels[tr], d['num_clients'], seed)
    result = {}
    for cid, idx in enumerate(parts):
        counts = np.bincount(labels[tr][idx], minlength=100)
        p = counts[counts > 0] / max(1, len(idx))
        result[cid] = {'label_entropy': float(-(p * np.log(p)).sum()), 'classes': len(p)}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('outputs', type=Path)
    ap.add_argument('--filter', default='p3b__zm')
    ap.add_argument('--out', type=Path, default=Path('docs/followup/legacy'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, clients = [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    mad_runs = 0
    for file in sorted(a.outputs.glob('*/metrics.json')):
        if a.filter not in file.parent.name:
            continue
        m = json.loads(file.read_text())
        cfg = yaml.safe_load((file.parent / 'config.yaml').read_text())
        ds = m.get('diagnostics') or []
        if not ds:
            continue
        attackers = set(m['attackers'])
        honest = set(range(len(m['client_train_sizes']))) - attackers
        isolated = set(ds[-1].get('isolated', []))
        flagged = set(ds[-1].get('excluded', m['detection'][-1]['flagged']))
        tp, fp = len(flagged & attackers), len(flagged & honest)
        rows.append({'run': file.parent.name, 'accuracy': m['final']['pooled'],
                     'precision': tp / len(flagged) if flagged else None,
                     'recall': tp / len(attackers) if attackers else None,
                     'false_exclusion_rate': fp / len(honest),
                     'false_isolation_rate': len(isolated & honest) / len(honest),
                     'isolated_honest': sorted(isolated & honest),
                     'asr': m['history'][-1].get('asr'),
                     'mad_available': bool(ds[-1].get('score_stats'))})
        stats = label_stats(cfg, m)
        for c in sorted(honest):
            entry = stats.get(c, {})
            entropy = entry.get('label_entropy')
            classes = 100 if cfg['data']['dataset'] == 'cifar100' else 10
            clients.append({'run': file.parent.name, 'client': c,
                            'n_train': m['client_train_sizes'][c], 'classes': entry.get('classes'),
                            'entropy': entropy,
                            'r_skew': (np.exp(entropy) - 1) / (classes - 1) if entropy is not None else None,
                            'isolated': c in isolated,
                            'first_isolation': next((x['round'] for x in ds
                                                     if c in x.get('isolated', [])), None)})
        logged = [d for d in ds if d.get('score_stats')]
        if logged:
            mad_runs += 1
            rounds = [d['round'] for d in logged]
            series = [[d['score_stats']['mad'] for d in logged],
                      [d['score_stats']['scale'] for d in logged],
                      [d['score_stats']['reference_n'] for d in logged],
                      [len(set(d['isolated']) & honest) / len(honest) for d in logged]]
            for ax, values in zip(axes.flat, series, strict=True):
                ax.plot(rounds, values, label=file.parent.name)
    titles = ['Raw MAD', 'Effective denominator', 'Reference population size', 'False-isolation rate']
    for ax, title in zip(axes.flat, titles, strict=True):
        ax.set_title(title)
        ax.set_xlabel('Round')
        ax.grid(alpha=.2)
        if mad_runs:
            ax.legend(fontsize=5)
    if mad_runs:
        fig.tight_layout()
        fig.savefig(a.out / 'mad_per_round.png', dpi=160)
    plt.close(fig)
    for name, data in [('runs', rows), ('honest_clients', clients)]:
        (a.out / f'{name}.json').write_text(json.dumps(data, indent=2))
        if data:
            with (a.out / f'{name}.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    print(f'{len(rows)} runs audited; {mad_runs} with MAD. Written to {a.out}')
    for isolated in [False, True]:
        subset = [c for c in clients if c['isolated'] == isolated and c['r_skew'] is not None]
        if subset:
            print(f'isolated={isolated}: n={len(subset)}, mean r_skew='
                  f'{np.mean([c["r_skew"] for c in subset]):.4f}, mean n_train='
                  f'{np.mean([c["n_train"] for c in subset]):.1f}')


if __name__ == '__main__':
    main()
