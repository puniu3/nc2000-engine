import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

out = Path('findings/operational-perish-bayes-v1')
protocol = json.loads((out / 'protocol.json').read_text())
source = Path('findings/operational-early-perish-v1/adaptive.jsonl')
rows = sorted(map(json.loads, source.read_text().splitlines()), key=lambda r: r['job'])
assert len(rows) == 128
assert all(r[a]['terminal'] and r[a]['score'] in [0, 1] for r in rows for a in 'abc')
patterns = np.array(list(itertools.product([0, 1], repeat=3)))
groups = {key: [r for r in rows if r['opponent_team'] == key] for key in sorted({r['opponent_team'] for r in rows})}
assert len(groups) == 8 and all(len(g) == 16 for g in groups.values())
counts = []
for group in groups.values():
    scores = np.array([[r[a]['score'] for a in 'abc'] for r in group])
    counts.append(np.array([np.all(scores == pat, axis=1).sum() for pat in patterns]))
results = {}
for mass in protocol['sensitivity_prior_mass_per_scenario']:
    rng = np.random.default_rng(protocol['seed'])
    wins = np.zeros((protocol['draws'], 3))
    for count in counts:
        wins += rng.dirichlet(count + mass / 8, size=protocol['draws']) @ patterns / 8
    comparisons = {}
    for first, second in [(0, 1), (0, 2), (1, 2)]:
        delta = wins[:, second] - wins[:, first]
        probability = float(np.mean(delta > 0))
        comparisons[f'{"abc"[second]}-{"abc"[first]}'] = {
            'probability_improved': probability,
            'monte_carlo_se': math.sqrt(probability * (1 - probability) / protocol['draws']),
            'posterior_mean_delta': float(delta.mean()),
            'credible95': np.quantile(delta, [.025, .975]).tolist(),
            'probability_loss_gt_1pp': float(np.mean(delta < -.01)),
            'probability_loss_gt_3pp': float(np.mean(delta < -.03)),
            'expected_downside': float(np.maximum(-delta, 0).mean()),
        }
    results[str(mass)] = {'comparisons': comparisons, 'arms': {
        a: {'posterior_mean_win_probability': float(wins[:, i].mean()),
            'probability_best': float(np.mean(np.argmax(wins, axis=1) == i)),
            'expected_regret': float(np.mean(np.max(wins, axis=1) - wins[:, i]))}
        for i, a in enumerate('abc')}}
normal = {}
for first, second in [('a', 'b'), ('a', 'c')]:
    diffs = [[r[second]['score'] - r[first]['score'] for r in g] for g in groups.values()]
    mean = statistics.mean(statistics.mean(g) for g in diffs)
    variance = sum(statistics.variance(g) / len(g) for g in diffs) / len(diffs) ** 2
    normal[f'{second}-{first}'] = {'observed_mean': mean, 'estimated_sampling_se': math.sqrt(variance), 'priors': {}}
    for sd in [.01, .03, .05, .1]:
        post_var = 1 / (1 / variance + 1 / sd ** 2)
        post_mean = post_var * mean / variance
        normal[f'{second}-{first}']['priors'][str(sd)] = {
            'posterior_mean_delta': post_mean,
            'posterior_sd': math.sqrt(post_var),
            'probability_improved': statistics.NormalDist(post_mean, math.sqrt(post_var)).cdf(0) * -1 + 1,
        }
report = {'estimand': protocol['estimand'], 'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'patterns': patterns.tolist(), 'counts_by_opponent': {key: count.tolist() for key, count in zip(groups, counts)},
          'primary_prior_mass': 1, 'primary': results['1'], 'dirichlet_sensitivity': results,
          'normal_approximation_sensitivity': normal, 'draws': protocol['draws'], 'seed': protocol['seed']}
(out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'primary': report['primary'], 'normal_sensitivity': normal}, indent=2))
