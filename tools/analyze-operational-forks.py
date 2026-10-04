#!/usr/bin/env python3
import collections
import hashlib
import json
from pathlib import Path

base = Path('findings/operational-combo-validation-v1')
cases = json.loads((base / 'forks.json').read_text())
summary = {'cases': [], 'roots': 0, 'exact_steps': 0, 'unresolved_steps': 0}
for case in cases:
    job = case['job']
    path = base / 'probes' / f'{job}.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    roots = [x for x in rows if x['kind'] == 'root']
    exact = [x for x in rows if x['kind'] == 'exact_step']
    nseeds = 8 if case['baseline']['seed'] == case['conditioned']['seed'] else 9
    assert len(roots) == nseeds * (6 if case['baseline']['turn'] == 2 else 2)
    assert len(exact) == 12
    variants = {}
    for v in sorted({x['variant'] for x in roots}):
        rs = [x for x in roots if x['variant'] == v]
        assert len({r['seed'] for r in rs}) == nseeds
        variants[v] = {'choices': dict(collections.Counter(x['choice'] for x in rs)),
                       'model_moves': rs[0]['model_moves'], 'model_item': rs[0]['model_item'],
                       'at_original_seeds': {str(r['seed']): r['choice'] for r in rs
                                             if r['seed'] in [case['baseline']['seed'], case['conditioned']['seed']]}}
    summary['cases'].append({'job': job, 'turn': case['baseline']['turn'],
                             'same_original_seed': nseeds == 8, 'variants': variants,
                             'exact_steps': exact, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    summary['roots'] += len(roots)
    summary['exact_steps'] += len(exact)
    summary['unresolved_steps'] += sum(x['stats'] is None for x in exact)

norm = lambda s: ''.join(c.lower() for c in s if c.isalnum())
prior = json.loads(Path('data/belief-pool-v3/belief-pool.json').read_text())
weights = collections.Counter()
total = 0
rows = 0
for team in prior['teams']:
    for s in team['sets']:
        moves = list(map(norm, s['moves']))
        if norm(s['species']) == 'misdreavus' and 'meanlook' in moves:
            w = team.get('weight', 1)
            total += w
            rows += 1
            for move in moves:
                weights[move] += w
summary['conditional_prior'] = {'rows': rows, 'weight': total,
    'move_mass': {k: v / total for k, v in weights.items()},
    'interpretation': 'Empirical prior weights conditional on Mean Look only; not calibrated human usage probabilities.'}
item_path = base / 'probes' / '25-item.jsonl'
if item_path.exists():
    item = [json.loads(s) for s in item_path.read_text().splitlines()]
    roots = [x for x in item if x['kind'] == 'root']
    assert len(roots) == 16
    summary['item_followup'] = {v: dict(collections.Counter(x['choice'] for x in roots if x['variant'] == v))
                                for v in sorted({x['variant'] for x in roots})}
(base / 'fork-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','conditional_prior']},indent=2))
