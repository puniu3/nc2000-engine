import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=Path, default=Path('findings/operational-early-perish-v1'))
args = parser.parse_args()
protocol = json.loads((args.dir / 'protocol.json').read_text())
summarize = runpy.run_path('tools/analyze-operational-combo-gate.py')['summarize']
report = {'complete': True, 'default_enabled': False, 'results': {}}
for cell, spec in protocol['cells'].items():
    path = args.dir / f'{cell}.jsonl'
    rows = sorted((json.loads(line) for line in path.read_text().splitlines()), key=lambda r: r['job'])
    assert [r['job'] for r in rows] == list(range(spec['games'])), cell
    for row in rows:
        assert row['seed'] == spec['seed'] and row['iterations'] == spec.get('iterations', protocol['iterations'])
        assert row['implementation'] == 'early-perish-v1' and row['ucb_c'] == protocol['ucb_c']
        for arm in ['a', 'b', 'c']:
            assert row[arm]['policy'] == protocol['arms']['a' if cell == 'same' else arm]
        if cell in ['same', 'onpool']:
            norm = lambda arm: {k: v for k, v in arm.items() if k not in ['elapsed_seconds', 'policy']}
            assert norm(row['a']) == norm(row['b']) == norm(row['c']), (cell, row['job'])
    comparisons = {}
    for first, second in [('a', 'b'), ('a', 'c'), ('b', 'c')]:
        paired = [dict(row, a=row[first], b=row[second]) for row in rows]
        result = summarize(paired)
        diffs = [r[second]['score'] - r[first]['score'] for r in rows
                 if r[first]['terminal'] and r[second]['terminal'] and r[first]['score'] != r[second]['score']]
        n = len(diffs)
        k = min(sum(d > 0 for d in diffs), sum(d < 0 for d in diffs))
        result['terminal_exact_two_sided_p'] = min(1., 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.
        comparisons[f'{second}-{first}'] = result
    arms = {a: {'wins': sum(r[a]['terminal'] and r[a]['score'] == 1 for r in rows),
                'caps': sum(not r[a]['terminal'] for r in rows),
                'early_exposed': sum(r[a]['early_exposed'] for r in rows),
                'early_combo_samples': sum(r[a]['early_combo_samples'] for r in rows),
                'opponent_bond_games': sum(r[a]['opponent_bond_moves'] > 0 for r in rows)} for a in ['a', 'b', 'c']}
    report['results'][cell] = {'triplets': len(rows), 'arms': arms, 'comparisons': comparisons,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
primary = report['results']['adaptive']['comparisons']
ordered = sorted(['b-a', 'c-a'], key=lambda c: primary[c]['terminal_exact_two_sided_p'])
adjusted = 0.
for rank, key in enumerate(ordered):
    adjusted = max(adjusted, min(1., (2 - rank) * primary[key]['terminal_exact_two_sided_p']))
    primary[key]['holm_two_comparisons_p'] = adjusted
report['demonstrated_adaptive_gain'] = {key: primary[key]['stratified_bootstrap95'][0] > 0
    and primary[key]['holm_two_comparisons_p'] < .05 for key in ['b-a', 'c-a']}
(args.dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({cell: {'wins': {a: x['wins'] for a, x in result['arms'].items()},
                       'differences': {key: {'delta': x['paired_delta'], 'ci95': x['stratified_bootstrap95'],
                         'p': x['terminal_exact_two_sided_p']} for key, x in result['comparisons'].items()}}
                  for cell, result in report['results'].items()}, indent=2))
