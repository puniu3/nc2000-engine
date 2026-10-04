import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=Path, default=Path('findings/operational-retain-perish-v1'))
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
        assert row['implementation'] == 'retain-perish-v1'
        for arm in ['a', 'b']:
            assert row[arm]['policy'] == protocol['arms']['a' if cell == 'same' else arm]
        if cell in ['same', 'onpool']:
            norm = lambda arm: {k: v for k, v in arm.items() if k not in ['elapsed_seconds', 'policy']}
            assert norm(row['a']) == norm(row['b']), (cell, row['job'])
    result = summarize(rows)
    discordant = [r['b']['score'] - r['a']['score'] for r in rows
                  if r['a']['terminal'] and r['b']['terminal'] and r['a']['score'] != r['b']['score']]
    n = len(discordant)
    k = min(sum(x > 0 for x in discordant), sum(x < 0 for x in discordant))
    result['terminal_exact_two_sided_p'] = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    for arm in ['a', 'b']:
        result[arm]['opponent_bond_games'] = sum(r[arm]['opponent_bond_moves'] > 0 for r in rows)
        result[arm]['opponent_switch_events'] = sum(r[arm]['opponent_switch_events'] for r in rows)
        result[arm]['total_turns'] = sum(r[arm]['turns'] for r in rows)
    result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['results'][cell] = result
primary = report['results']['adaptive']
report['primary_passed'] = primary['stratified_bootstrap95'][0] > 0 and primary['terminal_exact_two_sided_p'] < .05
(args.dir / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: {field: v[field] for field in ['paired_delta', 'stratified_bootstrap95', 'terminal_exact_two_sided_p', 'same_trajectory_pairs']}
                  for k, v in report['results'].items()}, indent=2))
