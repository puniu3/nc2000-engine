import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=Path, default=Path('tmp/retain-perish/loss-review'))
args = parser.parse_args()
out = Path('findings/operational-retain-perish-v1/loss-review')
out.mkdir(exist_ok=True)
read_lines = lambda path: [json.loads(s) for s in path.read_text().splitlines()]
original = {r['job']: r for r in read_lines(out.parent / 'adaptive.jsonl')}
replayed = read_lines(args.dir / 'games.jsonl')
jobs = sorted(r['job'] for r in original.values() if r['b']['score'] < r['a']['score'])
assert sorted(r['job'] for r in replayed) == jobs
norm = lambda x: {k: v for k, v in x.items() if k != 'elapsed_seconds'}
for row in replayed:
    for arm in ['a', 'b']:
        assert norm(row[arm]) == norm(original[row['job']][arm])
shutil.copy2(args.dir / 'games.jsonl', out / 'games.jsonl')
cases = []
for job in jobs:
    names = [f'{job}-layered-meta-rentals-learnset-v1', f'{job}-retain-perish-v1']
    traces = [read_lines(args.dir / 'traces' / f'{name}.jsonl') for name in names]
    a, b = next((a, b) for a, b in zip(*traces) if a['choice'] != b['choice'])
    assert a['state_key'] == b['state_key']
    for name, row in zip(names, [a, b]):
        (out / f'{name}.jsonl').write_text(json.dumps(row) + '\n')
    replay = json.loads((args.dir / 'traces' / f'{names[0]}-replay.json').read_text())
    end = next(i for i, x in enumerate(replay['inputs']) if x['key'] == a['state_key'])
    while end < len(replay['inputs']) and replay['inputs'][end]['key'] == a['state_key']:
        end += 1
    replay = {k: replay[k] for k in ['teams', 'battle_seed', 'inputs']}
    replay['inputs'] = replay['inputs'][:end]
    (out / f'{names[0]}-replay.json').write_text(json.dumps(replay) + '\n')
    roots = [r for r in read_lines(args.dir / 'probes' / f'{job}.jsonl') if r['kind'] == 'root']
    by = {v: {r['seed']: r for r in roots if r['variant'] == v} for v in ['baseline', 'retain-perish']}
    assert by['baseline'].keys() == by['retain-perish'].keys()
    assert len(by['baseline']) == (8 if a['seed'] == b['seed'] else 9)
    for variant, row in [('baseline', a), ('retain-perish', b)]:
        assert by[variant][row['seed']]['root'] == row['root']
    cases.append({'job': job, 'turn': a['turn'], 'original_choices': [a['choice'], b['choice']],
                  'original_seeds_equal': a['seed'] == b['seed'], 'paired_seeds': len(by['baseline']),
                  'identical_roots': sum(by['baseline'][s]['root'] == by['retain-perish'][s]['root'] for s in by['baseline']),
                  'choices': {v: dict(Counter(r['choice'] for r in rs.values())) for v, rs in by.items()},
                  'model_keys': {v: sorted({r['model_key'] for r in rs.values()}) for v, rs in by.items()},
                  'actual_opponent_reply': [x['choice'] for x in replay['inputs'] if x['key'] == a['state_key'] and x['side'] != a['side']]})
    shutil.copy2(args.dir / 'probes' / f'{job}.jsonl', out / f'{job}-probe.jsonl')
diffs = [r['b']['score'] - r['a']['score'] for r in original.values()]
z = statistics.NormalDist().inv_cdf(.975) + statistics.NormalDist().inv_cdf(.8)
summary = {'selection': 'All four worsened pairs in the frozen adaptive trial; post-outcome diagnosis, not new win-rate evidence.',
           'full_games_reproduced': 8, 'root_searches': sum(c['paired_seeds'] * 2 for c in cases), 'cases': cases,
           'outcome_accounting': 'Keep all 6 improved and 4 worsened pairs; do not discount losses based on this diagnosis.',
           'candidate_status': 'retained for further evaluation; superiority unproven; production unchanged',
           'planning_scenario': {'assumed_true_delta': statistics.mean(diffs), 'assumed_variance': statistics.variance(diffs),
                                 'two_sided_alpha': .05, 'power': .8,
                                 'normal_approximation_pairs': round(statistics.variance(diffs) * z * z / statistics.mean(diffs) ** 2),
                                 'limitation': 'Illustrative independent-pair normal approximation using a noisy pilot estimate; not a power guarantee or a new stopping rule.'}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
