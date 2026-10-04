import json
from pathlib import Path
import shutil

source = Path('tmp/early-perish/frozen')
replays = Path('tmp/early-perish/cluster-review')
out = Path('findings/operational-early-perish-v1')
detail = out / 'cluster-review'
detail.mkdir(exist_ok=True)
read = lambda p: [json.loads(s) for s in p.read_text().splitlines()]
original = {r['job']: r for r in read(source / 'adaptive.jsonl')}
cases = []
for job, row in sorted(original.items()):
    if row['opponent_team'] != 'kingdra-kit2' or row['a']['score'] <= row['c']['score']:
        continue
    logs = [(source / f'jobs/adaptive-{job}-{a}.log').read_text().splitlines() for a in 'ac']
    at = next(i for i, (a, c) in enumerate(zip(*logs)) if a != c)
    cases.append({'job': job, 'own_team': row['own_team'], 'scores': {a: row[a]['score'] for a in 'abc'},
                  'first_log_difference': {'line_index': at, 'a': logs[0][at], 'c': logs[1][at]},
                  'common_prefix': logs[0][:at]})
assert len(cases) == 6
norm = lambda x: {k: v for k, v in x.items() if k != 'elapsed_seconds'}
for arm in 'ac':
    rows = read(replays / f'{arm}.jsonl')
    assert sorted(r['job'] for r in rows) == [5, 21]
    for row in rows:
        assert norm(row[arm]) == norm(original[row['job']][arm])
    shutil.copy2(replays / f'{arm}.jsonl', detail / f'{arm}.jsonl')
for case in cases:
    job = case['job']
    if job not in [5, 21]:
        continue
    names = [f'{job}-{policy}' for policy in ['layered-meta-rentals-learnset-v1', 'early-perish-v1']]
    traces = [read(replays / 'traces' / f'{name}.jsonl') for name in names]
    a, c = next((a, c) for a, c in zip(*traces) if a['choice'] != c['choice'])
    assert a['state_key'] == c['state_key']
    foe = a['truth'][1 - a['side']]['roster']
    assert next(m for m in foe if m['species'] == 'misdreavus')['hp'] == 0
    roots = [r for r in read(replays / 'probes' / f'{job}.jsonl') if r['kind'] == 'root']
    by = {v: {r['seed']: r for r in roots if r['variant'] == v} for v in ['baseline', 'early-perish']}
    assert by['baseline'].keys() == by['early-perish'].keys()
    for variant, row, name in zip(['baseline', 'early-perish'], [a, c], names):
        assert by[variant][row['seed']]['root'] == row['root']
        (detail / f'{name}.jsonl').write_text(json.dumps(row) + '\n')
    case['root_review'] = {'turn': a['turn'], 'paired_seeds': len(by['baseline']),
                           'identical_roots': sum(by['baseline'][s]['root'] == by['early-perish'][s]['root'] for s in by['baseline']),
                           'opponent_misdreavus_fainted': True}
    replay = json.loads((replays / 'traces' / f'{names[0]}-replay.json').read_text())
    at = next(i for i, x in enumerate(replay['inputs']) if x['key'] == a['state_key'])
    while at < len(replay['inputs']) and replay['inputs'][at]['key'] == a['state_key']:
        at += 1
    replay = {k: replay[k] for k in ['teams', 'battle_seed', 'inputs']}
    replay['inputs'] = replay['inputs'][:at]
    (detail / f'{names[0]}-replay.json').write_text(json.dumps(replay) + '\n')
    shutil.copy2(replays / 'probes' / f'{job}.jsonl', detail / f'{job}-probe.jsonl')
result = {'selection': 'All six A wins / C losses in kingdra-kit2; selected after outcomes, exploratory only. No cases removed from win totals.',
          'root_selection': 'Jobs 5 and 21 have their first visible divergence after the opponent Misdreavus fainted; replay all four games and compare same-state roots with paired seeds.',
          'full_games_reproduced': 4, 'root_searches': sum(c.get('root_review', {}).get('paired_seeds', 0) * 2 for c in cases),
          'method': 'Public turn text is descriptive only; full state and root-search identity are checked only for the two replayed cases.', 'cases': cases}
(out / 'cluster-review.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))
