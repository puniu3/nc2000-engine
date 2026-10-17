#!/usr/bin/env python3
"""Audit the frozen experiment and report block-paired effects.
--progress prints completion/timing only, never interim strength conclusions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
parser.add_argument('--progress', action='store_true')
args = parser.parse_args()
raw = (args.directory / 'plan.json').read_bytes()
plan = json.loads(raw)
plan_sha = hashlib.sha256(raw).hexdigest()
jobs = {j['id']: j for j in plan['jobs']}
assert len(jobs) == len(plan['jobs'])
rows = {}
for path in sorted((args.directory / 'rows').glob('*.json')):
    row = json.loads(path.read_text())
    key = row['job']['id']
    assert key not in rows and row['job'] == jobs[key], path
    assert row['plan_sha256'] == plan_sha, path
    rows[key] = row
if args.progress:
    elapsed = [r['elapsed_seconds'] for r in rows.values()]
    print(json.dumps({'completed': len(rows), 'planned': len(jobs),
        'mean_game_seconds': float(np.mean(elapsed)) if elapsed else None,
        'completed_by_cell': {c: sum(r['job']['cell'] == c for r in rows.values())
            for c in sorted({j['cell'] for j in jobs.values()})}}, indent=2))
    raise SystemExit()
assert len(rows) == len(jobs), f'incomplete: {len(rows)}/{len(jobs)}'
assert not any(r['capped'] or r['score'] is None for r in rows.values()), 'unresolved games require bounds, not tie imputation'
for r in rows.values():
    assert r['score'] in (0, 0.5, 1)
    expected = {'Some(P1Win)': int(r['job']['leg'] == 0), 'Some(P2Win)': int(r['job']['leg'] == 1), 'Some(Tie)': 0.5}
    assert r['score'] == expected[r['outcome']]
    assert r['turns'] <= plan['max_turns']
    counts = r['target_stats']
    assert counts['preview']['decisions'] + counts['battle']['decisions'] == r['target_calls']
    for phase in counts.values():
        assert phase['iterations'] == phase['decisions'] * plan['budget']
        assert phase['mixed'] <= phase['decisions'] and phase['fallback'] <= phase['decisions']
    if r['job']['opponent'] == 'control':
        assert all(s['preview']['decisions'] == s['battle']['decisions'] == 0 for s in r['model_stats'])

def bootstrap(values, level=0.95):
    values = np.asarray(values, dtype=float)
    assert len(values) >= 2
    rng = np.random.default_rng(202610170303)
    means = np.concatenate([values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1) for _ in range(10)])
    q = (1 - level) / 2
    lo, hi = np.quantile(means, [q, 1 - q])
    return {'estimate': float(values.mean()), 'interval': [float(lo), float(hi)], 'level': level, 'blocks': len(values)}

cells = {}
for cell in sorted({j['cell'] for j in jobs.values()}):
    group = [r for r in rows.values() if r['job']['cell'] == cell]
    blocks = {}
    for r in group:
        blocks.setdefault(r['job']['block'], {})[r['job']['leg']] = r
    assert all(set(b) == {0, 1} for b in blocks.values())
    block_score = {i: (b[0]['score'] + b[1]['score']) / 2 for i, b in blocks.items()}
    block_win = {i: (int(b[0]['score'] == 1) + int(b[1]['score'] == 1)) / 2 for i, b in blocks.items()}
    if cell == 'direct-legacy':
        assert all(b[0]['trace_sha256'] == b[1]['trace_sha256'] and b[0]['outcome'] == b[1]['outcome'] for b in blocks.values())
        assert all(x == 0.5 for x in block_score.values()), 'side-swap null control failed'
    stats = {phase: {k: sum(r['target_stats'][phase][k] for r in group)
                    for k in group[0]['target_stats'][phase]} for phase in ['preview', 'battle']}
    cells[cell] = {'games': len(group), 'wins': sum(r['score'] == 1 for r in group),
        'draws': sum(r['score'] == 0.5 for r in group), 'losses': sum(r['score'] == 0 for r in group),
        'score': bootstrap(list(block_score.values())) if len(block_score) >= 2 else {'estimate': np.mean(list(block_score.values()))},
        'pure_win': bootstrap(list(block_win.values())) if len(block_win) >= 2 else {'estimate': np.mean(list(block_win.values()))},
        'block_scores': block_score, 'block_wins': block_win, 'stats': stats,
        'mean_turns': float(np.mean([r['turns'] for r in group])),
        'target_ms_per_decision': sum(r['target_ns'] for r in group) / sum(r['target_calls'] for r in group) / 1e6,
        'opponent_ms_per_decision': sum(r['opponent_ns'] for r in group) / sum(r['opponent_calls'] for r in group) / 1e6}

def paired(a, b, field='block_scores', level=0.9875):
    x, y = cells[a][field], cells[b][field]
    keys = sorted(set(x) & set(y))
    # The schedule is explicitly paired; verify physical conditions agree.
    for block in keys:
        ja = next(r['job'] for r in rows.values() if r['job']['cell'] == a and r['job']['block'] == block)
        jb = next(r['job'] for r in rows.values() if r['job']['cell'] == b and r['job']['block'] == block)
        for k in ['teams', 'battle_seed', 'agent_seeds', 'model_seeds']:
            assert ja[k] == jb[k]
    return bootstrap([x[i] - y[i] for i in keys], level)

primary, secondary = {}, {}
if not plan['pilot']:
    for mode in ['visit-sample', 'estimated-mix']:
        direct = bootstrap([s - 0.5 for s in cells[f'direct-{mode}']['block_scores'].values()], 0.9875)
        resistance = paired(f'oracle-{mode}', 'oracle-legacy')
        primary[mode] = {'direct_score_minus_half': direct, 'oracle_score_minus_legacy': resistance,
            'both_improvements_demonstrated': direct['interval'][0] > 0 and resistance['interval'][0] > 0}
        secondary[f'pure-win-oracle-{mode}-minus-legacy'] = paired(f'oracle-{mode}', 'oracle-legacy', 'block_wins', 0.95)
    for mode in ['legacy', 'visit-sample', 'estimated-mix']:
        secondary[f'oracle-loss-vs-control-{mode}'] = paired(f'control-{mode}', f'oracle-{mode}', level=0.95)

result = {'schema': 'nc2000-mixed-policy-evaluation-summary-v1', 'pilot': plan['pilot'],
    'plan_sha256': plan_sha, 'games': len(rows), 'all_terminal': True, 'null_control': 'pass',
    'numpy_version': np.__version__, 'cells': cells, 'primary': primary, 'secondary': secondary,
    'scope': plan['protocol']['scope'],
    'limitations': ['Approximate root-response exploiter, not a certified best response.',
        'Attacker has full state; control shares that advantage. Defender seeds and sampled actions are not exposed.',
        'Research live-preview path; no claim about published compiled-preview opening behavior.',
        'An interval crossing zero is inconclusive, not proof of equivalence.']}
(args.directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'games': result['games'], 'all_terminal': True, 'primary': primary, 'secondary': secondary}, indent=2))
