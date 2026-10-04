import argparse
import collections
import hashlib
import json
import pathlib
import random
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'findings/operational-combo-validation-v1'

def quantile(xs, p):
    at = (len(xs) - 1) * p
    lo = int(at)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (at - lo)

def summarize(rows):
    diffs = [r['b']['score'] - r['a']['score'] for r in rows]
    groups = collections.defaultdict(list)
    for r, d in zip(rows, diffs):
        groups[r['opponent_team']].append(d)
    rng = random.Random(20261004)
    boot = sorted(sum(rng.choice(g) for g in groups.values() for _ in g) / len(rows) for _ in range(10000))
    result = {'pairs': len(rows), 'paired_delta': statistics.mean(diffs), 'stratified_bootstrap95': [quantile(boot, .025), quantile(boot, .975)], 'better_pairs': sum(x > 0 for x in diffs), 'worse_pairs': sum(x < 0 for x in diffs), 'same_outcome_pairs': sum(x == 0 for x in diffs), 'same_trajectory_pairs': sum(r['a']['trajectory'] == r['b']['trajectory'] for r in rows)}
    for arm in ['a', 'b']:
        result[arm] = {'score': statistics.mean(r[arm]['score'] for r in rows), 'terminal': sum(r[arm]['terminal'] for r in rows), 'wins': sum(r[arm]['terminal'] and r[arm]['score'] == 1 for r in rows), 'losses': sum(r[arm]['terminal'] and r[arm]['score'] == 0 for r in rows), 'capped': sum(not r[arm]['terminal'] for r in rows), 'exposed_games': sum(r[arm]['exposed'] > 0 for r in rows), 'fallback_decisions': sum(r[arm]['fallback'] for r in rows), 'exposed_decisions': sum(r[arm]['exposed'] for r in rows), 'missing_perish_decisions': sum(r[arm]['missing'] for r in rows), 'opponent_sang_games': sum(r[arm]['opponent']['sing'] > 0 for r in rows), 'bot_count1_faints': sum(r[arm]['own']['perish_faint'] for r in rows), 'elapsed_seconds': sum(r[arm]['elapsed_seconds'] for r in rows)}
    result['cap_delta_bounds'] = [statistics.mean((r['b']['score'] if r['b']['terminal'] else b) - (r['a']['score'] if r['a']['terminal'] else a) for r in rows) for b, a in [(0, 1), (1, 0)]]
    result['by_opponent'] = {key: {'pairs': len(g), 'paired_delta': statistics.mean(g)} for key, g in groups.items()}
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=pathlib.Path, default=OUT)
    out = parser.parse_args().dir
    report = {'candidate_default_enabled': False, 'results': {}}
    for name, expected in [('same', 4), ('stress', 64), ('natural', 32), ('onpool', 8)]:
        path = out / (name + '.jsonl')
        if not path.exists():
            continue
        rows = sorted([json.loads(line) for line in path.read_text().splitlines()], key=lambda r: r['job'])
        assert len(rows) == expected and len({r['job'] for r in rows}) == expected, (name, len(rows))
        if name in ['same', 'onpool']:
            for r in rows:
                for key in ['score', 'turns', 'terminal', 'trajectory', 'decisions', 'fallback', 'exposed', 'missing', 'own', 'opponent']:
                    assert r['a'][key] == r['b'][key], (name, r['job'], key)
        report['results'][name] = summarize(rows)
        report['results'][name]['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    report['complete'] = set(report['results']) == {'same', 'stress', 'natural', 'onpool'}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
