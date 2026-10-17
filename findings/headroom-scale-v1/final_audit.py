#!/usr/bin/env python3
"""Post-experiment audit; independently recompute the frozen betting rule in Decimal."""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
FRACTIONS = tuple(map(Decimal, ['.025', '.05', '.1', '.2', '.3', '.4', '.5', '.7', '.9']))


def capital(values, mean):
    counts = Counter(values)
    total = Decimal(0)
    for fraction in FRACTIONS:
        product = Decimal(1)
        for value, count in counts.items():
            product *= (1 + fraction * (value / mean - 1)) ** count
        total += product
    return total / len(FRACTIONS)


def lower(values, threshold):
    lo, hi = Decimal(0), Decimal(1)
    for _ in range(70):
        midpoint = (lo + hi) / 2
        if capital(values, midpoint) >= threshold:
            lo = midpoint
        else:
            hi = midpoint
    return lo


def main():
    protocol = read(HERE / 'protocol.json')
    amendment = read(HERE / 'confirmation-amendment.json')
    binding = read(HERE / 'confirmation-implementation.json')
    summary = read(HERE / 'summary.json')
    controller = read(HERE / 'controller.json')
    for name, expected in protocol['inputs'].items():
        assert sha(ROOT / name) == expected, name
    for filename, key in [('confirmation-amendment.json', 'amendment_sha256'),
                          ('summarize.py', 'summarizer_sha256'),
                          ('test_confirmation.py', 'tests_sha256'),
                          ('advance.py', 'controller_sha256')]:
        assert sha(HERE / filename) == binding[key], filename
    assert sha(ROOT / 'tools/evaluate-learning.py') == binding['evaluator_sha256']
    assert amendment['original_protocol_sha256'] == sha(HERE / 'protocol.json')
    assert amendment['registered_at'] < binding['registered_at']
    assert amendment['completed_blocks_when_registered'] == [0]
    assert amendment['block_01_result_absent_when_registered']
    assert binding['block_01_result_absent_when_bound']
    assert sha(HERE / 'results/block-00-raw.jsonl') == amendment['discovery_raw_sha256']
    assert len(list((HERE / 'results').glob('block-??-raw.jsonl'))) == summary['completed_blocks'] == 7
    assert controller['status'] == 'positive_checkpoint_verified'
    binary_sha = sha(ROOT / 'target/release/examples/arena')
    pairs, games, raw_games, checkpoints, raw_digests = [], [], [], [], {}
    timing_samples = {'a': [], 'b': []}
    threshold = Decimal(2) / Decimal(str(protocol['alpha']))
    with localcontext() as context:
        context.prec = 80
        for index in range(summary['completed_blocks']):
            path = HERE / f'results/block-{index:02}-raw.jsonl'
            raw, compact = read(path), read(path.with_name(f'block-{index:02}.json'))
            assert compact['raw_sha256'] == sha(path)
            assert compact['binary_sha256'] == binary_sha
            assert raw['config']['threads'] == 10
            assert raw['result']['turn_caps'] == raw['result']['invalid_games'] == 0
            raw_digests[path.name] = sha(path)
            raw_games.extend(raw['result']['game_scores'])
            for side in ('a', 'b'):
                timing = raw['timing'][side]
                samples = timing['samples_ns']
                assert len(samples) == timing['moves'] and sum(samples) == timing['total_ns']
                assert {k: v for k, v in timing.items() if k != 'samples_ns'} == compact['timing'][side]
                timing_samples[side].extend(samples)
            if index == 0:
                continue
            retained = [Decimal(str(v)) for v in raw['result']['game_scores'][2:]]
            games.extend(retained)
            pairs.extend((a + b) / 2 for a, b in zip(retained[::2], retained[1::2]))
            interval = [lower(pairs, threshold), 1 - lower([1 - x for x in pairs], threshold)]
            checkpoint = summary['confirmation_checkpoints'][index - 1]
            assert all(abs(float(a) - b) < 1e-12 for a, b in zip(interval, checkpoint['betting_interval']))
            checkpoints.append({'block': index, 'games': len(games), 'score': float(sum(pairs) / len(pairs)),
                                'interval_decimal': list(map(str, interval)),
                                'evalue_at_half_decimal': str(capital(pairs, Decimal('.5'))),
                                'positive': interval[0] > Decimal('.5')})
    assert [c['block'] for c in checkpoints if c['positive']] == [6]
    assert summary['first_positive_checkpoint']['block'] == 6
    assert summary['positive'] and summary['analysis_games'] == len(games) == 756
    assert len(raw_games) == summary['games'] == 896
    assert [float(v) for v in pairs] == summary['pair_scores']
    assert abs(float(sum(games) / len(games)) - summary['score']) < 1e-15
    nulls, block_starts = [], []
    for name in ('execution-00.stdout.jsonl', 'execution-from-04.stdout.jsonl'):
        rows = [json.loads(line) for line in (HERE / name).read_text().splitlines()]
        assert rows[-1]['type'] == 'segment_complete'
        block_starts.extend(row['index'] for row in rows if row['type'] == 'block_start')
        null = next(row['result'] for row in rows if row['type'] == 'null')
        assert null['games'] == 8 and null['score'] == .5 and null['split_pairs'] == 0
        assert null['invalid_games'] == null['turn_caps'] == 0
        nulls.append({'stdout': name, 'games': 8, 'score': .5})
    assert block_starts == list(range(7)), 'Unexpected repeated or overrun block'
    for pid in (20051, 39496, 91736, 29015):
        command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
        assert 'headroom-scale-v1/' not in command and '/target/release/examples/arena ' not in command
    timing = {}
    for side, samples in timing_samples.items():
        samples.sort()
        timing[side] = {'choose_calls': len(samples), 'mean_ms': sum(samples) / len(samples) / 1e6,
                        'p95_ms': samples[(95 * len(samples) + 99) // 100 - 1] / 1e6,
                        'scope': 'All 896 executed games, including discovery and excluded pairs; loaded native host, 10 threads.'}
    result = {'schema': 'nc2000-headroom-scale-final-audit-v1', 'status': 'pass',
              'audited_at': datetime.now(timezone.utc).isoformat(),
              'script_sha256': sha(Path(__file__)), 'frozen_input_count': len(protocol['inputs']),
              'binary_sha256': binary_sha, 'protocol_sha256': sha(HERE / 'protocol.json'),
              'amendment_sha256': sha(HERE / 'confirmation-amendment.json'),
              'summary_sha256': sha(HERE / 'summary.json'), 'raw_sha256': raw_digests,
              'executed_comparison_games': len(raw_games), 'confirmation_games': len(games),
              'wins': games.count(Decimal(1)), 'losses': games.count(Decimal(0)),
              'ties': games.count(Decimal('.5')), 'invalid_games': 0, 'turn_caps': 0,
              'exclusions': {'discovery_games': 128, 'first_pair_games': 12},
              'checkpoints': checkpoints, 'two_sided_evalue_threshold': str(threshold),
              'null_controls': nulls, 'executed_block_indices': block_starts,
              'stopped': True, 'timing': timing,
              'mean_choose_time_ratio': timing['a']['mean_ms'] / timing['b']['mean_ms'],
              'conclusion': 'Additional search demonstrates strength headroom on this registered catalog distribution; equal-latency improvement and distance to game-theoretic optimality remain unmeasured.'}
    (HERE / 'final-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'frozen_input_count', 'executed_comparison_games',
                     'confirmation_games', 'wins', 'losses', 'ties', 'mean_choose_time_ratio', 'stopped')}, indent=2))


if __name__ == '__main__':
    main()
