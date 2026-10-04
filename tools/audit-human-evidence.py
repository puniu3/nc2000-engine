import argparse
import collections
import hashlib
import json
import pathlib
import re

parser = argparse.ArgumentParser()
parser.add_argument('--logs', type=pathlib.Path, required=True)
parser.add_argument('--bot', required=True)
parser.add_argument('--out', type=pathlib.Path, required=True)
args = parser.parse_args()
files = sorted(args.logs.glob('*.raw.log'))
assert files
counts = collections.Counter()
opponents = collections.Counter()
builds = 0
rated = 0
fingerprint = hashlib.sha256()
for path in files:
    raw = path.read_bytes()
    fingerprint.update(hashlib.sha256(raw).digest())
    lines = raw.decode().splitlines()
    players = {}
    winner = None
    tie = False
    for line in lines:
        a = line.split('|')
        if len(a) > 3 and a[1] == 'player' and a[3]:
            assert a[2] not in players or players[a[2]] == a[3], 'account changed within a log'
            players[a[2]] = a[3]
        if len(a) > 2 and a[1] == 'win':
            winner = a[2]
        if len(a) > 1 and a[1] == 'tie':
            tie = True
    bot = [s for s, name in players.items() if name == args.bot]
    if len(bot) != 1:
        counts['missing_bot'] += 1
        continue
    other = [name for side, name in players.items() if side != bot[0]]
    assert len(other) == 1
    opponents[other[0]] += 1
    assert winner is None or winner in players.values()
    counts['win' if winner == args.bot else 'loss' if winner else 'tie' if tie else 'incomplete'] += 1
    builds += any(re.search(r'\b(?:commit|revision|build)[=: ]+[0-9a-f]{7,40}\b', line, re.I) for line in lines)
    rated += any(line == '|rated' or line.startswith('|rated|') for line in lines)
result = {'games': len(files), 'results': dict(counts),
          'distinct_opponent_accounts': len(opponents),
          'games_per_opponent_account': sorted(opponents.values(), reverse=True),
          'games_with_recognized_build_marker': builds, 'rated_marker_games': rated,
          'raw_content_fingerprint': fingerprint.hexdigest(),
          'audit_source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Archived configured bot-account protocol logs only. Empty player records on disconnect do not erase identity. Account count is not a count of distinct people or verified experts. Missing recognized build markers do not exclude external version mappings. These logs do not establish current EarlyRest human performance. No raw logs, identifiers or per-game rows are exported.'}
args.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
