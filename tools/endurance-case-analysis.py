#!/usr/bin/env python3
"""Describe paired trajectories without treating their RNG coupling as causality."""
import collections
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
data = root / 'findings/deployed-endurance-v1'


def resources(game):
    choices = collections.Counter()
    full_heals = []
    exhausted = {}
    for row in game['trace']:
        side = row['state']['sides'][1]
        mon = next(p for p in side['party'] if p['species'] == 'Blissey')
        if not mon['fainted']:
            pp = {m['id']: m['pp'] for m in mon['moves']}
            for move, remaining in pp.items():
                if remaining == 0:
                    exhausted.setdefault(move, row['turn'])
        if side['active'] is None:
            continue
        active = side['party'][side['active']]
        if active['species'] != 'Blissey':
            continue
        choice = dict(row['picks']).get(1)
        if choice:
            choices[choice] += 1
        if choice == 'move softboiled' and active['status'] != 'frz' and active['hp'] == active['maxhp']:
            full_heals.append(row['turn'])
    return {'blisseyChoices': dict(choices), 'unfrozenFullHPHeals': full_heals,
            'firstObservedPPExhaustionWhileAlive': exhausted}


def all_choices(game):
    counters = [collections.Counter(), collections.Counter()]
    for row in game['trace']:
        for side, choice in row['picks']:
            state = row['state']['sides'][side]
            species = state['party'][state['active']]['species'] if state['active'] is not None else 'none'
            counters[side][species + ': ' + choice] += 1
    return [dict(c) for c in counters]


def hp_pp(state):
    return [sorted((mon['species'], mon['hp'],
                    tuple(sorted((move['id'], move['pp']) for move in mon['moves'])))
                   for mon in side['party']) for side in state['sides']]


rows = []
for trial in range(16):
    paths = [data / 'continuations' / f'g4-d12-{arm}-{trial}.json'
             for arm in ['baseline', 'conserve-heal']]
    if not all(p.exists() for p in paths):
        continue
    a, b = [json.loads(p.read_text()) for p in paths]
    if a['score'] is None or b['score'] is None:
        continue
    divergence = None
    for index, (x, y) in enumerate(zip(a['trace'], b['trace'])):
        # All prefix states must coincide until the first changed action.
        assert x['state'] == y['state'], (trial, index)
        if x['picks'] != y['picks']:
            side = x['state']['sides'][1]
            mon = side['party'][side['active']]
            assert dict(x['picks'])[1] == 'move softboiled'
            assert dict(y['picks'])[1] == 'move icebeam'
            divergence = {'index': index, 'turn': x['turn'], 'hp': mon['hp'],
                          'maxhp': mon['maxhp'], 'status': mon['status'],
                          'pp': {m['id']: m['pp'] for m in mon['moves']}}
            break
    if divergence is None:
        assert a['trace'] == b['trace'] and a['outcome'] == b['outcome']
    rows.append({'trial': trial, 'scores': [a['score'], b['score']],
                 'endTurns': [a['endTurn'], b['endTurn']],
                 'firstDivergence': divergence,
                 'baseline': resources(a), 'alternative': resources(b),
                 'choiceCounts': {'baseline': all_choices(a), 'alternative': all_choices(b)}})

long_games = []
for f in (data / 'continuations').glob('g4-d12-*.json'):
    if len(f.name.split('.')) != 2:
        continue
    game = json.loads(f.read_text())
    if game['score'] is None:
        continue
    both_switch = 0
    struggle = [0, 0]
    runs = []
    run = []
    for index, row in enumerate(game['trace']):
        picks = dict(row['picks'])
        both = (len(picks) == 2 and all(c.startswith('switch ') for c in picks.values())
                and all(s['request'] == 'move' for s in row['state']['sides']))
        both_switch += both
        if both:
            run.append(index)
        elif run:
            runs.append(run)
            run = []
        for side, choice in row['picks']:
            struggle[side] += choice == 'move struggle'
    if run:
        runs.append(run)
    longest = max(runs, key=len, default=[])
    detail = None
    if longest:
        first, last = longest[0], longest[-1]
        through = min(last + 1, len(game['trace']) - 1)
        start_resources = hp_pp(game['trace'][first]['state'])
        detail = {'decisions': len(longest), 'firstTurn': game['trace'][first]['turn'],
                  'lastTurn': game['trace'][last]['turn'],
                  'checkedThroughTurn': game['trace'][through]['turn'],
                  'includesStateAfterLastSwitch': last + 1 < len(game['trace']),
                  'hpAndPPUnchanged': all(hp_pp(row['state']) == start_resources
                                         for row in game['trace'][first:through + 1])}
    long_games.append({'file': f.name, 'trial': game['trial'], 'arm': game['action'],
                       'endTurn': game['endTurn'], 'score': game['score'],
                       'bothVoluntarilySwitchDecisions': both_switch,
                       'longestJointSwitchRun': detail, 'struggleSelections': struggle})
long_games.sort(key=lambda g: (-g['endTurn'], g['file']))

result = {'scope': 'Descriptive paired paths; action changes also change future RNG consumption',
          'completedPairs': len(rows),
          'identicalPairs': [r['trial'] for r in rows if r['firstDivergence'] is None],
          'rows': rows, 'longestCompletedGames': long_games[:3]}
(data / 'case-analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'pairs': len(rows), 'identicalPairs': result['identicalPairs']}))
