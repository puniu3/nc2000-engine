"""Report the illustrative case fixed before aggregate outcomes were inspected."""
import json
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
plan = read(P / 'selection-case-plan.json')
root = read(P / 'unknown' / f'unknown-{plan["root"]:02}.json')
seed_index = root['job']['seeds'].index(plan['search_seed'])
choices = {s['budget']: s['choice'] for s in root['samples'] if s['seed'] == plan['search_seed']}
low, high = choices[81000], choices[243000]
assert low != high and high == choices[729000] == choices[2187000]
rows = [read(P / 'continuation-merged' / f'r{plan["root"]:02}-s{seed_index}-t{trial:02}.json') for trial in range(16)]
assert all(r['root'] == plan['root'] and r['search_seed'] == plan['search_seed'] for r in rows)
scores = {}
for label, choice in [('81k', low), ('243k_and_above', high)]:
    counts = Counter(r['scores'][choice] for r in rows)
    slots = [int(s.strip()) - 1 for s in choice.removeprefix('team ').split(',')]
    scores[label] = dict(choice=choice, species=[root['job']['own'][i]['species'] for i in slots], wins=counts[1], ties=counts[.5], losses=counts[0], mean=sum(r['scores'][choice] for r in rows) / 16)
result = dict(root=plan['root'], search_seed=plan['search_seed'], trials=16, scores=scores,
              high_minus_low=scores['243k_and_above']['mean'] - scores['81k']['mean'],
              scope='Prespecified illustrative selection change, not a separate confirmatory strength claim.')
(P / 'selection-case-result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
