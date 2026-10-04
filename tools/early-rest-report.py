import collections
import gzip
import hashlib
import itertools
import json
import pathlib
import sys

import numpy as np


def read(path):
    return json.loads(path.read_text())


def outcome_summary(pairs):
    complete = [p for p in pairs if None not in p['scores']]
    return {'pairs': len(pairs), 'complete_pairs': len(complete),
        'scores': [sum(p['scores'][i] for p in complete) for i in range(2)],
        'better': sum(p['scores'][1] > p['scores'][0] for p in complete),
        'worse': sum(p['scores'][1] < p['scores'][0] for p in complete),
        'same': sum(p['scores'][1] == p['scores'][0] for p in complete),
        'caps_by_arm': [sum(p['scores'][i] is None for p in pairs) for i in range(2)],
        'delta_bounds': [sum((p['scores'][1] if p['scores'][1] is not None else limit) - (p['scores'][0] if p['scores'][0] is not None else 1-limit) for p in pairs) / len(pairs) for limit in [0, 1]],
        'changed_pairs': sum(p['first_difference'] is not None for p in pairs)}


def posterior(groups, plan):
    if any(None in p['scores'] for group in groups for p in group):
        return {'incomplete': True, 'reason': 'capped outcomes remain unknown'}
    patterns = np.array(list(itertools.product([0, .5, 1], repeat=2)))
    deltas = patterns[:, 1] - patterns[:, 0]
    result = {}
    for mass in plan['prior_mass']:
        rng = np.random.default_rng(plan['seed'])
        draws = np.zeros(plan['draws'])
        exact_mean = 0
        for group in groups:
            scores = np.array([p['scores'] for p in group])
            counts = np.array([np.all(scores == pattern, axis=1).sum() for pattern in patterns])
            draws += rng.dirichlet(counts + mass / len(patterns), size=plan['draws']) @ deltas / len(groups)
            exact_mean += float(counts @ deltas) / (len(group) + mass) / len(groups)
        assert abs(draws.mean() - exact_mean) < .002
        result[str(mass)] = {'probability_improved': float(np.mean(draws > 0)),
            'mean_delta': float(draws.mean()), 'analytic_mean_delta': exact_mean,
            'credible95': np.quantile(draws, [.025, .975]).tolist(),
            'probability_loss_gt_1pp': float(np.mean(draws < -.01))}
    return result


def main(run, out):
    protocol = read(out / 'protocol.json')
    plan = read(out / 'analysis-plan.json')
    proof = read(out / 'perish-guard-equivalence.json')
    plain_id = lambda key: ''.join(c.lower() for c in key if c.isalnum())
    selected_species = set()
    for case in protocol['cases']:
        for slot in map(lambda x: int(x.strip())-1, case['prefix'][0][1][5:].split(',')):
            mon = case['teams'][1][slot]
            selected_species.add(plain_id(mon['species']))
            assert not ({'meanlook','perishsong','mimic','transform','metronome','mirrormove'} & set(map(plain_id,mon['moves'])))
    pool = read(pathlib.Path(protocol['prior']))
    sources = [s for t in pool['teams'] for s in t['sets'] if plain_id(s['species']) in selected_species]
    assert {plain_id(s['species']) for s in sources} == selected_species
    assert all(not ({'meanlook','perishsong'} & set(map(plain_id,s['moves']))) for s in sources)
    assert proof['prior_sha256'] == protocol['prior_sha256']
    assert len(sources) == proof['checked_source_sets']
    manifest = read(run / 'pilot/manifest.json')
    assert manifest['protocol'] == protocol
    assert manifest['protocol_sha256'] == hashlib.sha256((out / 'protocol.json').read_bytes()).hexdigest()
    verification = read(run / 'pilot-verification.json')
    assert verification['games'] == 2 * sum(c['pairs'] for c in protocol['cases'])
    cases, pairs, games = [], [], []
    for case in protocol['cases']:
        group = []
        for trial in range(case['pairs']):
            arms = [read(run / 'pilot' / f"{case['id']}-t{trial:03}-a{arm}.json") for arm in range(2)]
            assert arms[0]['battle_seed'] == arms[1]['battle_seed']
            assert arms[0]['agent_seeds'] == arms[1]['agent_seeds']
            first = next((i for i,(a,b) in enumerate(zip(arms[0]['decisions'],arms[1]['decisions'])) if a['choices'] != b['choices']), None)
            for a,b in zip(arms[0]['decisions'][:first],arms[1]['decisions'][:first]):
                assert a == b
            if first is None:
                assert arms[0]['decisions'] == arms[1]['decisions'] and arms[0]['score'] == arms[1]['score']
            else:
                a,b = [arm['decisions'][first] for arm in arms]
                assert a['key'] == b['key'] and a['search_seeds'] == b['search_seeds']
                assert a['choices'][1-arms[0]['side']] == b['choices'][1-arms[0]['side']]
            if case['family'] == 'onpool-control':
                assert first is None
            row = {'case': case['id'], 'family': case['family'], 'actual_rest': case['actual_rest'], 'trial':trial,
                'scores': [a['score'] for a in arms], 'first_difference': first}
            pairs.append(row)
            group.append(row)
            games.extend(arms)
        cases.append({'case': case['id'], 'actual_rest': case['actual_rest'], **outcome_summary(group)})
    stress = [p for p in pairs if p['actual_rest'] is not None]
    by_group = {name:[p for p in stress if p['actual_rest'] == flag] for name,flag in [('rest',True),('no_rest',False)]}
    grouped = [[p for p in stress if p['case']==case['id']] for case in protocol['cases'] if case['actual_rest'] is not None]
    bayes = posterior(grouped, plan)
    calibration_plan = {**plan,'draws':50000,'prior_mass':[1]}
    null = posterior([[{'scores':[0,0]},{'scores':[1,1]}] for _ in range(8)],calibration_plan)['1']
    positive = posterior([[{'scores':[0,1]} for _ in range(4)] for _ in range(8)],calibration_plan)['1']
    negative = posterior([[{'scores':[1,0]} for _ in range(4)] for _ in range(8)],calibration_plan)['1']
    assert .48 < null['probability_improved'] < .52
    assert positive['probability_improved'] > .99 and negative['probability_improved'] < .01
    audit = read(run / 'first-divergence.json')
    assert len(audit) == sum(p['first_difference'] is not None for p in pairs)
    assert all(m['choice_reproduced'] for r in audit for m in r['models'])
    old = json.loads(gzip.decompress(pathlib.Path('findings/next-move-support-v1/records.json.gz').read_bytes()))
    corpus = [json.loads(line) for line in (run / 'corpus/rows.jsonl').read_text().splitlines()]
    assert len(old) == len(corpus)
    for a,b in zip(old,corpus):
        assert {k:v for k,v in a.items() if k!='arms'} == {k:v for k,v in b.items() if k!='arms'}
        if 'arms' in a:
            assert a['arms'][:2] == b['arms'][:2]
    valid = [r for r in corpus if 'arms' in r]
    support = {}
    for name,selected in [('all',valid),('rest',[r for r in valid if r['move']=='rest'])]:
        support[name] = {'labels':len(selected),'any_sample':[sum(r['arms'][i]['hits']>0 for r in selected) for i in [0,2]],
            'mean_inclusion':[sum(r['arms'][i]['hits']/64 for r in selected)/len(selected) for i in [0,2]]}
    for r in valid:
        for arm in r['arms']:
            plain=lambda key:'hiddenpower' if key.startswith('hiddenpower') else key
            assert sum(k['count'] for k in arm['kits']) == 64
            assert arm['hits'] == sum(k['count'] for k in arm['kits'] if r['move'] in {plain(m) for m in k['moves']})
            assert all(set(map(plain,r['revealed'])) <= set(map(plain,k['moves'])) and len(k['moves'])<=4 for k in arm['kits'])
    summary={'schema':'nc2000-early-rest-results-v1','default_changed':False,'cases':cases,'stress':outcome_summary(stress),
        'groups':{name:outcome_summary(group) for name,group in by_group.items()},'posterior':bayes,
        'posterior_calibration':{'null':null,'positive':positive,'negative':negative},'support':support,
        'default_corpus_reproduced_labels':len(valid),'replay':verification,'reproduced_first_divergence_searches':len(audit)*2,
        'limitations':['Fixed synthetic opponent teams and common bot; no expert-human comparison.',
            'Equal Rest/no-Rest weighting is a stress scenario assumption, not human prevalence.',
            'Exploratory pilot only. No adoption without fresh confirmation.',
            'Support diagnostic uses positive labels that motivated the candidate; it is not independent strength evidence.',
            'Pilot preceded the final Perish-slot guard. Fixture-level equivalence is recorded, and all first-divergence choices are rechecked with final code.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for src,dest in [(run/'pilot/manifest.json','pilot-manifest.json'),(run/'pilot-verification.json','pilot-verification.json'),(run/'first-divergence.json','first-divergence.json'),(run/'pilot-source-sha256.json','pilot-source-sha256.json')]:
        (out/dest).write_bytes(src.read_bytes())
    (out/'records.json.gz').write_bytes(gzip.compress(json.dumps({'pairs':pairs,'games':games,'corpus':corpus},separators=(',',':')).encode(),mtime=0))
    print(json.dumps({k:summary[k] for k in ['stress','groups','posterior','support']},indent=2))


if __name__=='__main__':
    main(*map(pathlib.Path,sys.argv[1:]))
