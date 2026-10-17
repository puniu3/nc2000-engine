#!/usr/bin/env python3
"""Audit probability arithmetic, root binding and the preregistered seed experiment."""
import collections, gzip, hashlib, json, math, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIR = ROOT / 'findings/deployed-bond-counterexample-v1'
def read(path):
    path = pathlib.Path(path)
    if not path.exists():
        path = pathlib.Path(str(path)+'.gz')
    return gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()
def doc(name):
    with read(DIR/name) as f:
        return json.load(f)
def lines(name):
    with read(DIR/name) as f:
        return [json.loads(line) for line in f]
def close(a,b):
    assert math.isclose(a,b,rel_tol=0,abs_tol=1e-11), (a,b)
def bounds(v):
    assert 0 <= v[0] <= v[1]+1e-11 <= 1+2e-11
def lower_audit(nodes):
    cache={(n['key'],n['remaining'],n['depth']):n for n in nodes}
    assert len(cache)==len(nodes)
    for n in nodes:
        bounds(n['win_bounds']);assert n['replies'];close(sum(p['prob'] for p in n['policy']),1)
        for r in n['replies']:
            assert len(r['cells'])==len(n['policy'])
            for c,p in zip(r['cells'],n['policy']):
                assert c['action']==p['action'];close(c['weight'],p['prob'])
                close(c['pending']+sum(e['prob'] for e in c['children']),1)
                for e in c['children']:
                    assert e['prob']>0;bounds(e['win_bounds'])
                    assert e['depth']==n['depth']-1 and e['remaining']==p['next_remaining']
                    if e['outcome'] is not None:
                        assert e['win_bounds']==([1,1] if e['outcome']=='P1Win' else [0,0])
                    elif (e['key'],e['remaining'],e['depth']) in cache:
                        assert e['win_bounds']==cache[e['key'],e['remaining'],e['depth']]['win_bounds']
                    else:
                        assert e['win_bounds']==[0,1]
                for k in range(2):
                    close(c['win_bounds'][k],(c['pending'] if k else 0)+sum(e['prob']*e['win_bounds'][k] for e in c['children']))
            for k in range(2):close(r['win_bounds'][k],sum(c['weight']*c['win_bounds'][k] for c in r['cells']))
        for k in range(2):close(n['win_bounds'][k],min(r['win_bounds'][k] for r in n['replies']))

def main():
    fixture=doc('screen-fixture.json');plan=doc('confirmation-plan.json');samples=lines('blind-confirmation.jsonl')
    assert len(samples)==len(plan['seeds'])==128
    assert [r['seed'] for r in samples]==plan['seeds']
    assert all(r['mode']=='blind' and r['side']==0 and r['iterations']==81000 and r['fixture']=='screen' for r in samples)
    assert all(r['belief']['fallback'] and r['belief']['count']==0 for r in samples)
    assert all({a['input'] for a in r['policy']['actions']}=={'move curse','move earthquake'} for r in samples)
    counts=collections.Counter(r['best'] for r in samples);assert counts=={'move curse':128}
    proof=doc('final-proof-v2/summary.json');nodes=lines('final-proof-v2/nodes.jsonl');lower_audit(nodes)
    assert nodes[-1]['key']==fixture['decisions'][-1]['key']
    assert nodes[-1]['win_bounds']==proof['pure_win_bounds'] and proof['nodes']==len(nodes)==395
    assert proof['truncated']==0 and proof['runs']==9142
    upper=doc('final-proof-v2/curse-upper.json')
    assert upper['pure_win_upper']==0 and upper['runs']==812
    upper_cache={(n['key'],n['depth']):n['win_upper'] for n in upper['nodes']}
    for n in upper['nodes']:
        assert n['cells']
        for c in n['cells']:
            close(c['pending']+sum(e['prob'] for e in c['children']),1)
            close(c['win_upper'],c['pending']+sum(e['prob']*e['win_upper'] for e in c['children']))
            assert c['reply']=='move thunderbolt'
            assert all(e['outcome'] is None or (e['outcome']=='P2Win' and e['win_upper']==0) for e in c['children'])
            for e in c['children']:
                if e['outcome'] is None:
                    close(e['win_upper'],upper_cache.get((e['key'],n['depth']-1),1.))
        close(n['win_upper'],max(c['win_upper'] for c in n['cells']))
    assert upper['nodes'][-1]['key']==nodes[-1]['key'] and upper['nodes'][-1]['first']
    assert [c['action'] for c in upper['nodes'][-1]['cells']]==['move curse']
    matrix=doc('fresh-matrix/records.json');assert all(v['ok'] for v in matrix['validations'])
    assert matrix['state_key']==nodes[-1]['key']
    assert matrix['actions'][0]==['move earthquake','move curse']
    assert matrix['actions'][1]==['move thunderbolt','move destinybond','switch 3']
    assert matrix['proof']['cells'][3]==[0,0]
    parity=doc('engine-parity.json');assert parity['trials']==768 and parity['state_and_rng_checks']==3731
    assert sum(parity['outcomes'].values())==768 and 'unfinished' not in parity['outcomes']
    deployment=doc('deployment.json');assert deployment['assets'][-1]['sha256']==plan['deployment_wasm_sha256']
    p_upper=1-math.exp(math.log(plan['alpha'])/len(samples))
    lower=proof['pure_win_bounds'][0]-proof['numerical_safety_margin']
    assert lower-p_upper>.05
    # Negative controls: do not silently accept an invented extra lower-bound win.
    mutated=json.loads(json.dumps(nodes));mutated[-1]['win_bounds'][0]+=.01
    try:lower_audit(mutated)
    except AssertionError:pass
    else:raise AssertionError('accepted false root value')
    mutated=json.loads(json.dumps(nodes));mutated[-1]['replies']=[]
    try:lower_audit(mutated)
    except AssertionError:pass
    else:raise AssertionError('accepted missing opponent replies')
    result={'schema':'nc2000-deployed-bond-counterexample-v1','choice_counts':dict(counts),
        'curse_pure_win_upper':upper['pure_win_upper'],'constructive_pure_win_lower':lower,
        'seed_probability_one_sided_confidence':1-plan['alpha'],'attack_probability_upper':p_upper,
        'current_seed_mixed_policy_win_upper':p_upper,'guaranteed_win_gap_lower':lower-p_upper,
        'proof_engine_runs':proof['runs']+upper['runs'],'proof_seconds':proof['seconds'],
        'native_matrix_value':matrix['proof']['value'],'searches':len(samples),
        'search_seconds':sum(s['seconds'] for s in samples),'engine_parity':parity,
        'limits':['Constructed legal 6-entry/3-selected custom-team endgame; occurrence frequency in ordinary games is not established.',
          'Bounds refer to this fixed true matchup against every legal opposing policy, not to a guarantee over every hidden loadout.',
          'The alternative reads public foe species and its own action history only. The baseline is deployed blind Strong without pinOpponent.',
          'The first-action Curse upper bound and alternative lower bound are exhaustive conservative engine calculations; the full seed-mixture comparison is statistical at one-sided 99.9% confidence.',
          'Chance enumeration uses independent per-draw engine probabilities and floating arithmetic with 1e-8 lower-bound padding, not an exhaustive count over all LCG initial seeds.',
          'Native/deployed engine parity is checked on sampled trajectories, not a formal equivalence proof. No bot changes or deployment.'],
        'arithmetic_audit':{'lower_nodes':len(nodes),'upper_nodes':len(upper['nodes']),'negative_controls':2}}
    (DIR/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DIR.rglob('*')) if p.is_file() and p.name not in {'manifest.json','index.html','preview-verification.json'}}
    for name in ['crates/bot/examples/bond_counterexample.rs','web/tests/deployed-bond-probe.mjs','web/tests/deployed-bond-parity.mjs','tools/deployed-bond-audit.py']:
        manifest[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (DIR/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'limits','engine_parity'}},ensure_ascii=False))

if __name__=='__main__':main()
