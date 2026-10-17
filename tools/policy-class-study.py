#!/usr/bin/env python3
"""Frozen-search selector ablations and generic interval-policy verification."""
import collections, hashlib, json, math, pathlib, subprocess, sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'findings/policy-class-v1'
def rows(path):return [json.loads(l) for l in pathlib.Path(path).read_text().splitlines()]
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def interval(record,side):
    n,m=map(len,record['actions']);c=record['cells']
    cells=c if side==0 else [[1-c[i*m+j][1],1-c[i*m+j][0]] for j in range(m) for i in range(n)]
    return {'replies':m if side==0 else n,'lower':[x[0] for x in cells],'upper':[x[1] for x in cells]}
def guarantee(t,p,upper=False):
    a=t['upper' if upper else 'lower'];m=t['replies']
    return min(sum(p[i]*a[i*m+j] for i in range(len(p))) for j in range(m))
def case(id,t,base):return dict(id=id,eligible=[True]*len(base),baseline=base,types=[t],model='WorstCase')

def prepare():
    certificates=rows(OUT/'certificates.jsonl');assert len(certificates)==9
    by_id={r['id']:r for r in certificates};searches=rows(OUT/'deployed-searches.jsonl');assert len(searches)==36
    cases=[]
    for r in searches:
        cert=by_id[r['id']];assert cert['state']==r['state'] and cert['seed']==r['chance_seed']
        actions=cert['actions'][0];base=[float(a==r['best']) for a in actions]
        cases.append(case(f"deployed:{r['id']}:{r['index']}",interval(cert,0),base))
    for r in certificates:
        for side in range(2):
            for i in range(len(r['actions'][side])):
                cases.append(case(f"pure:{r['id']}:{side}:{i}",interval(r,side),[float(j==i) for j in range(len(r['actions'][side]))]))
    for k in list(range(1,65))+[999,9999]:
        for g in [.1,.5,.9,1.]:
            a=[g,0.,0.,g/k];cases.append(case(f'family:{k}:{g}',dict(replies=2,lower=a,upper=a),[0.,1.]))
    (OUT/'inputs.jsonl').write_text(''.join(json.dumps(c,separators=(',',':'))+'\n' for c in cases))
    subprocess.run([str(ROOT/'target/release/examples/interval_root_probe'),str(OUT/'inputs.jsonl'),str(OUT/'outputs.jsonl')],check=True,cwd=ROOT)

def audit():
    assert '18 passed; 0 failed' in (OUT/'selector-tests.log').read_text()
    assert '2 passed; 0 failed' in (OUT/'game-tests.log').read_text()
    inputs=rows(OUT/'inputs.jsonl');outputs=rows(OUT/'outputs.jsonl');assert len(inputs)==len(outputs)
    for c,r in zip(inputs,outputs):
        assert c['id']==r['id'];v=r['result'];assert 'error' not in v
        t=c['types'][0];p=v['candidate'];assert abs(sum(p)-1)<1e-9 and min(p)>=0
        assert abs(v['candidate_value'][0]-max(0,guarantee(t,p)-2e-8))<1e-10
        assert abs(v['baseline_value'][1]-min(1,guarantee(t,c['baseline'],True)+2e-8))<1e-10
        assert v['replace']==(v['improvement_lower']>0)
        assert v['selected']==(p if v['replace'] else c['baseline'])
        assert abs(v['mixing_advantage_lower']-(v['candidate_value'][0]-min(1,max(min(t['upper'][i*t['replies']:(i+1)*t['replies']]) for i in range(len(p)))+2e-8)))<1e-10
        if c['id'].startswith('family:'):
            _,k,g=c['id'].split(':');k=int(k);g=float(g)
            assert abs(p[0]-1/(k+1))<1e-8 and abs(guarantee(t,p)-g/(k+1))<1e-8
    certified=rows(OUT/'certificates.jsonl');searches=rows(OUT/'deployed-searches.jsonl');lookup={r['id']:r['result'] for r in outputs}
    old=rows(ROOT/'findings/deployed-bond-counterexample-v1/blind-confirmation.jsonl')
    for r in old+searches:
        chosen=next(a for a in r['policy']['actions'] if a['input']==r['best'])
        assert chosen['visits']==max(a['visits'] for a in r['policy']['actions'])
    bond=next(r for r in certified if r['id']=='wait-7');t=interval(bond,0)
    fractions=[next(a['frac'] for a in r['policy']['actions'] if a['input']=='move earthquake') for r in old]
    mean=sum(fractions)/len(fractions)
    policies={'modal':[0.,1.],'uniform':[.5,.5],'average_visits':[mean,1-mean],'exact_family':[.125,.875]}
    ablation={k:dict(policy=p,guarantee=[guarantee(t,p),guarantee(t,p,True)]) for k,p in policies.items()}
    ablation['individual_visit_guarantees']={'min':min(guarantee(t,[p,1-p]) for p in fractions),'max':max(guarantee(t,[p,1-p]) for p in fractions)}
    cases=[]
    for c in certified:
        rs=[r for r in searches if r['id']==c['id']];assert len(rs)==4
        decisions=[lookup[f"deployed:{r['id']}:{r['index']}"] for r in rs]
        for r in rs:assert r['state']==c['state'] and r['chance_seed']==c['seed']
        if c['id'].startswith('wait-'):
            k=int(c['id'].split('-')[1]);v=.91461181640625/(k+1)
            assert c['value'][0]-1e-9<=v<=c['value'][1]+1e-9
            assert all(d['replace']==(k>0) for d in decisions)
        cases.append(dict(id=c['id'],value=c['value'],actions=c['actions'][0],choices=dict(collections.Counter(r['best'] for r in rs)),
            candidate=decisions[0]['candidate'],candidate_guarantee=decisions[0]['candidate_value'],
            replaced=sum(d['replace'] for d in decisions),runs=c['runs'],seconds=c['seconds']))
    g=.91461181640625
    result={'schema':'nc2000-policy-class-v1','fresh_certified_roots':len(certified),'deployed_searches':len(searches),'state_and_seed_parity':len(searches),
        'generic_selector_cases':len(inputs),'parameterized_family_cases':sum(c['id'].startswith('family:') for c in inputs),
        'fresh_engine_runs':sum(c['runs'] for c in certified),'fresh_proof_seconds':sum(c['seconds'] for c in certified),
        'deployed_search_seconds':sum(r['seconds'] for r in searches),'cases':cases,'frozen_128_search_ablation':ablation,
        'gengar_pure_penalty':lookup['pure:wait-7:1:1']['mixing_advantage_lower'],
        'continuation_model':{'correct_dynamic':g/8,'root_only_then_always_wait':0,'constant_one_eighth':g/8*(7/8)**7},
        'finite_game_validation':{'games':200,'nodes_per_game':7,'sweeps_per_game':7,'maximum_final_value_error':4e-7,'scope':'Known finite acyclic stochastic zero-sum games, exact fixed-policy continuation values. No hidden information.'},
        'limits':['The interval selector is an opt-in library API. Default bot and public deployment are unchanged.',
          'Game certificates in this study are score bounds with optimal continuations; they do not certify a new runnable blind full-game policy.',
          'The 36 deployed searches use four deterministic diagnostic seeds per state, not an independent statistical strength test.',
          'The 128 archived deployed searches are reused to isolate output-selection effects, not counted as new evidence.',
          'Interval inputs must be justified externally. Nonstationary MCTS means are not certified payoff intervals.',
          'Known type weights are explicit assumptions. WorstCase covers only the supplied complete family. Shared information-consistent continuations are required.',
          'Native/WASM parity covers the replayed roots; counterfactual parity for the original case remains the previously documented sampled check.'],
        'sources':[{'url':'https://arxiv.org/abs/1804.09045','title':'Analysis of Hannan Consistent Selection for Monte Carlo Tree Search in Simultaneous Move Games','relevance':'Node-local no-regret on observed rewards alone does not ensure tree-level equilibrium convergence; not a finite-budget theorem for this bot.'}]}
    dump(OUT/'summary.json',result)
    paths=['crates/bot/src/root_policy.rs','crates/bot/src/root_policy/interval.rs','crates/bot/tests/interval_root_games.rs','crates/bot/examples/interval_root_probe.rs','crates/bot/examples/root_class_certificates.rs','web/tests/deployed-policy-class.mjs','tools/policy-class-study.py','tools/policy-class-report.py','findings/deployed-bond-counterexample-v1/blind-confirmation.jsonl']
    paths += [str(p.relative_to(ROOT)) for p in OUT.iterdir() if p.is_file() and p.name not in ['manifest.json','index.html','preview-verification.json']]
    dump(OUT/'manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)})
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','limits','sources']},ensure_ascii=False))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
    audit()
