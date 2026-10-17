#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
P=Path('findings/theoretical-headroom-v1');B=Path('findings/deployed-endurance-v1')
rows=[json.loads(z)for f in ['screen-v6.jsonl','wide.jsonl']for z in (P/f).read_text().splitlines()]
keys={(r['id'],r['decision'])for r in rows};assert len(keys)==len(rows)
expected=set();corpus={};inputs=[]
for p in sorted((B/'games').glob('g?.jsonl')):
    inputs.append(p);ds=[json.loads(z)for z in p.read_text().splitlines() if json.loads(z).get('type')=='decision'];corpus[p.stem]=ds
    for i,r in enumerate(ds):
        if r['turn']>0 and all(s['request']=='move'for s in r['state']['sides']):expected.add((p.stem,i))
assert {k for k in keys if len(k[0])==2}==expected
for p in sorted((B/'continuations').glob('*.json')):
    if 'process.' in p.name:continue
    x=json.loads(p.read_text());assert 'trace' in x;corpus[p.stem]=x['trace'];inputs.append(p)
assert len(corpus)==40 and sum(map(len,corpus.values()))==6082
possible_overrides=[]
for r in rows:
    original=corpus[r['id']][r['decision']];assert r['turn']==original['turn']
    for s,pick in original['picks']:
        chosen=r['actions'][s][r['selected'][s]]
        assert chosen==pick or chosen=='move hiddenpower' and pick.startswith('move hiddenpower')
    if 'conserve-heal' in r['id']:
        sides=original['state']['sides'];own=sides[1]['party'][sides[1]['active']];foe=sides[0]['party'][sides[0]['active']]
        if [1,'move icebeam'] in original['picks'] and own['species']=='Blissey' and own['hp']>.6*own['maxhp'] and foe['species']=='Blissey' and own['status']!='frz' and foe['status']!='frz':possible_overrides.append((r['id'],r['decision']))
assert not possible_overrides
for f,n in [('screen-v6.stderr',165),('wide.stderr',len(rows)-165),('focus.stderr',4),('audit.stderr',0)]:
    assert f'audited=6082 roots={n}' in (P/f).read_text()
control=json.loads((P/'positive-control.jsonl').read_text());assert control['synthetic_not_evidence_of_bot_error'] is True and control['win_gap_lower']>.999999
summary=json.loads((P/'summary.json').read_text());assert summary['counts']['roots']==len(rows)
assert not summary['certified_inferior_current_actions']
files=inputs+[Path('crates/bot/examples/theoretical_headroom.rs'),Path('tools/theoretical-headroom-report.py'),Path('tools/theoretical-headroom-audit.py')]+[P/f for f in ['protocol.json','screen-v6.jsonl','wide.jsonl','focus.jsonl','positive-control.jsonl','summary.json','index.html']]
audit={'passed':True,'positive_control_gap':control['win_gap_lower'],'unique_roots':len(rows),'original_idle_roots_complete':len(expected),'historical_states_checked':6082,'terminal_outcomes_checked':40,'possible_policy_override_at_evaluated_roots':possible_overrides,'input_and_output_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}}
(P/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in audit.items()if k!='input_and_output_sha256'},ensure_ascii=False))
