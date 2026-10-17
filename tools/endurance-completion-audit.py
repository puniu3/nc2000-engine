#!/usr/bin/env python3
"""Fail closed against the actual scope of this investigation."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];p=root/'findings/deployed-endurance-v1'
def read(f):return json.loads((p/f).read_text())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
deployment=read('deployment.json')
for f,h in deployment['sha256'].items():assert sha(p/f)==h,f
assert read('deployment-verification.json')['selectedTier']=='strong'
assert read('deployment-verification.json')['iterations']==81000
counts={'games':0,'decisions':0,'searchedDecisions':0,'exactRootPolicies':0}
for i in range(8):
 game=read(f'games/g{i}.json');audit=read(f'games/g{i}-audit.json')
 ds=[json.loads(l) for l in (p/f'games/g{i}.jsonl').read_text().splitlines()];ds=[r for r in ds if r['type']=='decision']
 assert game['outcome'] in ['p1','p2','tie'] and game['outcome']==audit['terminal']
 assert len(ds)==game['decisions']==audit['states'] and len(ds)<2200
 assert game['turn']==audit['turn'];assert audit['exactRootPolicies']>0
 for d in ds:
  for s,r in enumerate(d['results']):
   if not r:continue
   assert any(a['input']==r['best'] for a in d['legal'][s])
   if d['turn']>0:
    assert r['src']=='search' and r['policy']['iterations']==81000
    assert sum(a['visits'] for a in r['policy']['actions'])==81000
    counts['searchedDecisions']+=1
   else:assert r['src']=='table' and r['policy']['iterations']==0
 counts['games']+=1;counts['decisions']+=len(ds);counts['exactRootPolicies']+=audit['exactRootPolicies']
cf_audits={a['file']:a for a in read('continuation-audit.json')}
expected=[];scores={}
for t in range(16):
 for arm in ['baseline','conserve-heal']:
  name=f'g4-d12-{arm}-{t}.json';expected.append(name);r=read('continuations/'+name)
  assert (r['id'],r['decision'],r['side'],r['action'],r['trial'],r['budget'])==('g4',12,1,arm,t,81000)
  assert r['chanceSeed']==970000+t and r['searchSeeds']==[870000+2*t,870001+2*t]
  assert r['outcome'] in ['p1','p2','tie'] and r['score'] in [0,.5,1]
  assert r['score']==(.5 if r['outcome']=='tie' else int(r['outcome']=='p2'))
  a=cf_audits[name];assert a['fullStateAndLegalReplay'] and a['decisions']==r['decisions'] and a['outcome']==r['outcome'];scores[t,arm]=r['score']
assert set(cf_audits)==set(expected)
s=read('summary.json')['counterfactual'];assert s['completedPairs']==s['plannedPairs']==16
assert s['baselineScore']==sum(scores[t,'baseline'] for t in range(16))/16
assert s['alternativeScore']==sum(scores[t,'conserve-heal'] for t in range(16))/16
assert s['delta']==sum(scores[t,'conserve-heal']-scores[t,'baseline'] for t in range(16))/16
h=read('historical/audit.json');assert h['fullStateAndLegalReplay'] and h['firstDivergence']['turn']==10
for name in ['cloud-3','cloud-4']:
 c=read(name+'/execution/state.json');assert c['cleanup_complete'] and c['billable_resources_clear'],name
canary=read('cloud-4/results/files/cloud-canary.json');assert canary['exactRootPolicies']==2 and canary['rootBeliefPreserved']
parity=read('cloud-choice-parity.json');assert len(parity['rows'])==12 and all(r['exactChoice'] for r in parity['rows'])
assert read('case-analysis.json')['completedPairs']==16
long_case=next(c for c in read('case-analysis.json')['longestCompletedGames'] if c['file']=='g4-d12-baseline-4.json')
assert long_case['bothVoluntarilySwitchDecisions']==504
assert long_case['longestJointSwitchRun']['decisions']==483 and long_case['longestJointSwitchRun']['hpAndPPUnchanged']
probe=read('continuation-probes/g4-d12-baseline-4-t500.json')
assert {(r['side'],r['budget']) for r in probe['rows']}=={(s,b) for s in [0,1] for b in [81000,243000,729000]}
assert all(r['best']=='switch 2' and sum(a['visits'] for a in r['policy']['actions'])==r['budget'] for r in probe['rows'])
limit=read('turn-limit-audit.json');assert limit['outcome']=='tie' and limit['endTurn']==1001
assert any('You have hit the turn limit' in line for line in limit['terminalLog'])
for file,h in read('code-provenance.json').items():assert sha(root/file)==h,file
report=root/'previews/deployed-endurance-v1/index.html';assert report.exists()
assert '全16組の比較が完了しました。' in report.read_text()
result={'requirements':{'cleanStartingCheckpoint':'00b4371','actualDeploymentHashed':True,'largerDeployedBudget':81000,'terminalBaselineAndReplay':counts,'conditionalCounterfactualGames':32,'historicalCounterfactualAudited':True,'cloudCleanupVerified':True},'scope':'Exploratory selected endurance matchups and one policy intervention; not global optimality or general strength','hashes':{str(f.relative_to(root)):sha(f) for f in sorted(p.glob('games/g*.json*')) if f.is_file()},'reportSha256':sha(report)}
(p/'completion-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['requirements'],ensure_ascii=False))
