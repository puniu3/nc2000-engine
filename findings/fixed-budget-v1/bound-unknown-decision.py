"""Certify the unchanged registered decision over every pending outcome.
This does not create synthetic games or replace the complete-data analysis.
Only applies when pending rows differ at 81k alone; all higher choices agree.
"""
import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
budgets=[81000,243000,729000,2187000]
pairs=[(a,b) for i,a in enumerate(budgets) for b in budgets[i+1:]]
matrices=[{pair:np.zeros((16,2,16)) for pair in pairs} for _ in range(2)]
missing=[]
for root in range(16):
 row=json.loads((p/'unknown'/f'unknown-{root:02}.json').read_text())
 for si,seed in enumerate(row['job']['seeds']):
  choices={s['budget']:s['choice'] for s in row['samples'] if s['seed']==seed and s['budget']>=81000}
  for trial in range(16):
   f=p/'continuation-merged'/f'r{root:02}-s{si}-t{trial:02}.json'
   if f.exists():
    r=json.loads(f.read_text());assert r['root']==root and r['search_seed']==seed and r['trial']==trial and dict(r['choices'])==choices
    for pair in pairs:
     value=0 if r['equal'] else r['scores'][choices[pair[1]]]-r['scores'][choices[pair[0]]]
     for m in matrices:m[pair][root,si,trial]=value
   else:
    assert len({choices[b] for b in budgets[1:]})==1 and choices[81000]!=choices[243000]
    missing.append(root*32+si*16+trial)
    for pair in pairs:
     if pair[0]==81000:
      matrices[0][pair][root,si,trial]=-1
      matrices[1][pair][root,si,trial]=1
# The same frozen resampling bank and thresholds as analyze.py.
draws=np.random.default_rng(20261017012).integers(0,16,(100000,16))
def summarize(m):
 results=[];guard=[]
 for b in budgets[:-1]:
  roots=m[(b,2187000)].mean(axis=(1,2));dist=roots[draws].mean(axis=1);upper=float(np.quantile(dist,1-.05/3))
  results.append(dict(budget=b,reference_gain=float(roots.mean()),ci95=np.quantile(dist,[.025,.975]).tolist(),upper_simultaneous=upper,passes=upper<=.03))
 for pair in pairs:
  roots=m[pair].mean(axis=(1,2));upper=float(np.quantile(roots[draws].mean(axis=1),1-.05/6))
  guard.append(dict(low=pair[0],high=pair[1],gain=float(roots.mean()),upper_simultaneous=upper,passes=upper<=.03))
 selected=next((r['budget'] for r in results if r['passes'] and all(g['passes'] for g in guard if g['low']==r['budget'])),None)
 return dict(results=results,nonmonotonic_guard=guard,selected_budget=selected)
low,high=map(summarize,matrices)
assert low['results'][1:]==high['results'][1:]
invariant=low['selected_budget']==high['selected_budget']
out=dict(complete_conditions=512-len(missing),pending_conditions=missing,optimistic_for_81k=low,pessimistic_for_81k=high,decision_invariant=invariant,certified_budget=low['selected_budget'] if invariant else None,
 proof='Pending rows affect only 81k-versus-higher contrasts. Their score differences are bounded by [-1,1]. Bootstrap means and quantiles are monotone in those differences. Every intermediate completion therefore has the same minimum qualifying budget when both extreme decisions agree. No pending outcome is recorded as a realized score.',numpy_version=np.__version__)
(p/'unknown-decision-bounds.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
