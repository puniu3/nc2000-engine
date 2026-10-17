"""Frozen analyses; never selects on partially completed jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def strength(folder):
    protocol=read(P/'strength-protocol.json');plan=read(P/'strength-analysis.json')
    manifest=read(folder/'manifest.json')
    assert manifest['protocol']==protocol
    assert manifest['protocol_sha256']==sha(P/'strength-protocol.json')
    scores=[];turns=[];changes=[]
    for pair in range(protocol['pairs']):
        records=[read(folder/f'battle243-p{pair:03}-s{side}.json') for side in [0,1]]
        for side,r in enumerate(records):
            assert r['pair']==pair and r['a_side']==side and r['profile']=='battle243'
            assert not r['result']['cap']
            assert r['a_score']==(r['result']['p1_score'] if side==0 else 1-r['result']['p1_score'])
            for d in r['result']['decisions']:
                budget=0 if d['phase']=='forced' else 243000 if d['phase']=='preview' or d['side']==side else 81000
                assert d['iterations']==budget
            turns.append(r['result']['turns'])
        for seat in [0,1]:
            previews=[[d['choice'] for d in r['result']['decisions'] if d['phase']=='preview' and d['side']==seat] for r in records]
            assert previews[0]==previews[1] and len(previews[0])==1
        first,second=[r['result']['decisions'] for r in records]
        for left,right in zip(first,second):
            assert (left['side'],left['key'],left['turn'])==(right['side'],right['key'],right['turn'])
            if left['choice']!=right['choice']:
                high,low=(left,right) if left['side']==0 else (right,left)
                changes.append(dict(pair=pair,side=left['side'],turn=left['turn'],key=left['key'],high_choice=high['choice'],low_choice=low['choice'],phase=left['phase']))
                break
        else:
            assert len(first)==len(second)
        scores.append([r['a_score'] for r in records])
    assert len(list(folder.glob('battle243-p*-s*.json')))==256
    values=np.array(scores).mean(axis=1)
    draws=np.random.default_rng(plan['bootstrap_seed']).integers(0,len(values),(plan['bootstrap_replicates'],len(values)))
    lo,hi=np.quantile(values[draws].mean(axis=1),[.025,.975])
    all_scores=np.array(scores).flatten()
    result=dict(numpy_version=np.__version__,games=256,pairs=128,wins=int(sum(all_scores==1)),ties=int(sum(all_scores==.5)),losses=int(sum(all_scores==0)),score=float(values.mean()),ci95=[float(lo),float(hi)],add243=bool(lo>.5),pair_scores=values.tolist(),first_decision_changes=changes,max_turns=max(turns),checks=dict(complete=True,no_caps=True,budgets=True,same_preview=True),protocol_sha256=sha(P/'strength-protocol.json'))
    (P/'strength-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result|{'pair_scores':'omitted','first_decision_changes':len(changes)},indent=2))

def unknown(folder):
    plan=read(P/'unknown-analysis.json')['continuation'];budgets=plan['candidates'];reference=plan['reference']
    differences={b:np.zeros((16,2,16)) for b in budgets};checkpoints=[]
    all_budgets=[*budgets,reference]
    pairwise={(low,high):np.zeros((16,2,16)) for i,low in enumerate(all_budgets) for high in all_budgets[i+1:]}
    for root in range(16):
        record=read(P/'unknown'/f'unknown-{root:02}.json');seeds=record['job']['seeds']
        assert record['fallback']
        for si,seed in enumerate(seeds):
            selected={s['budget']:s['choice'] for s in record['samples'] if s['seed']==seed}
            checkpoints.append(dict(root=root,seed=seed,choices=selected))
            for trial in range(16):
                r=read(folder/f'r{root:02}-s{si}-t{trial:02}.json')
                assert (r['root'],r['search_seed'],r['trial'])==(root,seed,trial)
                assert dict(r['choices'])=={b:selected[b] for b in [*budgets,reference]}
                if not r['equal']:
                    for (low,high), values in pairwise.items():
                        values[root,si,trial]=r['scores'][selected[high]]-r['scores'][selected[low]]
                for b in budgets:
                    if r['equal']:
                        assert len(set(selected[x] for x in [*budgets,reference]))==1
                    else:
                        differences[b][root,si,trial]=r['scores'][selected[reference]]-r['scores'][selected[b]]
    assert len(list(folder.glob('r*-s*-t*.json')))==512
    draws=np.random.default_rng(20261017012).integers(0,16,(100000,16));results=[]
    for b in budgets:
        roots=differences[b].mean(axis=(1,2));dist=roots[draws].mean(axis=1)
        upper=float(np.quantile(dist,1-.05/3));results.append(dict(budget=b,reference_gain=float(roots.mean()),ci95=np.quantile(dist,[.025,.975]).tolist(),upper_simultaneous=upper,passes=upper<=.03,changed=sum(r['choices'][b]!=r['choices'][reference] for r in checkpoints)))
    guard=[]
    for (low,high),values in pairwise.items():
        roots=values.mean(axis=(1,2));upper=float(np.quantile(roots[draws].mean(axis=1),1-.05/6))
        guard.append(dict(low=low,high=high,gain=float(roots.mean()),upper_simultaneous=upper,passes=upper<=.03))
    primary_accepted=next((r['budget'] for r in results if r['passes']),None)
    accepted=next((r['budget'] for r in results if r['passes'] and all(g['passes'] for g in guard if g['low']==r['budget'])),None)
    result=dict(numpy_version=np.__version__,reference=reference,roots=16,searches=32,trials_per_search=16,results=results,selected_budget=accepted,primary_selected_budget=primary_accepted,nonmonotonic_guard=guard,checkpoints=checkpoints,scope='Relative to tested 2187k reference under 81k continuations; limited custom-team generator, not a universal saturation proof.')
    (P/'unknown-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result|{'checkpoints':'omitted'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['strength','unknown']);parser.add_argument('folder',type=Path);args=parser.parse_args()
    globals()[args.mode](args.folder)
