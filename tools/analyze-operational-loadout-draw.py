import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('--dir',type=Path,default=Path('findings/operational-loadout-draw-v1'))
args=parser.parse_args()
protocol=json.loads(Path('findings/operational-loadout-draw-v1/protocol.json').read_text())
summarize=runpy.run_path('tools/analyze-operational-combo-gate.py')['summarize']
report={'complete':True,'default_enabled':False,'results':{}}
for cell,spec in protocol['cells'].items():
    path=args.dir/f'{cell}.jsonl'
    if not path.exists():report['complete']=False;continue
    rows=sorted((json.loads(s) for s in path.read_text().splitlines()),key=lambda r:r['job'])
    assert len(rows)==spec['games'] and [r['job'] for r in rows]==list(range(spec['games'])),cell
    if cell in ['same','onpool']:
        for r in rows:
            norm=lambda a:{k:v for k,v in a.items() if k not in ['elapsed_seconds','policy']}
            assert norm(r['a'])==norm(r['b'])==norm(r['c']),(cell,r['job'])
    comparisons={}
    for first,second in [('a','c'),('b','c'),('a','b')]:
        paired=[dict(r,a=r[first],b=r[second]) for r in rows]
        result=summarize(paired)
        discord=[r[second]['score']-r[first]['score'] for r in rows
                 if r[first]['terminal'] and r[second]['terminal']
                 and r[second]['score']!=r[first]['score']]
        n=len(discord);k=min(sum(d>0 for d in discord),sum(d<0 for d in discord))
        result['terminal_discordant_pairs']=n
        result['terminal_exact_two_sided_p']=min(1.0,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.0
        comparisons[f'{second}-{first}']=result
    arms={}
    for arm in ['a','b','c']:
        arms[arm]={'policy':rows[0][arm]['policy'],
                   'wins':sum(r[arm]['score']==1 and r[arm]['terminal'] for r in rows),
                   'caps':sum(not r[arm]['terminal'] for r in rows),
                   'opponent_bond_games':sum(r[arm]['opponent_bond_moves']>0 for r in rows),
                   'opponent_bond_moves':sum(r[arm]['opponent_bond_moves'] for r in rows),
                   'opponent_switch_events':sum(r[arm]['opponent_switch_events'] for r in rows),
                   'game_seconds_per_own_decision':sum(r[arm]['elapsed_seconds'] for r in rows)/sum(r[arm]['decisions'] for r in rows),
                   'total_turns':sum(r[arm]['turns'] for r in rows)}
    report['results'][cell]={'triplets':len(rows),'arms':arms,'comparisons':comparisons,
                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(args.dir/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'complete':report['complete'],'results':{k:{'wins':{a:x['wins'] for a,x in v['arms'].items()},'C-A':v['comparisons']['c-a']['paired_delta'],'C-A95':v['comparisons']['c-a']['stratified_bootstrap95'],'C-B':v['comparisons']['c-b']['paired_delta']} for k,v in report['results'].items()}},indent=2))
