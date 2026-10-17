#!/usr/bin/env python3
"""Derive reproducible observations from the saved production traces."""
import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1];src=root/'findings/deployed-endurance-v1'
def load_game(g):
 rows=[json.loads(l) for l in (src/'games'/f'{g}.jsonl').read_text().splitlines()]
 return rows[0],[r for r in rows if r['type']=='decision'],[r for r in rows if r['type']=='log']
head,ds,logs=load_game('g4');events=[];full_heals=[];consecutive=[]
for r in ds:
 for side,action in r['picks']:
  board=r['state']['sides'][side]
  if board['active'] is None:continue
  own=board['party'][board['active']]
  if action=='move softboiled' and own['hp']==own['maxhp'] and own['status']!='frz':full_heals.append({'turn':r['turn'],'decision':r['decision'],'side':side,'hp':own['hp'],'pp':next(m['pp'] for m in own['moves'] if m['id']=='softboiled')})
for heal in full_heals:
 lines=next(r['lines'] for r in logs if r['decision']==heal['decision'])
 mon=f"p{heal['side']+1}a: Blissey"
 at=next(i for i,l in enumerate(lines) if l.startswith(f'|move|{mon}|Soft-Boiled|'))
 stop=next((i for i in range(at+1,len(lines)) if lines[i].startswith('|move|')),len(lines))
 block=lines[at:stop]
 current=heal['hp']
 for l in lines[:at]:
  if l.startswith((f'|-damage|{mon}|',f'|-heal|{mon}|')):
   value=l.split('|')[3].split()[0]
   if value.endswith('/'+str(heal['hp'])):current=int(value.split('/')[0])
 recovered=0
 for l in block:
  if l.startswith(f'|-heal|{mon}|'):
   value=l.split('|')[3].split()[0]
   if value.endswith('/'+str(heal['hp'])):recovered=int(value.split('/')[0])-current
 heal.update(failed=f'|-fail|{mon}' in block,hpRecovered=recovered,
             opponentChoice=dict(next(d for d in ds if d['decision']==heal['decision'])['picks']).get(1-heal['side']))
assert sum(h['failed'] for h in full_heals)==9
assert [h['hpRecovered'] for h in full_heals if not h['failed']]==[35,35]
for row in logs:
 d=next(d for d in ds if d['decision']==row['decision'])
 for l in row['lines']:
  if any(v in l for v in ['|-status|','|-curestatus|','|faint|','|win|']):events.append({'turn':d['turn'],'line':l})
probes=[]
for f in sorted((src/'probes').glob('*.jsonl')):
 rows=[json.loads(l) for l in f.read_text().splitlines()]
 for side in [0,1]:
  selected=[r for r in rows if r['side']==side]
  probes.append({'case':f.stem,'side':side,'choices':{str(r['budget']):r['best'] for r in selected},'values':{str(r['budget']):{a['input']:a['mean'] for a in r['policy']['actions']} for r in selected}})
frozen=json.loads((src/'frozen-action-audit.json').read_text())
a={'case':'g4','terminalTurn':129,'fullHPSoftBoiledChoices':full_heals,'timeline':events,'budgetProbes':probes,'frozenActionTransitions':{k:v for k,v in frozen.items() if k!='examples'},'freezeResourceCheckpoints':[{ 'turn':t,'sides':[{'hp':next(p for p in r['state']['sides'][side]['party'] if p['species']=='Blissey')['hp'],'damagePP':sum(m['pp'] for m in next(p for p in r['state']['sides'][side]['party'] if p['species']=='Blissey')['moves'] if m['id'] in ['thunder','icebeam'])} for side in [0,1]]} for t in [20,33] for r in ds if r['turn']==t],'interpretation':[
 {'status':'observed','claim':'In the 129-turn mirror, the bot spent Soft-Boiled PP at full HP and spent many turns selecting Counter.'},
 {'status':'observed','claim':'At turn 10, side1 chooses Soft-Boiled at81k but Ice Beam at243k and729k. At turn53, side0 still chooses full-HP Soft-Boiled at729k.'},
 {'status':'inference','claim':'Some early-healing choices are sensitive to compute, but increasing compute up to729k does not remove the late full-HP healing pattern. Resource valuation and long-horizon continuation quality are candidates, not isolated causes.'},
 {'status':'observed','claim':'Frozen Blissey remained in, thawed after turn32, and recovered. Keeping a frozen tank in was not automatically a wasted turn.'},
 {'status':'limited_transition_evidence','claim':'At the frozen root, Thunder, Ice Beam and Soft-Boiled had matching next stateView and PRNG for all320 tested foe-action/seed combinations per comparison. Counter priority changed random consumption. This is not a proof over hidden fields or all states.'},
 {'status':'hypothesis','claim':'Deduplicating truly equivalent incapacitated actions may improve budget allocation, but requires a semantic equivalence proof and a strength test before shipping.'},
 {'status':'scope','claim':'Eight purposively selected games and one conditional strategy experiment do not establish catalog-wide frequency, general playing strength, or global optimality.'}
]}
(src/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'fullHPHeals':len(full_heals),'budgetProbeRoots':len(probes),'timelineEvents':len(events)}))
