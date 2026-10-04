import collections,json,pathlib,re,sys,hashlib
root=pathlib.Path.cwd(); corpus=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=True); dex=json.loads((root/'data/gen2stadium2.json').read_text())
toid=lambda s:re.sub('[^a-z0-9]','',s.lower())
alltypes={v['name'] for k,v in dex['typechart'].items() if k not in ('normal','fairy','stellar')}
assert len(alltypes)==16
def candidates(types,flag):
 out=set()
 for atk in alltypes:
  values=[dex['typechart'][t.lower()]['damageTaken'].get(atk,0) for t in types]
  if 3 in values: continue
  exponent=sum({0:0,1:1,2:-1}[v] for v in values)
  if (flag=='-supereffective' and exponent>0) or (flag=='-resisted' and exponent<0):out.add(atk)
 return out
assert candidates(['Grass','Psychic'],'-supereffective')=={'Bug','Dark','Fire','Flying','Ghost','Ice','Poison'}
assert candidates(['Normal'],'-supereffective')=={'Fighting'}
assert 'Ground' not in candidates(['Flying'],'-resisted')
cases=[]; counts=collections.Counter(); files=sorted(corpus.glob('*.raw.log'))
assert files, "no corpus logs"
for index,file in enumerate(files):
 lines=file.read_text().splitlines()
 if any('|Transform|' in x or '|Conversion|' in x or '|Conversion 2|' in x or '|typechange|' in x or '|Foresight|' in x for x in lines):counts['excluded_typechanging_games']+=1;continue
 active={}; allowed={}; history={}; move=None; turn=0; pending=[]; hp_events=[]
 for ln in lines:
  p=ln.split('|')
  if len(p)<3:continue
  tag=p[1]
  if tag in ('switch','drag') and len(p)>3:
   side=int(p[2][1])-1; sp=toid(p[3].split(',')[0]); active[side]=sp; move=None
  elif tag=='turn':
   turn=int(p[2]); move=None
   for ev in pending:
    if active.get(1-ev['side'])==ev['foe']:
     ev.update(turn=turn,own=active.get(ev['side']))
     cases.append(ev)
   pending=[]
  elif tag=='move':
   move=None
   if len(p)>4 and p[3]=='Hidden Power' and '[from]' not in ln:
    attacker=int(p[2][1])-1; defender=int(p[4][1])-1 if p[4].startswith('p') else 1-attacker
    if attacker in active and defender in active:
     move=(attacker,defender,active[attacker],active[defender]);counts['direct_hp_uses']+=1
  elif tag in ('-supereffective','-resisted') and move:
   attacker,defender,sp,target=move
   if int(p[2][1])-1!=defender:continue
   prior=allowed.get((attacker,sp),alltypes)
   nxt=prior & candidates(dex['species'][target]['types'],tag)
   counts['effectiveness_events']+=1
   if not nxt:counts['contradictory_constraints']+=1; continue
   allowed[attacker,sp]=nxt
   history.setdefault((attacker,sp),[]).append({'turn':turn,'target':target,'target_types':dex['species'][target]['types'],'effect':tag,'compatible_types':sorted(candidates(dex['species'][target]['types'],tag))})
   if nxt!=prior:
    counts['constraint_reductions']+=1
    ev={'battle':index,'log':str(file),'side':defender,'trigger':'Hidden Power '+tag,'foe':sp,'observed_turn':turn,'target':target,'allowed_types':sorted(nxt),'evidence':list(history[attacker,sp])}
    pending.append(ev)
(output/'cases.json').write_text(json.dumps(cases,indent=2))
summary={'games':len(files),'counts':dict(counts),'next_turn_cases':len(cases),'species':dict(collections.Counter(c['foe'] for c in cases)),'constraint_sizes':dict(collections.Counter(len(c['allowed_types']) for c in cases)),'corpus_sha256':hashlib.sha256(b''.join(hashlib.sha256(p.read_bytes()).digest() for p in files)).hexdigest(),'typechart_sha256':hashlib.sha256((root/'data/gen2stadium2.json').read_bytes()).hexdigest()}
(output/'census-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
