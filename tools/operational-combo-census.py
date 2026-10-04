import pathlib,json,collections,struct,zlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
archive=ROOT/'logs_20260804.zip'
if archive.exists():
 data=archive.read_bytes();offset=0;recovered={};errors=0
 while True:
  offset=data.find(b'PK\x03\x04',offset)
  if offset<0:break
  try:
   header=struct.unpack_from('<4s5H3I2H',data,offset)
   method,size,nl,el=header[3],header[7],header[-2],header[-1]
   name=data[offset+30:offset+30+nl].decode();start=offset+30+nl+el
   raw=data[start:start+size]
   if method==8:raw=zlib.decompress(raw,-15)
   elif method!=0:raise ValueError('unsupported compression')
   if name.endswith('.raw.log'):recovered[pathlib.Path(name).name]=raw.decode()
  except (ValueError,struct.error,zlib.error,UnicodeDecodeError):errors+=1
  offset+=4
 out=ROOT/'tmp/operational-audit/ladder';out.mkdir(parents=True,exist_ok=True)
 for name,raw in recovered.items():
  if any(ln.startswith('|player|') and '|puniu3|' in ln for ln in raw.splitlines()):(out/name).write_text(raw)
 print(json.dumps({'recovered_raw_logs':len(recovered),'corrupt_entries':errors}))
moves=['Perish Song' ,'Mean Look','Encore','Baton Pass','Disable','Destiny Bond','Curse']
sets={}
for label,folder in [('spectator','tmp/corpus-spectator'),('ladder','tmp/operational-audit/ladder')]:
 files=sorted((ROOT/folder).glob('*.raw.log'));counts=collections.Counter();sidecounts=collections.Counter();cases=[]
 for p in files:
  lines=p.read_text().splitlines();seen=set();side_seen=set();turn=0;bot=None;active={};done=set()
  for ln in lines:
   a=ln.split('|')
   if len(a)>3 and a[1]=='player' and a[3]=='puniu3':bot=int(a[2][1])-1
  for ln in lines:
   a=ln.split('|')
   if len(a)>2 and a[1]=='turn':turn=int(a[2])
   if len(a)>3 and a[1] in ['switch','drag']:active[int(a[2][1])-1]=a[3].split(',')[0]
   if len(a)>3 and a[1]=='move' and a[3] in moves:
    seen.add(a[3]);s=int(a[2][1])-1;side_seen.add((a[3],s==bot))
    if bot is not None and s!=bot and a[3] in moves[:4] and a[3] not in done:
     done.add(a[3]);cases.append({'log':str(p.relative_to(ROOT)),'side':bot,'turn':turn,'trigger':a[3],'foe':active.get(s),'own':active.get(bot)})
  counts.update(seen);sidecounts.update(side_seen)
 sets[label]={'games':len(files),'move_games':dict(counts),'opponent_move_games':{m:sidecounts[m,False] for m in moves} if label=='ladder' else None}
 if label=='ladder':(ROOT/'tmp/operational-audit/candidates.json').write_text(json.dumps(cases,indent=2))
for label,f,field in [('catalog','data/team-pool-v2/team-pool.json','drawWeight'),('nash','data/meta-nash-v3/pool-artifact.json','weight')]:
 teams=json.loads((ROOT/f).read_text())['teams'];sets[label]={'teams':len(teams),'move_draw_mass':{m:sum(t[field] for t in teams if any(m in s['moves'] for s in t['sets'])) for m in moves}}
(ROOT/'tmp/operational-audit/census.json').write_text(json.dumps(sets,indent=2))
print(json.dumps(sets,indent=2))
