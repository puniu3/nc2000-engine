"""Recover exact prefixes/finished branches for missing jobs only."""
import argparse,json
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['strength','continuation']);parser.add_argument('jobs',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
assert not list(args.output.iterdir()),'Use a new resume directory'
jobs=json.loads(args.jobs.read_text());sources={}
for job in jobs:
 name=job.removesuffix('.json')+'.partial' if args.mode=='strength' else f'r{job//32:02}-s{(job//16)%2}-t{job%16:02}.partial'
 folder='games' if args.mode=='strength' else 'continuations'
 paths=sorted(p.glob(f'cloud/*/results/files/{folder}/{name}'))
 if not paths:continue
 rows=[json.loads(f.read_text()) for f in paths]
 if args.mode=='strength':
  rows.sort(key=lambda r:len(r['decisions']));best=rows[-1]
  semantic=lambda ds:[{k:v for k,v in d.items() if k!='ns'} for d in ds]
  for r in rows:
   assert r['context']==best['context']
   assert semantic(r['decisions'])==semantic(best['decisions'][:len(r['decisions'])])
   if len(r['decisions'])==len(best['decisions']):
    assert all(r[k]==best[k] for k in ['rng_states','next_key','turn'])
 else:
  best=rows[0]
  for r in rows[1:]:
   assert r['identity']==best['identity']
   for choice,score in r['scores'].items():
    if choice in best['scores']:assert best['scores'][choice]==score and best['turns'][choice]==r['turns'][choice]
    best['scores'][choice]=score;best['turns'][choice]=r['turns'][choice]
 (args.output/name).write_text(json.dumps(best)+'\n')
 sources[name]=[str(f.relative_to(p)) for f in paths]
(args.output/'provenance.json').write_text(json.dumps(sources,indent=2)+'\n')
print(f'{len(sources)} exact checkpoints recovered for {len(jobs)} missing jobs')
