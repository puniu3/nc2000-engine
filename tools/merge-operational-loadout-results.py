import argparse,json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,default=Path('tmp/loadout-draw/frozen'))
parser.add_argument('--corrections',type=Path,default=Path('tmp/loadout-draw/corrected-c'))
parser.add_argument('--out',type=Path,default=Path('findings/operational-loadout-draw-v1'))
a=parser.parse_args();protocol=json.loads((a.out/'protocol.json').read_text());discarded=[];counts={}
metadata=['job','mode','iterations','seed','battle_seed','bot_seed','opp_seed','side','own_team','opponent_team']
for cell,spec in protocol['cells'].items():
    source=a.source/f'{cell}.jsonl'
    rows=sorted([json.loads(s) for s in source.read_text().splitlines()],key=lambda r:r['job'])
    assert len(rows)==spec['games'] and [r['job'] for r in rows]==list(range(spec['games']))
    count=0
    for row in rows:
        if cell=='same':continue
        path=a.corrections/f'{cell}-{row["job"]}.jsonl'
        if row.get('implementation')!='compatible-gender-v1':
            assert path.exists(),f'missing corrected C: {cell} {row["job"]}'
            corrected=json.loads(path.read_text())
            assert corrected['implementation']=='compatible-gender-v1'
            for k in metadata:assert row[k]==corrected[k],(cell,row['job'],k)
            discarded.append(dict(cell=cell,**{k:row[k] for k in metadata},c=row['c']))
            row['c']=corrected['c'];count+=1
        elif path.exists():
            corrected=json.loads(path.read_text())
            norm=lambda x:{k:v for k,v in x.items() if k!='elapsed_seconds'}
            assert norm(row['c'])==norm(corrected['c']),(cell,row['job'],'duplicate C mismatch')
        row['c_implementation']='compatible-gender-v1'
    counts[cell]=count
    (a.out/f'{cell}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(a.out/'superseded-c.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in discarded))
print(json.dumps({'corrected_replacements':counts,'superseded_count':len(discarded)}))
