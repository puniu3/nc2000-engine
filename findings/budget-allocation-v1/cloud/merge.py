"""Merge a fixed, disjoint split only after platform and artifact checks."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

here = Path(__file__).resolve().parent
root = here.parent
remote = Path(sys.argv[1]).resolve()
partition = json.loads((here/'partition.json').read_text())
original = json.loads((root/'games/manifest.json').read_text())
parity = json.loads((remote.parent/'platform-parity.json').read_text())
assert parity['passed'] and parity['canary_games'] == 2
for name in partition['canary']:
    a = json.loads((here/'expected'/name).read_text())
    b = json.loads((remote.parent/'canary'/name).read_text())
    for r in [a,b]:
        for d in r['result']['decisions']:
            del d['ns']
    assert a == b, name
sources = [('completed', root/'games', 'mac-arm64'),
           ('local', root/'games-local', 'mac-arm64'),
           ('local_extra', root/'games-local-extra', 'mac-arm64'),
           ('local_recovery', root/'games-local-recovery', 'mac-arm64'),
           ('cloud', remote, 'gcp-x86_64')]
seen = set()
provenance = {}
for label, folder, platform in sources:
    names = partition[label]
    manifest = json.loads((folder/'manifest.json').read_text())
    for key in ['protocol', 'protocol_sha256', 'blind_sha256', 'dex_sha256', 'catalog_sha256', 'prior_sha256', 'schedules']:
        assert manifest[key] == original[key], (label, key)
    if label != 'completed':
        requested = partition['cloud_requested'] if label == 'cloud' else names
        assert set(manifest['selected_jobs']) == set(requested)
        assert set(f.name for f in folder.glob('*-p*-s*.json')) == set(names)
        assert manifest['source_sha256'] == hashlib.sha256((here/'measured-source.rs').read_bytes()).hexdigest()
    for name in names:
        assert name not in seen
        seen.add(name)
        source = folder/name
        row = json.loads(source.read_text())
        assert not row['result']['cap']
        assert name == f'{row["profile"]}-p{row["pair"]:03}-s{row["a_side"]}.json'
        dest = root/'games'/name
        if dest.exists():
            assert dest.read_bytes() == source.read_bytes()
        else:
            shutil.copy2(source, dest)
        provenance[name] = dict(platform=platform, shard=label, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
assert len(seen) == 240
result = dict(platform_parity=parity, game_count=240, provenance=provenance,
              note='One fixed experiment, unchanged schedule and algorithm. Native elapsed time must be split by platform. Canaries excluded from scores.')
(root/'merge.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: 240 unique games merged, exact canaries, identical inputs/schedules; 195 Mac and 45 Google Cloud games.')
