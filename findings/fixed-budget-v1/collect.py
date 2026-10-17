"""Merge deterministic completed jobs, retaining provenance and rejecting drift.
Does not inspect aggregate scores; interrupted jobs remain missing.
"""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def strength():
    folders = sorted(P.glob('cloud/*/results/files/games')) + sorted(P.glob('strength-local-*')) + [P/'strength-canary']
    expected = [f'battle243-p{pair:03}-s{side}.json' for pair in range(128) for side in range(2)]
    out = P/'strength-merged'
    identities = ['protocol', 'protocol_sha256', 'blind_sha256', 'dex_sha256', 'catalog_sha256', 'prior_sha256', 'schedules']
    manifest = None
    records, provenance, incomplete = {}, {}, []
    for folder in folders:
        if not (folder/'manifest.json').exists():
            continue
        current = read(folder/'manifest.json')
        reference_source=sha(P.parent.parent/'crates/bot/examples/budget_allocation.rs')
        if current['source_sha256'] != reference_source:
            proof=read(P/'resume-validation.json')
            assert proof['passed'] and proof['schedule_parity']
            assert current['source_sha256']==proof['source_sha256']
            assert current['reference_source_sha256']==proof['reference_source_sha256']==reference_source
        if manifest is None:
            manifest = current
        assert all(current[k] == manifest[k] for k in identities), f'Mixed experiment: {folder}'
        for path in sorted(folder.glob('battle243-p*-s*.json')):
            assert path.name in expected
            try:
                record = read(path)
            except json.JSONDecodeError:
                incomplete.append(str(path.relative_to(P)))
                continue
            semantic = json.loads(json.dumps(record))
            for decision in semantic['result']['decisions']:
                del decision['ns']
            if path.name in records:
                assert records[path.name] == semantic, f'CPU/repeat trajectory mismatch: {path}'
            else:
                records[path.name] = semantic
                write(out/path.name, record)
            provenance.setdefault(path.name, []).append({'path': str(path.relative_to(P)), 'sha256': sha(path), 'binary_sha256': current['binary_sha256'], 'source_sha256': current['source_sha256']})
    if manifest:
        write(out/'manifest.json', manifest)
    write(out/'provenance.json', provenance)
    missing = [name for name in expected if name not in records]
    write(P/'strength-missing.json', missing)
    return {'complete': len(records), 'expected': len(expected), 'duplicates_verified': sum(len(v)-1 for v in provenance.values()), 'incomplete_files': incomplete}


def continuation():
    folders = [P/'continuations', P/'continuation-equal'] + sorted(P.glob('cloud/*/results/files/continuations')) + sorted(P.glob('continuation-local-*'))
    out = P/'continuation-merged'
    records, provenance = {}, {}
    imports=read(P/'native-continuation-imports.json') if (P/'native-continuation-imports.json').exists() else {}
    for folder in folders:
        for path in sorted(folder.glob('r*-s*-t*.json')):
            if folder==P/'continuations' and path.name in imports:
                try:
                    assert read(path)==read(P/imports[path.name]), f'Imported/native continuation mismatch: {path}'
                except json.JSONDecodeError:
                    pass  # The old native writer may currently be replacing a cache copy.
                continue  # Count its preserved Cloud source, not this skip-cache copy.
            try:
                record = read(path)
            except json.JSONDecodeError:
                continue  # The original local equal-choice writer was not atomic.
            if path.name in records:
                assert record == records[path.name], f'CPU/repeat continuation mismatch: {path}'
            else:
                records[path.name] = record
                write(out/path.name, record)
            provenance.setdefault(path.name, []).append({'path':str(path.relative_to(P)), 'sha256':sha(path)})
    missing = [i for i in range(512) if f'r{i//32:02}-s{(i//16)%2}-t{i%16:02}.json' not in records]
    write(out/'provenance.json', provenance)
    write(P/'continuation-missing.json', missing)
    return {'complete':len(records), 'expected':512, 'duplicates_verified':sum(len(v)-1 for v in provenance.values()), 'simulated_duplicates_verified':sum(len(v)-1 for name,v in provenance.items() if not records[name]['equal'])}


if __name__ == '__main__':
    summary = {'strength':strength(), 'continuation':continuation()}
    write(P/'progress.json',summary)
    print(json.dumps(summary,indent=2))
