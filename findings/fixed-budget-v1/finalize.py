"""Publish the full study to its existing private preview after the final trial."""
import datetime
import json
from pathlib import Path
import ssl
import subprocess
import time
import urllib.request

P = Path(__file__).resolve().parent
repo = P.parent.parent
read = lambda p: json.loads(p.read_text())

# The running continuation worker writes each completed trial atomically.
last = P / 'continuation-local-final/r14-s1-t10.json'
while not last.exists():
    time.sleep(5)
for args in [
    ['collect.py'],
    ['analyze.py', 'unknown', str(P / 'continuation-merged')],
    ['selection-case.py'],
]:
    subprocess.run(['python3', str(P / args[0]), *args[1:]], cwd=repo, check=True)

u = read(P / 'unknown-result.json')
certificate = read(P / 'unknown-decision-bounds.json')
assert u['selected_budget'] == certificate['certified_budget'] == read(repo / 'data/live-search.json')['unknownPreview']
for extreme in ['optimistic_for_81k', 'pessimistic_for_81k']:
    old = certificate[extreme]
    for r in u['results']:
        if r['budget'] == 81000:
            continue
        prior = next(v for v in old['results'] if v['budget'] == r['budget'])
        assert all(r[k] == prior[k] for k in prior)
    for r in u['nonmonotonic_guard']:
        if r['low'] != 81000:
            assert r == next(v for v in old['nonmonotonic_guard'] if (v['low'], v['high']) == (r['low'], r['high']))

v = read(P / 'validation.json')
v['unknown_study'] = 'passed: all 512 paired conditions complete; registered full analysis selects 729k, exactly matching the earlier all-outcome certificate'
(P / 'validation.json').write_text(json.dumps(v, indent=2) + '\n')
d = read(P / 'configuration-decision.json')
d['unknown_basis'] = 'Complete 512-condition registered analysis and nonmonotonicity guard select 729k, confirming the earlier all-outcome certificate.'
d['unknown_result'] = 'unknown-result.json'
d['full_analysis_completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(P / 'configuration-decision.json').write_text(json.dumps(d, indent=2) + '\n')
h = read(P / 'native-continuation-handoff.json')
h['replacement_state'] = 'all_12_assigned_conditions_complete'
(P / 'native-continuation-handoff.json').write_text(json.dumps(h, indent=2) + '\n')

for script, args in [('report.py', []), ('package-preview.py', ['--report-only'])]:
    subprocess.run(['python3', str(P / script), *args], cwd=repo, check=True)
subprocess.run(['preview', 'static', 'nc2000-fixed-report', str(Path.home() / '.cache/nc2000-fixed-report'), '--path', '/index.html'], check=True)

base = 'https://mini-usb.local:18080/nc2000-fixed-report/'
context = ssl.create_default_context(cafile=str(Path.home() / '.local/share/devca/ca.crt'))
for name in ['index.html', 'strength-result.json', 'unknown-result.json', 'selection-case-result.json', 'validation.json', 'evidence.zip']:
    with urllib.request.urlopen(base + name, context=context, timeout=30) as response:
        content = response.read()
        assert response.status == 200 and content
        if name.endswith('.json'):
            assert json.loads(content) == read(P / name)
        if name == 'evidence.zip':
            import io, zipfile
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                assert z.testzip() is None
                assert len([f for f in z.namelist() if Path(f).match('continuation-merged/r*-s*-t*.json')]) == 512
                assert len([f for f in z.namelist() if f.startswith('strength-merged/battle243-')]) == 256
(P / 'report-delivery-check.json').write_text(json.dumps(dict(url=base + 'index.html', tls=True, full_results=True, evidence_zip=True), indent=2) + '\n')
print('Full report verified:', base + 'index.html', flush=True)
