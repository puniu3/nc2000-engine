"""Stage only public app assets and the completed study for the shared hub."""
import json, shutil, zipfile, sys, subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
repo=P.parent.parent
app=Path.home()/'.cache/nc2000-fixed-preview'
report=Path.home()/'.cache/nc2000-fixed-report'
app_only='--app-only' in sys.argv
report_only='--report-only' in sys.argv
assert not (app_only and report_only)
if not app_only:assert (P/'strength-result.json').exists() and (P/'unknown-result.json').exists()
assert json.loads((P/'validation.json').read_text()).get('production_build')=='passed'
for dest in [app,report]: dest.mkdir(parents=True,exist_ok=True)
if not report_only:
 shutil.copytree(repo/'web/dist',app,dirs_exist_ok=True)
 files=['gen2stadium2.json','i18n-ja.json','belief-prior-v0.sample.json','team-pool-v2/team-pool.json','belief-pool-v3/belief-pool.json','meta-nash-v3/pool-artifact.json']
 for f in files:
  dest=app/'data'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/'data'/f,dest)
 for f in ['meta-pool-v0','forks']:shutil.copytree(repo/'data'/f,app/'data'/f,dirs_exist_ok=True)
if not app_only:
 subprocess.run(['python3',str(repo/'tools/render-preview-cache-doc.py'),str(report)],check=True)
if app_only:
 print(app);raise SystemExit(0)
share=['strength-protocol.json','strength-analysis.json','strength-result.json','unknown-analysis.json','unknown-nonmonotonic-guard.json','unknown-result.json','validation.json','compiled-coverage.json','browser-timing.json','browser-battle-timing.json','live-strength-choice-check.json','cloud-policy-identity.json','live-unknown-preview-check.json','selection-case-result.json','shared-preview-check.json','index.html']
for f in share:shutil.copy2(P/f,report/f)
receipts=[]
for folder in sorted((P/'cloud').glob('*-run')):
 f=next((f for f in [folder/'receipt.json',folder/'cancellation-audit.json'] if f.exists()),None)
 if f is None:continue
 r=json.loads(f.read_text());receipts.append(dict(job=f.parent.name,**{k:r.get(k) for k in ['status','cleanup','backend','machine_type','provisioning','returncode','allowance_yen','estimated_usage_yen','estimate_scope']}))
(report/'cloud-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
artifact=json.loads((repo/'data/compiled-preview-v1.json').read_text());artifact.pop('policies')
(report/'compiled-metadata.json').write_text(json.dumps(artifact,indent=2)+'\n')
with zipfile.ZipFile(report/'evidence.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in share+['analyze.py','collect.py','report.py','bake-jobs.json','unknown-jobs.json','resume-validation.json','resume-real-canary.json','record-cancellation.py','unknown-bounds-plan.json','unknown-decision-bounds.json','bound-unknown-decision.py','configuration-decision.json','selection-case-plan.json','selection-case.py','finalize.py']:
  z.write(P/name,name)
 for folder in ['strength-merged','continuation-merged','unknown']:
  for f in sorted((P/folder).glob('*.json')):z.write(f,str(f.relative_to(P)))
 for name in ['budget_allocation.rs','budget_resume.rs','compile_preview.rs','preview_continuation.rs']:
  z.write(repo/'crates/bot/examples'/name,'harness/'+name)
 for name in ['cloud-receipts.json','compiled-metadata.json']:z.write(report/name,name)
print(json.dumps(dict(app=str(app),report=str(report)),indent=2))
