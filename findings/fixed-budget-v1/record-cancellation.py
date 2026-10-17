"""Summarize explicit cleanup of a duplicate-only Cloud shard honestly."""
import datetime as dt,hashlib,json,sys
from pathlib import Path
p=Path(__file__).resolve().parent;name=sys.argv[1];run=p/'cloud'/f'{name}-run'
reason=json.loads((p/'cloud'/f'{name}-cancel-reason.json').read_text())
state_path=run/'execution/state.json';s=json.loads(state_path.read_text())
assert s['cleanup_complete'] and s['billable_resources_clear'] and not s['cleanup_errors']
assert all(not entries for entries in s['last_inventory'].values())
if reason.get('mode','strength')=='strength':
 assert all((p/'strength-merged'/job).exists() for job in reason['jobs'])
else:
 assert all((p/'continuation-merged'/f'r{job//32:02}-s{(job//16)%2}-t{job%16:02}.json').exists() for job in reason['jobs'])
hours=(dt.datetime.fromisoformat(s['cleared_at'])-dt.datetime.fromisoformat(s['created'])).total_seconds()/3600
r=dict(status='cancelled_duplicate_only',cleanup='verified',backend='gcp-batch',machine_type=s['machine'],provisioning=s['provisioning'],returncode=None,allowance_yen=s['allowance_yen'],estimated_usage_yen=round(hours*s['allowance_hourly_usd']*s['spec']['resources']['yen_per_usd'],3),estimate_scope='Local conservative estimate from recorded admission hourly rate and creation-to-verified-cleanup elapsed time; before credits, not invoice',audit_origin='Local summary of explicit spot-batch cleanup; no new computation results collected',cleanup_state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),reason=reason['reason'])
(run/'cancellation-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(r['status'],r['cleanup'],r['estimated_usage_yen'])
