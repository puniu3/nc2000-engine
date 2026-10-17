import concurrent.futures
import hashlib
import json
import pathlib
import subprocess

root=pathlib.Path(__file__).resolve().parents[2]
folder=pathlib.Path(__file__).resolve().parent
protocol=json.loads((folder/'confirmation-protocol.json').read_text())
binary=root/'target/release/examples/strength_investigation'
assert hashlib.sha256(binary.read_bytes()).hexdigest()==protocol['binary_sha256']
out=folder/'confirmation';out.mkdir()
def trial(i):
    rows=[]
    for arm,action in enumerate(protocol['actions']):
        destination=out/f'{i:03}-{arm}'
        subprocess.run([str(binary),'trace',str(destination),protocol['case'],action,str(protocol['seed']+i),str(protocol['iterations'])],cwd=root,check=True)
        r=json.loads((destination/'trace.json').read_text())
        assert r['score'] in [0,1] and r['decisions']<1600
        rows.append({'action':action,'result':r})
    (out/f'{i:03}.json').write_text(json.dumps({'trial':i,'seed':protocol['seed']+i,'iterations':protocol['iterations'],'rows':rows})+'\n')
    print(f'trial {i} complete',flush=True)
with concurrent.futures.ThreadPoolExecutor(protocol['threads']) as executor:
    list(executor.map(trial,range(protocol['trials'])))
