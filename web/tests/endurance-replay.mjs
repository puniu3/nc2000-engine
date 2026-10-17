import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
const dir=path.resolve('../findings/deployed-endurance-v1');
const source=fs.readFileSync(`${dir}/assets/bot-worker-mrxcXXq0.js`,'utf8');
const glue=source.slice(0,source.indexOf('const m5="strong"'))+'\nexport {F as Battle,p as Dex,W as BlindSearcher,r5 as default};';
const {default:init,Dex,Battle,BlindSearcher}=await import('data:text/javascript;base64,'+Buffer.from(glue).toString('base64'));
await init({module_or_path:fs.readFileSync(`${dir}/assets/nc2000_wasm_bg-CrMX1Tng.wasm`)});
const dex=new Dex(),pool=fs.readFileSync(`${dir}/belief-pool-v3--belief-pool.json`,'utf8');
for(const id of process.argv.slice(2)){
 const rows=fs.readFileSync(`${dir}/games/${id}.jsonl`,'utf8').trim().split('\n').map(JSON.parse),head=rows[0];
 const b=new Battle(dex,...head.teams.map(JSON.stringify),head.job.seed),bots=[0,1].map(s=>new BlindSearcher(b,s,pool,head.job.searchSeeds[s],.4));
 let checked=0,parity=0;
 for(const row of rows.filter(r=>r.type==='decision')){
  assert.deepEqual(JSON.parse(b.stateView()),row.state);assert.equal(b.seed(),row.seed);
  for(let s=0;s<2;s++)if(row.results[s]){
   bots[s].observe(b);
   if(row.turn>0&&[1,5,10,20,50,100].includes(row.decision)){
    bots[s].step(81000);assert.equal(bots[s].best(),row.results[s].best);assert.deepEqual(JSON.parse(bots[s].rootPolicy()),row.results[s].policy);parity++;
   }
  }
  for(const [s,p]of row.picks)b.applyChoice(s,p);checked++;
 }
 const result={job:head.job,outcome:b.outcome(),turn:b.turn(),decisions:checked,state:JSON.parse(b.stateView()),kifu:b.exportKifu(false,0)};
 if(result.outcome&&!fs.existsSync(`${dir}/games/${id}.json`))fs.writeFileSync(`${dir}/games/${id}.json`,JSON.stringify(result,null,2)+'\n');
 const audit={id,states:checked,exactRootPolicies:parity,terminal:result.outcome,turn:result.turn};
 fs.writeFileSync(`${dir}/games/${id}-audit.json`,JSON.stringify(audit,null,2)+'\n');console.log(audit);bots.forEach(b=>b.free());b.free();
}
dex.free();
