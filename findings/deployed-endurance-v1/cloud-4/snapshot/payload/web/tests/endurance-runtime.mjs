import fs from 'node:fs';
import path from 'node:path';
export const dir=path.resolve('../findings/deployed-endurance-v1');
const source=fs.readFileSync(`${dir}/assets/bot-worker-mrxcXXq0.js`,'utf8');
const boundary=source.indexOf('const m5="strong"');if(boundary<0)throw Error('deployment glue boundary changed');
const glue=source.slice(0,boundary)+'\nexport {F as Battle,p as Dex,W as BlindSearcher,r5 as default};';
export const {default:init,Dex,Battle,BlindSearcher}=await import('data:text/javascript;base64,'+Buffer.from(glue).toString('base64'));
await init({module_or_path:fs.readFileSync(`${dir}/assets/nc2000_wasm_bg-CrMX1Tng.wasm`)});
export const dex=new Dex(),pool=fs.readFileSync(`${dir}/belief-pool-v3--belief-pool.json`,'utf8');
export function load(id){const rows=fs.readFileSync(`${dir}/games/${id}.jsonl`,'utf8').trim().split('\n').map(JSON.parse);return {head:rows[0],decisions:rows.filter(r=>r.type==='decision')};}
export function replay(id,target,seeds){
 const {head,decisions}=load(id);let battle=new Battle(dex,...head.teams.map(JSON.stringify),head.job.seed);
 const bots=[0,1].map(s=>new BlindSearcher(battle,s,pool,(seeds??head.job.searchSeeds)[s],.4));
 for(const r of decisions.slice(0,target)){
  for(let s=0;s<2;s++)if(r.results[s])bots[s].observe(battle);
  for(const [s,p]of r.picks)battle.applyChoice(s,p);
 }
 return {battle,bots,head,decisions};
}
export function fork(id,target,chanceSeed,seeds){
 const {head,decisions}=load(id);const b=new Battle(dex,...head.teams.map(JSON.stringify),head.job.seed);
 // Include the target action, so the replay format accepts a fork at target.
 for(const r of decisions.slice(0,target+1))for(const [s,p]of r.picks)b.applyChoice(s,p);
 const code=b.exportKifu(false,0);b.free();
 const runtime=replay(id,target,seeds);
 const spec={schema:'nc2000-fork-v1',info:'blind',position:{schema:'nc2000-position-v1',side:0,turn:decisions[target].turn,own_sets:[],sides:[{mons:[]},{mons:[]}]},opponent_team:[],arms:[],exact_replay:{code,round:target}};
 const battle=Battle.fromFork(dex,JSON.stringify(spec),chanceSeed);
 if(battle.stateView()!==runtime.battle.stateView())throw Error('fork changed state');
 // Ingest the final historical log before fromFork clears it. The next
 // observe on the empty fork resets the log cursor, preserving all reveals.
 // This consumes one extra search-seed draw without searching, identically
 // in both arms; trial search seeds are fresh and unrelated to the source game.
 for(let side=0;side<2;side++)if(decisions[target].results[side])runtime.bots[side].observe(runtime.battle);
 runtime.beliefAtRoot=runtime.bots.map(b=>b.beliefInfo());
 runtime.battle.free();runtime.battle=battle;return runtime;
}
