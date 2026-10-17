// Check the actual compiled WASM against native evaluation-adapter traces.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {initSync,Dex,Battle,BlindSearcher} from '../crates/wasm/pkg-web/nc2000_wasm.js';
const bytes=fs.readFileSync('crates/wasm/pkg-web/nc2000_wasm_bg.wasm');
initSync({module:bytes});
const dex=new Dex();
const pool=fs.readFileSync('data/belief-pool-v3/belief-pool.json','utf8');
const catalog=JSON.parse(fs.readFileSync('data/team-pool-v2/team-pool.json','utf8'));
const native=JSON.parse(fs.readFileSync('findings/mixed-policy-strength-v1/parity.json','utf8'));
const teams=[0,7].map(i=>JSON.stringify(catalog.teams[i].sets));
let battle,search,previous;
try {
 for(const row of native.records){
  const key=`${row.mode}/${row.seed}`;
  if(key!==previous){search?.free();battle?.free();battle=new Battle(dex,...teams,'1,2,3,4');search=new BlindSearcher(battle,1,pool,row.seed,.4);previous=key;}
  search.observe(battle);search.step(512);
  const decision=JSON.parse(search.commitDecision(row.mode));
  assert.equal(decision.input,row.input);
  assert.equal(decision.policy.source,row.policy.source);
  assert.equal(decision.policy.actions.length,row.policy.actions.length);
  for(let i=0;i<row.policy.actions.length;i++){
   const a=decision.policy.actions[i],b=row.policy.actions[i];
   assert.equal(a.input,b.input);assert(Math.abs(a.probability-b.probability)<1e-12);
  }
  const diagnostics=JSON.parse(search.rootPolicy());
  for(const r of row.diagnostics){const actual=diagnostics.actions.find(a=>a.input===r.input);assert.equal(actual.visits,r.visits);assert(Math.abs(actual.mean-r.mean)<1e-12);}
  if(row.own!==null)battle.applyChoice(0,row.own);
  battle.applyChoice(1,row.input);
 }
 console.log(JSON.stringify({decisions:native.decisions,modes:native.modes,seeds_per_mode:native.seeds_per_mode,
  wasmSha256:createHash('sha256').update(bytes).digest('hex'),
  checks:['actual compiled WASM action equals native evaluation adapter','execution distributions match','all visits and means match'],status:'pass'},null,2));
}finally{search?.free();battle?.free();dex.free();}
