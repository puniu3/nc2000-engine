import fs from 'node:fs';
import assert from 'node:assert/strict';
import {dir,replay,fork} from './endurance-runtime.mjs';
const [id,ds]=process.argv.slice(2),d=Number(ds);const rt=replay(id,d);const {battle,bots,decisions}=rt;
for(let s=0;s<2;s++)if(decisions[d].results[s]){
 bots[s].observe(battle);for(const n of [81000,243000,729000]){
  bots[s].step(n-bots[s].iterations());const policy=JSON.parse(bots[s].rootPolicy());
  if(n===81000){assert.deepEqual(policy,decisions[d].results[s].policy);assert.equal(bots[s].best(),decisions[d].results[s].best);}
  console.log(JSON.stringify({id,decision:d,turn:battle.turn(),side:s,budget:n,best:bots[s].best(),policy}));
 }
}
bots.forEach(b=>b.free());battle.free();
const f=fork(id,d,456,[123,234]);console.error('Exact fork state preserved, reseeded chance:',f.battle.seed());f.bots.forEach(b=>b.free());f.battle.free();
