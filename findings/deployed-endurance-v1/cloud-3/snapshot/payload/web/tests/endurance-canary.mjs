import fs from 'node:fs';
import assert from 'node:assert/strict';
import {dir,replay,fork} from './endurance-runtime.mjs';
const {battle,bots,decisions}=replay('g4',12);
assert.deepEqual(JSON.parse(battle.stateView()),decisions[12].state);
for(let s=0;s<2;s++){bots[s].observe(battle);bots[s].step(81000);assert.deepEqual(JSON.parse(bots[s].rootPolicy()),decisions[12].results[s].policy);assert.equal(bots[s].best(),decisions[12].results[s].best);bots[s].free();}battle.free();
const f=fork('g4',12,456,[123,234]);for(let s=0;s<2;s++){f.bots[s].observe(f.battle);assert.equal(f.bots[s].beliefInfo(),f.beliefAtRoot[s]);f.bots[s].free();}f.battle.free();
fs.writeFileSync(`${dir}/cloud-canary.json`,JSON.stringify({exactRootPolicies:2,fullState:true,rootBeliefPreserved:true,platform:process.platform,arch:process.arch,node:process.version})+'\n');
console.log('Production-policy parity and fork history canary passed');
