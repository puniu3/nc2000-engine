import fs from 'node:fs';
import assert from 'node:assert/strict';
import {dir,replay} from './endurance-runtime.mjs';
const r=JSON.parse(fs.readFileSync(`${dir}/historical/g4-d12-conserve-heal.json`,'utf8'));
const {battle,bots,head,decisions}=replay(r.id,r.decision);bots.forEach(b=>b.free());assert.equal(battle.seed(),r.chanceSeed);assert.deepEqual(head.job.searchSeeds,r.searchSeeds);
let divergence=null;
for(let i=0;i<r.trace.length;i++){
 const t=r.trace[i];assert.equal(battle.turn(),t.turn);assert.deepEqual(JSON.parse(battle.stateView()),t.state);
 const old=decisions[r.decision+i];if(!divergence&&JSON.stringify(t.picks)!==JSON.stringify(old.picks)){divergence={turn:t.turn,baseline:old.picks,alternative:t.picks};assert.deepEqual(t.state,old.state);}
 for(const [s,p]of t.picks){assert(JSON.parse(battle.legalChoices(s)).some(a=>a.input===p));battle.applyChoice(s,p);}
}
assert.equal(battle.outcome(),r.outcome);assert.equal(battle.turn(),r.endTurn);assert.equal(r.score,Number(r.outcome===`p${r.side+1}`));
const audit={fullStateAndLegalReplay:true,decisions:r.trace.length,outcome:r.outcome,endTurn:r.endTurn,score:r.score,firstDivergence:divergence};fs.writeFileSync(`${dir}/historical/audit.json`,JSON.stringify(audit,null,2)+'\n');console.log(audit);battle.free();
