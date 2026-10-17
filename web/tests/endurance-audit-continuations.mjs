import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {dir,fork} from './endurance-runtime.mjs';
const reports=[];
for(const name of fs.readdirSync(`${dir}/continuations`).filter(n=>/^g4-d12-(baseline|conserve-heal)-\d+\.json$/.test(n))){
 const r=JSON.parse(fs.readFileSync(`${dir}/continuations/${name}`,'utf8'));
 assert.equal(r.budget,81000);assert.equal(r.chanceSeed,970000+r.trial);assert.deepEqual(r.searchSeeds,[870000+2*r.trial,870001+2*r.trial]);
 const {battle,bots}=fork(r.id,r.decision,r.chanceSeed,r.searchSeeds);bots.forEach(b=>b.free());
 for(const t of r.trace){assert.equal(battle.turn(),t.turn);assert.deepEqual(JSON.parse(battle.stateView()),t.state);for(const [s,p]of t.picks){assert(JSON.parse(battle.legalChoices(s)).some(a=>a.input===p));battle.applyChoice(s,p);}}
 assert.equal(battle.outcome(),r.outcome);assert.equal(battle.turn(),r.endTurn);assert.equal(r.trace.length,r.decisions);assert.equal(r.score,r.outcome==='tie'?.5:Number(r.outcome===`p${r.side+1}`));
 reports.push({file:name,decisions:r.decisions,outcome:r.outcome,fullStateAndLegalReplay:true});battle.free();
}
fs.writeFileSync(`${dir}/continuation-audit.json`,JSON.stringify(reports,null,2)+'\n');console.log(JSON.stringify({games:reports.length,decisions:reports.reduce((n,r)=>n+r.decisions,0)}));
