// Re-evaluate sampled cloud decisions on the Mac using the same production WASM.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {dir,fork} from './endurance-runtime.mjs';
const manifest=JSON.parse(fs.readFileSync(`${dir}/cloud-import.json`,'utf8'));const rows=[];
for(const {file}of manifest.completed){
 const r=JSON.parse(fs.readFileSync(`${dir}/continuations/${file}`,'utf8'));
 const {battle,bots}=fork(r.id,r.decision,r.chanceSeed,r.searchSeeds);
 const samples=new Set([0,Math.floor(r.trace.length/2),r.trace.length-1]);
 for(let d=0;d<r.trace.length;d++){
  const state=JSON.parse(battle.stateView());assert.deepEqual(state,r.trace[d].state);
  for(const [s,expected]of r.trace[d].picks){
   bots[s].observe(battle);
   if(samples.has(d)){
    bots[s].step(81000);let choice=bots[s].best();const raw=choice;
    if(s===r.side&&r.action==='conserve-heal'&&choice==='move softboiled'){
     const own=state.sides[s].party[state.sides[s].active],foe=state.sides[1-s].party[state.sides[1-s].active];
     if(own.hp>.6*own.maxhp&&own.status!=='frz'&&foe.species==='Blissey'&&foe.status!=='frz'&&JSON.parse(battle.legalChoices(s)).some(a=>a.input==='move icebeam'))choice='move icebeam';
    }
    assert.equal(choice,expected,`${file} d${d} side${s}`);rows.push({file,decision:d,turn:battle.turn(),side:s,raw,choice,exactChoice:true});console.log(file,d,s,choice);
   }
  }
  for(const [s,p]of r.trace[d].picks)battle.applyChoice(s,p);
 }
 assert.equal(battle.outcome(),r.outcome);bots.forEach(b=>b.free());battle.free();
}
fs.writeFileSync(`${dir}/cloud-choice-parity.json`,JSON.stringify({linuxNode:manifest.canary.node,linuxArch:manifest.canary.arch,localNode:process.version,localArch:process.arch,rows},null,2)+'\n');
