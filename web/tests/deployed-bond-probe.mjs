import fs from 'node:fs';
import assert from 'node:assert/strict';
import {dex,Battle,BlindSearcher,pool} from './endurance-runtime.mjs';

const [fixture='screen', mode='blind', countText='1', output] = process.argv.slice(2);
const input=JSON.parse(fs.readFileSync(`../findings/deployed-bond-counterexample-v1/${fixture}-fixture.json`,'utf8'));
const count=Number(countText);
const plan=JSON.parse(fs.readFileSync('../findings/deployed-bond-counterexample-v1/confirmation-plan.json','utf8'));
assert(['blind','pinned'].includes(mode));
const emit = row => {
  const line=JSON.stringify(row)+'\n';
  if(output)fs.appendFileSync(output,line);else process.stdout.write(line);
};
if(output)fs.writeFileSync(output,'',{flag:'wx'});
for(let index=0;index<count;index++){
  const seeds=[0,1].map(s=>count===128?plan.seeds[index]:(710001+index*104729+s*10427)>>>0);
  const battle=new Battle(dex,...input.teams.map(JSON.stringify),input.seed);
  const bots=[0,1].map(s=>new BlindSearcher(battle,s,pool,seeds[s],.4));
  if(mode==='pinned')for(let s=0;s<2;s++)bots[s].pinOpponent(JSON.stringify(input.teams[1-s]));
  for(const [i,decision] of input.decisions.slice(0,-1).entries()){
    const choices=[0,1].map(s=>decision.choices[s]===null?null:JSON.parse(battle.legalChoices(s))[decision.choices[s]].input);
    for(let s=0;s<2;s++)if(choices[s]!==null)bots[s].observe(battle);
    for(let s=0;s<2;s++)if(choices[s]!==null)battle.applyChoice(s,choices[s]);
  }
  const actual=JSON.parse(battle.stateView()),expected=input.expected_root;
  assert.equal(actual.turn,expected.turn);assert.equal(battle.seed(),expected.prngSeed);
  const mon=p=>({species:p.species.toLowerCase(),hp:p.hp,maxhp:p.maxhp,fainted:p.fainted,
    status:p.status,boosts:p.boosts,item:p.item??'',moves:p.moves.map(m=>({id:m.id,pp:m.pp,disabled:m.disabled}))});
  for(let side=0;side<2;side++){
    assert.equal(actual.sides[side].pokemonLeft,expected.sides[side].pokemonLeft);
    assert.deepEqual(actual.sides[side].party.map(mon),expected.sides[side].pokemon.map(mon));
  }
  for(let side=0;side<(count===128?1:2);side++){
    bots[side].observe(battle);
    const started=performance.now();bots[side].step(81000);
    emit({fixture,mode,index,side,seed:seeds[side],iterations:bots[side].iterations(),best:bots[side].best(),
      policy:JSON.parse(bots[side].rootPolicy()),belief:JSON.parse(bots[side].beliefInfo()),seconds:(performance.now()-started)/1000});
  }
  bots.forEach(b=>b.free());battle.free();
  console.error(`${fixture} ${mode} ${index+1}/${count}`);
}
