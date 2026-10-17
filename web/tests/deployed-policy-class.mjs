import fs from 'node:fs';
import assert from 'node:assert/strict';
import {dex,Battle,BlindSearcher,pool} from './endurance-runtime.mjs';
const dir='../findings/policy-class-v1';
const cases=JSON.parse(fs.readFileSync(`${dir}/fixtures.json`));
const output=process.argv[2]??`${dir}/deployed-searches.jsonl`;fs.writeFileSync(output,'',{flag:'wx'});
for(const c of cases)for(let index=0;index<4;index++){
  const seed=813001+index*104729,b=new Battle(dex,...c.teams.map(JSON.stringify),c.seed);
  const bot=new BlindSearcher(b,0,pool,seed,.4);
  const apply=actions=>{if(actions[0]!==null)bot.observe(b);for(let s=0;s<2;s++)if(actions[s]!==null)b.applyChoice(s,actions[s]);};
  for(const d of c.decisions.slice(0,-1))apply(d.choices.map((i,s)=>i===null?null:JSON.parse(b.legalChoices(s))[i].input));
  for(let k=0;k<c.extra_waits;k++)apply(['move curse','move destinybond']);
  const state=JSON.parse(b.stateView()),chance_seed=b.seed();
  bot.observe(b);const start=performance.now();bot.step(81000);
  const record={id:c.id,index,seed,side:0,state,chance_seed,best:bot.best(),policy:JSON.parse(bot.rootPolicy()),belief:JSON.parse(bot.beliefInfo()),seconds:(performance.now()-start)/1000};
  assert.equal(record.policy.iterations,81000);
  fs.appendFileSync(output,JSON.stringify(record)+'\n');bot.free();b.free();console.error(`${c.id} ${index+1}/4`);
}
