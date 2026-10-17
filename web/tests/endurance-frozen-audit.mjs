// Test the observed frozen-action redundancy with production engine transitions.
import fs from 'node:fs';
import {dir,fork} from './endurance-runtime.mjs';
const id='g4',decision=22,side=0,examples=[];let tests=0,stateEqual=0,seedEqual=0;const byAction={};
for(let trial=0;trial<64;trial++){
 const first=fork(id,decision,55000+trial,[1,2]);const acts=[0,1].map(s=>JSON.parse(first.battle.legalChoices(s)).map(a=>a.input));first.bots.forEach(b=>b.free());first.battle.free();
 for(const foe of acts[1]){
  let reference;
  for(const own of acts[0].filter(a=>a.startsWith('move '))){
   const f=fork(id,decision,55000+trial,[1,2]);f.bots.forEach(b=>b.free());f.battle.applyChoice(side,own);f.battle.applyChoice(1-side,foe);
   const row={own,foe,state:JSON.parse(f.battle.stateView()),seed:f.battle.seed(),log:JSON.parse(f.battle.takeNewLog())};f.battle.free();
   if(!reference)reference=row;else {tests++;const stats=byAction[own]??={comparisons:0,visibleStateEqual:0,prngEqual:0};stats.comparisons++;const same=JSON.stringify(reference.state)===JSON.stringify(row.state);if(same){stateEqual++;stats.visibleStateEqual++;}if(reference.seed===row.seed){seedEqual++;stats.prngEqual++;}if(!same&&examples.length<4)examples.push({trial,reference,row});}
  }
 }
}
const result={id,decision,side,trialSeeds:[55000,55063],comparisons:tests,byAction,visibleStateEqual:stateEqual,prngEqual:seedEqual,scope:'One-step stateView and PRNG audit, not equality of all hidden engine fields or all chance outcomes',examples};
fs.writeFileSync(`${dir}/frozen-action-audit.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({...result,examples:examples.length}));
