import fs from 'node:fs';
import {dir,replay} from './endurance-runtime.mjs';
const [id,ds,ss,action,ts,out]=process.argv.slice(2),d=Number(ds),side=Number(ss),trial=Number(ts);
const {battle,bots,head}=replay(id,d);const chanceSeed=battle.seed(),searchSeeds=head.job.searchSeeds;
let decisions=0;const trace=[];
try{
 while(!battle.outcome()&&decisions<2200){
  const picks=[],state=JSON.parse(battle.stateView());
  for(let s=0;s<2;s++){
   const legal=JSON.parse(battle.legalChoices(s));if(!legal.length)continue;
   bots[s].observe(battle);let pick;
   if(decisions===0&&s===side&&!['baseline','conserve-heal'].includes(action))pick=action;
   else {bots[s].step(81000);if(bots[s].iterations()!==81000)throw Error('budget drift');pick=bots[s].best();}
   if(s===side&&action==='conserve-heal'&&pick==='move softboiled'){const own=state.sides[s].party[state.sides[s].active],foe=state.sides[1-s].party[state.sides[1-s].active];if(own.hp>.6*own.maxhp&&own.status!=='frz'&&foe.species==='Blissey'&&foe.status!=='frz'&&legal.some(x=>x.input==='move icebeam'))pick='move icebeam';}
   if(!legal.some(x=>x.input===pick))throw Error('illegal arm');picks.push([s,pick]);
  }
  trace.push({turn:battle.turn(),picks,state});for(const [s,p]of picks)battle.applyChoice(s,p);decisions++;
 }
 const outcome=battle.outcome(),score=outcome==='tie'?.5:outcome?Number(outcome===`p${side+1}`):null;
 const result={id,decision:d,side,action,trial,budget:81000,scope:'Historical initial future PRNG and original search seeds; action changes can alter future random consumption',chanceSeed,searchSeeds,outcome,score,endTurn:battle.turn(),decisions,trace};
 fs.writeFileSync(out,JSON.stringify(result)+'\n',{flag:'wx'});console.log(id,d,action,trial,score,battle.turn());
}finally{bots.forEach(b=>b.free());battle.free();}
