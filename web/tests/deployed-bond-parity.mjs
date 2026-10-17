import fs from 'node:fs';
import readline from 'node:readline';
import assert from 'node:assert/strict';
import {createGunzip} from 'node:zlib';
import {dex,Battle} from './endurance-runtime.mjs';
const dir='../findings/deployed-bond-counterexample-v1';
const fixture=JSON.parse(fs.readFileSync(`${dir}/screen-fixture.json`));
const proofDir=process.argv[2]??`${dir}/final-proof-v2`;
const replay=JSON.parse(fs.readFileSync(`${proofDir}/replay.json`));
const battle=new Battle(dex,...fixture.teams.map(JSON.stringify),fixture.seed);
let checks=0;
function check(b,state,seed){assert.deepEqual(JSON.parse(b.stateView()),state);assert.equal(b.seed(),seed);checks++;}
function apply(b,actions){for(let s=0;s<2;s++)if(actions[s]!==null){
  assert(JSON.parse(b.legalChoices(s)).some(a=>a.input===actions[s]));b.applyChoice(s,actions[s]);
}}
for(const r of replay.trace){check(battle,r.state,r.seed);apply(battle,r.actions);}
check(battle,replay.root,replay.seed);
apply(battle,[0,1].map(s=>JSON.parse(battle.legalChoices(s))[0].input));
const code=battle.exportKifu(false,0);battle.free();
const spec={schema:'nc2000-fork-v1',info:'blind',position:{schema:'nc2000-position-v1',side:0,turn:replay.root.turn,own_sets:[],sides:[{mons:[]},{mons:[]}]},opponent_team:[],arms:[],exact_replay:{code,round:replay.trace.length}};
let trials=0;const outcomes={},joint=new Set();
const parityPath=`${proofDir}/parity.jsonl`;
const input=fs.existsSync(parityPath)?fs.createReadStream(parityPath):fs.createReadStream(parityPath+'.gz').pipe(createGunzip());
for await(const line of readline.createInterface({input,crlfDelay:Infinity})){
  const r=JSON.parse(line),b=Battle.fromFork(dex,JSON.stringify(spec),r.chance_seed);
  for(const d of r.records){check(b,d.state,d.seed);joint.add(JSON.stringify(d.actions));apply(b,d.actions);}
  check(b,r.final,r.final_seed);const o=b.outcome()??'unfinished';outcomes[o]=(outcomes[o]??0)+1;b.free();trials++;
}
const result={trials,state_and_rng_checks:checks,outcomes,distinct_joint_actions:[...joint],scope:'Complete stateView and PRNG equality for original prefix and 768 deterministic counterfactual trajectories. This checks sampled trajectories, not every enumerated chance branch.'};
fs.writeFileSync(`${dir}/engine-parity.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
