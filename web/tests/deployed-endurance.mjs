// Production bytes for bot and referee, pinned in deployment.json.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createServer} from 'node:http';
import {chromium} from '@playwright/test';
const dir=path.resolve('../findings/deployed-endurance-v1');
const workerName='bot-worker-mrxcXXq0.js';
const source=fs.readFileSync(`${dir}/assets/${workerName}`,'utf8');
const boundary=source.indexOf('const m5="strong"');assert(boundary>0);
const glue=source.slice(0,boundary)+'\nexport {F as Battle,p as Dex,W as BlindSearcher,r5 as default};';
const files=new Map(fs.readdirSync(`${dir}/assets`).map(n=>['/nc2000-engine/assets/'+n,fs.readFileSync(`${dir}/assets/${n}`)]));files.set('/referee.js',Buffer.from(glue));
const server=createServer((req,res)=>{if(req.url==='/'){res.end('<!doctype html><title>Endurance experiment</title>');return;}const b=files.get(req.url);if(!b){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',req.url.endsWith('.wasm')?'application/wasm':'text/javascript');res.end(b);});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const teams=JSON.parse(fs.readFileSync(`${dir}/team-pool-v2--team-pool.json`)).teams;
const poolJson=fs.readFileSync(`${dir}/belief-pool-v3--belief-pool.json`,'utf8');
const pairs=[[13,14],[14,14],[13,13],[5,21]];
const jobs=pairs.flatMap((pair,i)=>[0,1].map(rep=>({id:`g${i*2+rep}`,pair,seed:`${1701+i},${41+rep},2026,1017`,searchSeeds:[7001+i*10+rep,8001+i*10+rep]})));
const selected=process.argv.slice(2);const todo=jobs.filter(j=>(!selected.length||selected.includes(j.id))&&!fs.existsSync(`${dir}/games/${j.id}.json`));
fs.writeFileSync(`${dir}/protocol.json`,JSON.stringify({scope:'Purposive endurance stress cases, not catalog-wide win rate',budget:81000,preview:'production compiled preview, fallback 729000',jobs,decisionCap:2200,sourceWorker:workerName,referee:'Exact wasm-bindgen prefix from production worker, same downloaded WASM',baselineCommit:'00b4371'},null,2)+'\n');
const browser=await chromium.launch();
try{
 for(let start=0;start<todo.length;start+=2)await Promise.all(todo.slice(start,start+2).map(async job=>{
  const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const trace=`${dir}/games/${job.id}.jsonl`;const prior=fs.existsSync(trace)?fs.readFileSync(trace,'utf8').trim().split('\n').map(JSON.parse).filter(r=>r.type==='decision'):[];
  await page.exposeFunction('record',r=>{fs.appendFileSync(trace,JSON.stringify(r)+'\n');if(r.type==='decision'&&r.decision%10===0)console.log(job.id,'decision',r.decision,'turn',r.turn,r.results.map(x=>x?.best));});
  try{
   const result=await page.evaluate(async({job,teams,poolJson,workerName,prior})=>{
    const {default:init,Dex,Battle}=await import('/referee.js');await init();const dex=new Dex();const sets=job.pair.map(i=>JSON.stringify(teams[i].sets));const battle=new Battle(dex,...sets,job.seed);
    const bots=[0,1].map(side=>{
     const worker=new Worker('/nc2000-engine/assets/'+workerName,{type:'module'});let resolveReady,rejectResult,resolveResult;let belief=null;const ready=new Promise(r=>resolveReady=r);
     worker.onmessage=({data:m})=>{if(m.t==='ready')resolveReady();if(m.t==='belief')belief=m;if(m.t==='error')rejectResult?.(new Error(m.message));if(m.t==='result')resolveResult({...m,policy:JSON.parse(m.policy),belief});};worker.onerror=e=>rejectResult?.(new Error(e.message));
     return {worker,ready,async search(id,budget){await ready;return await new Promise((resolve,reject)=>{resolveResult=resolve;rejectResult=reject;worker.postMessage({t:'search',id,side,budget,seed:job.searchSeeds[side],ponder:false});});}};
    });
    await Promise.all(bots.map(x=>x.ready));for(let side=0;side<2;side++)bots[side].worker.postMessage({t:'battle',p1:sets[0],p2:sets[1],seed:job.seed,searcher:{poolJson,side,seed:job.searchSeeds[side]}});
    if(!prior.length)await window.record({type:'start',job,teamIds:job.pair.map(i=>teams[i].id),teams:sets.map(JSON.parse)});
    let decisions=0;
    try{
     while(!battle.outcome()&&decisions<2200){
      const state=JSON.parse(battle.stateView()),legal=[0,1].map(s=>JSON.parse(battle.legalChoices(s)));
      const replay=prior[decisions];
      const results=await Promise.all(bots.map((bot,s)=>legal[s].length?bot.search(decisions,replay?0:(battle.turn()===0?729000:81000)):null));
      if(replay){if(JSON.stringify(state)!==JSON.stringify(replay.state)||battle.seed()!==replay.seed)throw Error('resume mismatch');for(const [s,p]of replay.picks)battle.applyChoice(s,p);for(const bot of bots)bot.worker.postMessage({t:'apply',picks:replay.picks});battle.takeNewLog();decisions++;continue;}
      const picks=[];for(let s=0;s<2;s++)if(results[s]){const r=results[s];if(!legal[s].some(x=>x.input===r.best))throw Error('illegal action');if(battle.turn()>0&&r.policy.iterations!==81000)throw Error('budget drift '+r.policy.iterations);picks.push([s,r.best]);}
      await window.record({type:'decision',decision:decisions,turn:battle.turn(),seed:battle.seed(),state,legal,results,picks});
      for(const [s,p]of picks)battle.applyChoice(s,p);for(const bot of bots)bot.worker.postMessage({t:'apply',picks});
      await window.record({type:'log',decision:decisions,lines:JSON.parse(battle.takeNewLog())});decisions++;
     }
     return {job,outcome:battle.outcome(),turn:battle.turn(),decisions,state:JSON.parse(battle.stateView()),kifu:battle.exportKifu(false,0)};
    }finally{for(const bot of bots)bot.worker.terminate();battle.free();dex.free();}
   },{job,teams,poolJson,workerName,prior});
   fs.writeFileSync(`${dir}/games/${job.id}.json`,JSON.stringify(result,null,2)+'\n');console.log('DONE',job.id,result.outcome,result.turn);
  }finally{await page.close();}
 }));
}finally{await browser.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
