// Measure the downloaded, unmodified PUBLIC worker on the Mac, not local WASM.
import { createServer } from 'node:http';
import { readFileSync, readdirSync, appendFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium, webkit } from '@playwright/test';
const dir=resolve('../findings/bot-strength-investigation-v1');
const files=new Map(readdirSync(dir).filter(n=>/\.(js|wasm)$/.test(n)).map(n=>['/nc2000-engine/assets/'+n,readFileSync(resolve(dir,n))]));
const originalWorker=[...files.keys()].find(n=>n.includes('bot-worker-'));
const worker='/nc2000-engine/assets/instrument-worker.js';
files.set(worker,Buffer.from(`let memory; for(const name of ['instantiate','instantiateStreaming']) {const original=WebAssembly[name];if(original)WebAssembly[name]=async function(...args){const result=await original.apply(this,args);memory=(result.instance??result).exports.memory??memory;return result;};}const post=self.postMessage.bind(self);self.postMessage=m=>post(m.t==='result'?{...m,wasmMemoryBytes:memory?.buffer.byteLength}:m);await import('${originalWorker}');`));
const server=createServer((req,res)=>{if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><title>Internal benchmark</title>');return;}const b=files.get(req.url);if(!b){res.statusCode=404;res.end();return;}res.setHeader('Content-Type',req.url.endsWith('.wasm')?'application/wasm':'text/javascript');res.end(b);});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const catalog=JSON.parse(readFileSync(resolve(dir,'public-team-pool.json')));
const poolJson=readFileSync(resolve(dir,'public-belief-pool.json'),'utf8');
const fork=readFileSync(resolve('../../nc2000-engine/data/forks/4296-t11.json'),'utf8');
const output=resolve(dir,'public-browser-timing.jsonl');
writeFileSync(output,'',{flag:'wx'});
try {
 for(const [name,engine] of [['chromium',chromium],['webkit',webkit]]) {
  let browser;
  try {browser=await engine.launch({timeout:15000});}catch(e){appendFileSync(output,JSON.stringify({type:'unavailable',browser:name,error:String(e)})+'\n');continue;}
  try {
   const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
   await page.exposeFunction('record',r=>{appendFileSync(output,JSON.stringify({browser:name,version:browser.version(),...r})+'\n');console.log(name,r.case,r.budget,r.search_ms,r.memory);});
   await page.evaluate(async ({worker,teams,poolJson,fork})=>{
    async function run(kind,budget,seed,ponder=false,flushAt=null){
     const w=new Worker(worker,{type:'module'});let onReady;const ready=new Promise(r=>onReady=r);let finish,fail;const result=new Promise((r,j)=>{finish=r;fail=j;});let first=null,maxGap=0,last=null;
     w.onmessage=({data:m})=>{if(m.t==='ready')onReady();if(m.t==='result')finish(m);if(m.t==='error')fail(new Error(m.message));if(m.t==='progress'){const now=performance.now();if(first===null)first=now;if(last!==null)maxGap=Math.max(maxGap,now-last);last=now;}};w.onerror=e=>fail(new Error(e.message));
     let timer,flusher;
     try {
      await ready;
      if(kind==='fork')w.postMessage({t:'fork',fork,seed:11,searcher:{poolJson,side:JSON.parse(fork).position.side,seed,mode:'blind'}});
      else {w.postMessage({t:'battle',p1:teams[0],p2:teams[1],seed:'1,2,3,4',searcher:{poolJson,side:1,seed}});if(kind==='turn1')w.postMessage({t:'apply',picks:[[0,'team 1, 4, 5'],[1,'team 1, 4, 5']]});}
      const start=performance.now();w.postMessage({t:'search',id:1,side:kind==='fork'?JSON.parse(fork).position.side:1,budget,seed,ponder});
      if(flushAt!==null)flusher=setTimeout(()=>w.postMessage({t:'flush'}),flushAt);
      const m=await Promise.race([result,new Promise((_,j)=>timer=setTimeout(()=>j(new Error('120s guard')),120000))]);
      await window.record({type:'timing',case:kind,budget,seed,ponder,flushAt,request_ms:performance.now()-start,search_ms:m.ms,first_progress_ms:first-start,max_progress_gap_ms:maxGap,memory:m.wasmMemoryBytes,iterations:JSON.parse(m.policy).iterations,choice:m.best,policy:JSON.parse(m.policy)});
     }finally{clearTimeout(timer);clearTimeout(flusher);w.terminate();}
    }
    await window.record({type:'environment',userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency});
    for(const seed of [71,72])for(const kind of ['preview','turn1','fork'])for(const budget of (seed===71?[27000,81000,243000]:[81000,27000]))await run(kind,budget,seed);
    await run('turn1',27000,73,true,5000);
    await run('turn1',81000,73,true,5000);
   },{worker,teams:catalog.teams.slice(0,2).map(t=>JSON.stringify(t.sets)),poolJson,fork});
  }finally{await browser.close();}
 }
}finally{server.closeAllConnections();await new Promise(r=>server.close(r));}
