// Real browser timing and completion counts on held-out unknown previews.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createServer as http } from 'node:http';
import { createServer } from 'vite';
import { chromium } from '@playwright/test';
const jobs=JSON.parse(fs.readFileSync('../findings/fixed-budget-v1/unknown-jobs.json','utf8'));
const server=http();let vite,browser;
try {
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 vite=await createServer({server:{middlewareMode:true,hmr:{server,clientPort:server.address().port}}});server.on('request',vite.middlewares);
 browser=await chromium.launch();const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const rows=await page.evaluate(async jobs=>{
  const {BotWorker}=await import('/src/bot.ts');const {liveBudget,loadStrength,storeStrength}=await import('/src/search-profile.ts');
  if(liveBudget('standard')!==27000||liveBudget('strong')!==81000)throw Error('wrong battle tiers');
  storeStrength('standard');if(loadStrength()!=='standard')throw Error('preference not saved');storeStrength('strong');
  const poolJson=await(await fetch('/data/belief-pool-v3/belief-pool.json')).text();const rows=[];
  for(const job of jobs)for(const budget of [243000,729000]){
   const bot=new BotWorker();try{
    await bot.newBattle(JSON.stringify(job.foe),JSON.stringify(job.own),'1,2,3,4',{poolJson,side:1,seed:94133});
    const result=await bot.search(1,budget,77,true);
    if(result.policy.iterations!==budget||result.src!=='search')throw Error('unknown preview budget drift');
    rows.push({id:job.id,budget,ms:result.ms,memoryBytes:result.wasmMemoryBytes,best:result.best});
   }finally{bot.terminate();}
  }return rows;
 },[jobs[0],jobs[5],jobs[10],jobs[15]]);
 assert.equal(rows.length,8);fs.writeFileSync('../findings/fixed-budget-v1/browser-timing.json',JSON.stringify(rows,null,2)+'\n');console.log(rows);
}finally{await browser?.close();await vite?.close();server.close();}
