// Full fixed-budget browser searches on four reproducible turn-one states.
import fs from 'node:fs';
import {createServer as http} from 'node:http';
import {createServer} from 'vite';
import {chromium} from '@playwright/test';
const server=http();let vite,browser;
try {
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 vite=await createServer({server:{middlewareMode:true,hmr:false}});server.on('request',vite.middlewares);
 browser=await chromium.launch();const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'networkidle'});
 const rows=await page.evaluate(async()=>{
  const {BotWorker}=await import('/src/bot.ts');
  const {loadEngine,Battle,legalChoices}=await import('/src/engine.ts');const dex=await loadEngine();
  const catalog=await(await fetch('/data/team-pool-v2/team-pool.json')).json();
  const poolJson=await(await fetch('/data/belief-pool-v3/belief-pool.json')).text();const rows=[];
  for(const [human,own] of [[0,1],[3,10],[11,15],[21,23]])for(const budget of [81000,243000]){
   const teams=[human,own].map(i=>JSON.stringify(catalog.teams[i].sets));
   const b=new Battle(dex,...teams,'1,2,3,4');const bot=new BotWorker();
   try{
    const picks=[0,1].map(s=>[s,legalChoices(b,s)[0].input]);
    await bot.newBattle(...teams,'1,2,3,4',{poolJson,side:1,seed:71,compiledPreview:false});
    await bot.search(1,1,77,false);
    bot.apply(picks);for(const [s,p] of picks)b.applyChoice(s,p);
    const result=await bot.search(1,budget,77,true);
    if(result.policy.iterations!==budget||result.src!=='search')throw Error('battle budget drift');
    if(!legalChoices(b,1).some(c=>c.input===result.best))throw Error('illegal choice');
    rows.push({human:catalog.teams[human].id,own:catalog.teams[own].id,picks,budget,ms:result.ms,memoryBytes:result.wasmMemoryBytes,best:result.best});
   }finally{bot.terminate();b.free();}
  }
  return rows;
 });
 fs.writeFileSync('../findings/fixed-budget-v1/browser-battle-timing.json',JSON.stringify(rows,null,2)+'\n');
 console.log(rows.map(({budget,ms,memoryBytes,best})=>({budget,ms,memoryBytes,best})));
}finally{await browser?.close();await vite?.close();server.close();}
