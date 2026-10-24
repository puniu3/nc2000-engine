// Exercise the production worker with real offline canaries. The Vite override is test-only; production data is untouched.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { createServer as http } from 'node:http';
import { createServer } from 'vite';
import { chromium } from '@playwright/test';
const root='../findings/fixed-budget-release-v1';
const rows=fs.readdirSync(root+'/canary').filter(n=>n.endsWith('.json')).sort().map(n=>JSON.parse(fs.readFileSync(root+'/canary/'+n,'utf8')));
const artifact={format:'nc2000-compiled-preview-v1',c:0.4,build:rows[0].build,priorSha256:crypto.createHash('sha256').update(fs.readFileSync('../data/belief-pool-v3/belief-pool.json')).digest('hex'),dexSha256:crypto.createHash('sha256').update(fs.readFileSync('../data/gen2stadium2.json')).digest('hex'),policies:{...JSON.parse(fs.readFileSync('../data/compiled-preview-v1.json','utf8')).policies,...Object.fromEntries(rows.map(r=>[r.key,r.samples.map(s=>s.choice)]))}};
const server=http();let vite,browser;
try{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 vite=await createServer({plugins:[{name:'compiled-preview-fixture',enforce:'pre',load(id){if(id.endsWith('/data/compiled-preview-v1.json?raw'))return `export default ${JSON.stringify(JSON.stringify(artifact))}`;}}],server:{middlewareMode:true,hmr:{server}}});server.on('request',vite.middlewares);
 browser=await chromium.launch();const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:"networkidle"});
 const results=await page.evaluate(async rows=>{
  const {BotWorker}=await import('/src/bot.ts');const {compiledPreviewPick,compiledPreviewMatches}=await import('/src/compiled-preview.ts');
  if(compiledPreviewPick('stale',rows[0].key,0,.4)!==undefined)throw Error('stale build used');
  if(compiledPreviewPick(rows[0].build,rows[0].key,0,1)!==undefined)throw Error('wrong coefficient used');
  const poolJson=await(await fetch('/data/belief-pool-v3/belief-pool.json')).text();const results=[];
  if(await compiledPreviewMatches(poolJson,'stale embedded dex'))throw Error('stale binary data accepted');
  const run=async(row,scenario,seed=7)=>{
   const bot=new BotWorker();try{
    const foe=structuredClone(row.job.foe),own=structuredClone(row.job.own);
    if(scenario==='hidden-change'){foe[0].item=foe[0].item?'Leftovers':'';foe[0].moves=['Tackle'];foe[0].happiness=0;}
    if(scenario==='item-presence')foe[0].item=foe[0].item?'':'Leftovers';
    if(scenario==='public-gender'){const p=foe.find(p=>['M','F'].includes(p.gender));p.gender=p.gender==='M'?'F':'M';}
    if(scenario==='own-change')own[0].happiness=own[0].happiness===0?255:0;
    await bot.newBattle(JSON.stringify(foe),JSON.stringify(own),'1,2,3,4',{poolJson:scenario==='prior-change'?poolJson+' ':poolJson,side:1,seed});
    const promise=bot.search(1,512,99,true);if(scenario==='early')bot.flush();
    const result=await promise;
    if(['own-change','prior-change','item-presence','public-gender'].includes(scenario)){if(result.src!=='search'||result.policy.iterations!==512)throw Error('mismatched input used table');}
    else {if(result.src!=='table'||result.policy.iterations!==0||result.best!==row.samples[seed%row.samples.length].choice)throw Error('compiled lookup did not match native key');}
    return {scenario,source:result.src,choice:result.best,iterations:result.policy.iterations};
   }finally{bot.terminate();}
  };
  for(const row of rows)for(const scenario of ['early','background','hidden-change','own-change','prior-change','item-presence','public-gender'])results.push(await run(row,scenario));
  return results;
 },rows);
 assert.equal(results.length,14);fs.writeFileSync(root+'/compiled-worker-check.json',JSON.stringify(results,null,2)+'\n');console.log(results);
}finally{await browser?.close();await vite?.close();server.close();}
