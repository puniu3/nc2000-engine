// Exercise the actual UI -> Game -> Worker budget path at production counts.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createServer as http} from 'node:http';
import {createServer} from 'vite';
import {chromium} from '@playwright/test';
const config=JSON.parse(fs.readFileSync('../data/live-search.json','utf8'));
const requested=process.argv.slice(2);
const tiers=Object.entries(config.battle).filter(([name])=>!requested.length||requested.includes(name));
assert(tiers.length);
const server=http();let vite,browser;const rows=[];
try{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 vite=await createServer({server:{middlewareMode:true,hmr:{server}}});server.on('request',vite.middlewares);
 browser=await chromium.launch();
 for(const [strength,budget] of tiers){
  const context=await browser.newContext();const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>{
   localStorage.setItem('nc2000-locale','en');
   const NativeWorker=window.Worker;window.__budgetRequests=[];window.__budgetResults=[];
   window.Worker=class extends NativeWorker{
    constructor(url,options){super(url,options);this.addEventListener('message',e=>{if(e.data?.t==='result')window.__budgetResults.push(e.data);});}
    postMessage(message,...rest){if(message?.t==='search')window.__budgetRequests.push(message);return super.postMessage(message,...rest);}
   };
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'networkidle'});
  await page.getByRole('combobox',{name:'Bot strength',exact:true}).selectOption(strength);
  await page.getByRole('button',{name:'Start battle',exact:true}).click();
  for(let i=0;i<3;i++)await page.locator('.pick-head[aria-pressed="false"][aria-disabled="false"]').first().click();
  await page.getByRole('button',{name:'Confirm picks',exact:true}).click();
  await page.waitForFunction(()=>window.__budgetResults.some(r=>r.src==='search'),null,{timeout:120000});
  const data=await page.evaluate(()=>({requests:window.__budgetRequests,results:window.__budgetResults}));
  const preview=data.results.find(r=>r.src==='table');const battle=data.results.find(r=>r.src==='search');
  assert(preview,'known UI preview must use compiled selection');assert.equal(JSON.parse(preview.policy).iterations,0);
  assert.equal(JSON.parse(battle.policy).iterations,budget);assert(data.requests.some(r=>r.budget===budget&&r.ponder));assert.deepEqual(errors,[]);
  rows.push({strength,budget,previewIterations:0,battleIterations:JSON.parse(battle.policy).iterations,completedWhileHumanChoosing:true});
  await context.close();
 }
 const path=`${process.env.NC2000_CHECK_OUT ?? '../findings/fixed-budget-release-v1'}/live-strength-choice-check.json`;
 const prior=fs.existsSync(path)?JSON.parse(fs.readFileSync(path,'utf8')):[];
 fs.writeFileSync(path,JSON.stringify([...prior.filter(r=>!rows.some(n=>n.strength===r.strength)),...rows],null,2)+'\n');console.log(rows);
}finally{await browser?.close();await vite?.close();server.close();}
