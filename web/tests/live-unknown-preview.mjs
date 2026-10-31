// Check the selected unknown-preview budget through the real live game UI.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createServer as http} from 'node:http';
import {createServer} from 'vite';
import {chromium} from '@playwright/test';
const config=JSON.parse(fs.readFileSync('../data/live-search.json','utf8'));
const sets=JSON.parse(fs.readFileSync('../findings/fixed-budget-release-v1/unknown-jobs.json','utf8'))[0].foe;
const record={id:'custom-budget-check',name:'Unknown preview check',sets,species:sets.map(s=>s.species),levels:sets.map(s=>s.level??55),savedAt:1};
const server=http();let vite,browser;
try{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 vite=await createServer({server:{middlewareMode:true,hmr:{server}}});server.on('request',vite.middlewares);
 browser=await chromium.launch();const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.addInitScript(record=>{
  localStorage.setItem('nc2000-locale','en');localStorage.setItem('nc2000-custom-teams',JSON.stringify([record]));
  localStorage.setItem('nc2000-start-picks',JSON.stringify({human:{kind:'custom',id:record.id}}));
  const NativeWorker=window.Worker;window.__previewRequests=[];window.__previewResults=[];
  window.Worker=class extends NativeWorker{
   constructor(url,options){super(url,options);this.addEventListener('message',e=>{if(e.data?.t==='result')window.__previewResults.push(e.data);});}
   postMessage(message,...rest){if(message?.t==='search')window.__previewRequests.push(message);return super.postMessage(message,...rest);}
  };
 },record);
 await page.goto(`http://127.0.0.1:${server.address().port}/`,{waitUntil:'networkidle'});
 await page.getByRole('button',{name:'Start battle',exact:true}).click();
 await page.waitForFunction(()=>window.__previewResults.length>0,null,{timeout:600000});
 const data=await page.evaluate(()=>({request:window.__previewRequests[0],result:window.__previewResults[0]}));
 const policy=JSON.parse(data.result.policy);assert.equal(data.result.src,'search');assert.equal(policy.preview,true);
 assert.equal(data.request.budget,config.unknownPreview);assert.equal(policy.iterations,config.unknownPreview);assert(data.request.ponder);assert.deepEqual(errors,[]);
 const result={budget:config.unknownPreview,actualIterations:policy.iterations,source:data.result.src,ms:data.result.ms,memoryBytes:data.result.wasmMemoryBytes,completedWhileHumanChoosing:true};
 fs.writeFileSync(`${process.env.NC2000_CHECK_OUT ?? '../findings/fixed-budget-release-v1'}/live-unknown-preview-check.json`,JSON.stringify(result,null,2)+'\n');console.log(result);
}finally{await browser?.close();await vite?.close();server.close();}
