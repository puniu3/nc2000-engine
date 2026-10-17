import {createServer as http} from 'node:http';
import {createServer} from 'vite';
import {chromium} from '@playwright/test';
import {fileURLToPath} from 'node:url';
import {readFileSync,writeFileSync} from 'node:fs';
const root=fileURLToPath(new URL('..',import.meta.url));
const dir=fileURLToPath(new URL('../../findings/bot-strength-investigation-v1/',import.meta.url));
const reference=readFileSync(dir+'public-browser-timing.jsonl','utf8').trim().split('\n').map(JSON.parse).filter(r=>r.type==='timing'&&!r.ponder&&r.seed===71&&r.budget===27000);
const server=http();await new Promise(r=>server.listen(0,'127.0.0.1',r));
let vite,browser;
try{
 vite=await createServer({root,server:{middlewareMode:true,hmr:{server,clientPort:server.address().port}}});server.on('request',vite.middlewares);browser=await chromium.launch();const page=await browser.newPage();await page.goto(`http://127.0.0.1:${server.address().port}/`);
 const results=await page.evaluate(async reference=>{
  const {BotWorker}=await import('/src/bot.ts');
  const catalog=await(await fetch('/data/team-pool-v2/team-pool.json')).json();const teams=catalog.teams.slice(0,2).map(t=>JSON.stringify(t.sets));const poolJson=await(await fetch('/data/belief-pool-v3/belief-pool.json')).text();const fork=await(await fetch('/data/forks/4296-t11.json')).text();const results=[];
  for(const r of reference){const w=new BotWorker();try{const side=r.case==='fork'?JSON.parse(fork).position.side:1;if(r.case==='fork')await w.newFork(fork,11,{poolJson,side,seed:r.seed,mode:'blind'});else {await w.newBattle(teams[0],teams[1],'1,2,3,4',{poolJson,side,seed:r.seed});if(r.case==='turn1')w.apply([[0,'team 1, 4, 5'],[1,'team 1, 4, 5']]);}const result=await w.search(side,r.budget,r.seed,false);results.push({case:r.case,seed:r.seed,budget:r.budget,public_choice:r.choice,local_choice:result.best,public_policy:r.policy,local_policy:result.policy});}finally{w.terminate();}}
  return results;
 },reference);
 writeFileSync(dir+'public-local-parity.json',JSON.stringify(results,null,2)+'\n');console.log(results.map(r=>({case:r.case,public:r.public_choice,local:r.local_choice,exact:JSON.stringify(r.public_policy)===JSON.stringify(r.local_policy)})));
}finally{await browser?.close();await vite?.close();server.closeAllConnections();await new Promise(r=>server.close(r));}
