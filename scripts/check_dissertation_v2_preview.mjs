// CPU-only headless local-document check. Uses an isolated browser, never a user session.
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const root=path.resolve(import.meta.dirname,'..');
const dep=process.env.CODEX_DEPENDENCY_NODE_MODULES || 'C:/Users/leoni/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const require=createRequire(path.join(dep,'_resolver.cjs'));
const {chromium}=require('playwright');
const preview=path.join(root,'docs/dissertation/refactor_v2/preview');
const browser=await chromium.launch({executablePath:process.env.DISSERTATION_BROWSER || 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--disable-gpu','--disable-extensions','--disable-background-networking']});
try{
 const page=await browser.newPage({viewport:{width:1250,height:1000}});
 await page.goto(pathToFileURL(path.join(preview,'dissertation_reading_draft.html')).href,{waitUntil:'load'});
 const report=await page.evaluate(()=>({
   sections:document.querySelectorAll('section').length,
   math:document.querySelectorAll('math').length,
   brokenImages:[...document.images].filter(i=>!i.complete||i.naturalWidth===0).map(i=>i.src),
   unresolvedLocalAnchors:[...document.querySelectorAll('a[href^="#"]')].map(a=>a.getAttribute('href').slice(1)).filter(id=>!document.getElementById(decodeURIComponent(id))),
   documentOverflow:document.documentElement.scrollWidth>innerWidth,
   horizontalMathScroll:[...document.querySelectorAll('math')].filter(m=>m.scrollWidth>m.clientWidth+2).length,
   tableScroll:[...document.querySelectorAll('table')].filter(t=>t.scrollWidth>t.clientWidth+2).length
 }));
 for(const [name,prefix] of [['chapter1_memory','1.3.4'],['chapter2_results','2.2.6'],['chapter3_loss','3.3.1'],['chapter3_position','3.7.4'],['chapter4_pending','4.8.2'],['appendix_gates','Ж.v2.1']]){
   const h=page.locator('h2,h3,h4').filter({hasText:new RegExp('^'+prefix.replaceAll('.','\\.'))}).first();
   if(await h.count()){await h.evaluate(e=>window.scrollTo(0,e.getBoundingClientRect().top+window.scrollY-110));await page.screenshot({path:path.join(preview,name+'.png')});}
 }
 fs.writeFileSync(path.join(root,'docs/dissertation/refactor_v2/registries/browser_qa.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));
 if(report.brokenImages.length||report.unresolvedLocalAnchors.length||report.documentOverflow)process.exitCode=1;
}finally{await browser.close();}
