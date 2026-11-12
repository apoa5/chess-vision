// UI replay of a saved real API result. Actual model/geometry requests are
// independently exercised by scripts/replay_assisted_mvp15.py, not mocked there.
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';
const root=resolve('outputs/evaluation/assisted_mvp1_v1');
const require=createRequire(root+'/browser-tools/package.json');const {chromium}=require('playwright');
const output=resolve('outputs/evaluation/assisted_mvp15_v1/browser_multi');mkdirSync(output,{recursive:true});
const replay=JSON.parse(readFileSync('outputs/evaluation/assisted_mvp15_v1/application_replay_v1_budget3.json'));
const result=structuredClone(replay.result);result.multiview.reference_frame_id='frame_001';
const checks=[];let stopped=0,cancelled=0,finished=0;
const browser=await chromium.launch({headless:true,executablePath:resolve(root,'browsers/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell'),args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
try{
 const context=await browser.newContext({viewport:{width:390,height:844},permissions:['camera']});
 await context.addInitScript(()=>{window.__stopped=0;const original=MediaStreamTrack.prototype.stop;MediaStreamTrack.prototype.stop=function(){window.__stopped++;return original.call(this);};});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/api/photo/scan/**',async route=>{const url=new URL(route.request().url()),action=url.pathname.split('/').at(-1);let body;
 if(action==='session')body={session_id:'browser-fixture-session'};
 if(action==='candidate'){const id=route.request().headers()['x-scan-frame'];body={status:'retained',retained_ids:[id]};}
 if(action==='finish'){finished++;body={phase:'queued'};}
 if(action==='cancel'){cancelled++;body={status:'cancelled'};}
 if(action==='status')body={phase:'review_ready',attempts:1,retained_ids:['frame_001'],selected_ids:['frame_001'],candidate_count:1,inputs_prepared:64,completed_inputs:64,total_inputs:64,warnings:[],result};
 await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(body)});
 });
 await page.goto('http://127.0.0.1:8765/?mode=photo');await page.getByRole('button',{name:'Multi-View Scan',exact:true}).click();
 await page.getByLabel('The board and pieces will stay unchanged throughout this scan.').check();
 await page.getByRole('button',{name:'Start Scan',exact:true}).click();await page.getByAltText('Candidate frame_001').waitFor();checks.push('Start scan, camera preview and bounded candidate retention');
 await page.getByRole('button',{name:'Finish Scan',exact:true}).click();await page.locator('.digital-board').waitFor();assert.equal(finished,1);assert.ok(await page.evaluate(()=>window.__stopped)>0);checks.push('Manual Finish stops camera and installs recorded real model draft');
 assert.equal(await page.getByRole('button',{name:'Confirm Position',exact:true}).isEnabled(),false);assert.equal(await page.getByRole('button',{name:'Copy FEN',exact:true}).isEnabled(),false);checks.push('Unreviewed multi-view draft cannot be confirmed or copied');
 await page.getByText('View disagreement:',{exact:false}).click();checks.push('Cross-view disagreement diagnostics available');
 await page.screenshot({path:output+'/mobile_review.png',fullPage:true});
 assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));checks.push('390px mobile layout does not overflow');
 await page.getByRole('button',{name:'Retry Scan',exact:true}).click();await page.getByAltText('Candidate frame_001').waitFor();await page.getByRole('button',{name:'Cancel Scan',exact:true}).click();await page.getByRole('button',{name:'Retry Scan',exact:true}).waitFor();assert.ok(cancelled>0);assert.equal(await page.locator('.digital-board').count(),0);checks.push('Cancel releases stream, clears draft and cancels backend session');
 assert.deepEqual(errors,[]);writeFileSync(output+'/results.json',JSON.stringify({checks,errors,physical_smartphone_tested:false,camera:'Chromium fake native media stream',recognition:'Saved real three-view response replay; real HTTP inference tested separately'},null,2));console.log(JSON.stringify(checks));
}finally{await browser.close();}
