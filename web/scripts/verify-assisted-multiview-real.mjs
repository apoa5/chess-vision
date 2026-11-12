// Actual geometry + ConvNeXt API requests. Camera is a canvas replay of one
// approved development recording frame, not a physical smartphone.
import {createRequire} from 'node:module';import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';import {resolve} from 'node:path';import assert from 'node:assert/strict';
const root=resolve('outputs/evaluation/assisted_mvp1_v1'),require=createRequire(root+'/browser-tools/package.json'),{chromium}=require('playwright');
const source=JSON.parse(readFileSync('outputs/evaluation/assisted_mvp15_v1/replay_source_v1.json'));
const replay=JSON.parse(readFileSync('outputs/evaluation/assisted_mvp15_v1/application_replay_v1_budget3.json'));
const id=replay.candidates.find(c=>c.status==='retained').frame_id,frame=source.frames.find(f=>f.frame_id===id);
const image='data:image/png;base64,'+readFileSync(frame.path).toString('base64');
const output=resolve('outputs/evaluation/assisted_mvp15_v1/browser_actual');mkdirSync(output,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:resolve(root,'browsers/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell')});
try{
 const context=await browser.newContext({viewport:{width:1280,height:960}});
 await context.addInitScript(image=>{
  navigator.mediaDevices.getUserMedia=async()=>{const img=new Image();img.src=image;await img.decode();const c=document.createElement('canvas');c.width=img.naturalWidth;c.height=img.naturalHeight;c.getContext('2d').drawImage(img,0,0);return c.captureStream(10);};
  window.__responses=[];const original=fetch.bind(window);window.fetch=async(...args)=>{const r=await original(...args);if(String(args[0]).startsWith('/api/photo/scan/status'))void r.clone().json().then(body=>window.__responses.push(body));return r;};
 },image);
 const page=await context.newPage();page.on('pageerror',e=>console.log('PAGE ERROR',e.message));await page.goto('http://127.0.0.1:8765/?mode=photo');await page.getByRole('button',{name:'Multi-View Scan',exact:true}).click();await page.getByLabel('The board and pieces will stay unchanged throughout this scan.').check();await page.getByRole('button',{name:'Start Scan',exact:true}).click();await page.getByAltText('Candidate frame_001').waitFor({timeout:60000});
 await page.getByRole('button',{name:'Finish Scan',exact:true}).click();await page.locator('.digital-board').waitFor({timeout:120000});
 const response=await page.evaluate(()=>window.__responses.findLast(r=>r.result)?.result);assert.ok(response?.multiview.accepted_view_count===1);assert.equal(response.multiview.selected_strategy,'single_photo_fallback');
 assert.equal(await page.getByRole('button',{name:'Copy FEN',exact:true}).isEnabled(),false);
 await page.getByLabel('I checked every piece, empty square, grid orientation and all game-state fields against my physical board.').check();await page.getByRole('button',{name:'Confirm Position',exact:true}).click();await page.getByLabel('Confirmed FEN',{exact:true}).waitFor();
 const fen=await page.getByLabel('Confirmed FEN',{exact:true}).inputValue();assert.equal(fen,'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1');
 await page.screenshot({path:output+'/confirmed.png',fullPage:true});writeFileSync(output+'/result.json',JSON.stringify({source_frame_id:id,source_frame_path:frame.path,physical_smartphone_tested:false,camera:'Approved recording frame through canvas MediaStream',actual_backend_inference:true,confirmed_fen:fen,result:response},null,2));console.log('PASS actual browser Start → capture → Finish → geometry → ConvNeXt → review → confirm FEN');
}finally{await browser.close();}
