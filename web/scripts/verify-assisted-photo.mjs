// Local opt-in MVP browser acceptance. One real C10 frozen regression request;
// replayed responses below exercise UI behavior, not additional model accuracy.
import {createRequire} from 'node:module';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {resolve} from 'node:path';
import assert from 'node:assert/strict';
const root=resolve('outputs/evaluation/assisted_mvp1_v1');
const require=createRequire(root+'/browser-tools/package.json');
const {chromium}=require('playwright');
const base=process.env.MVP1_URL||'http://127.0.0.1:8765';
const output=resolve(process.env.MVP1_BROWSER_OUTPUT||root+'/browser');mkdirSync(output,{recursive:true});
const manifest=JSON.parse(readFileSync('data/metadata/convnext_independent_c10_manifest_v1.json'));
const reference=JSON.parse(readFileSync('outputs/evaluation/convnext_c10_v1/convnext_result.json'));
const photo=manifest.photos[0],original=reference.rows.find(r=>r.position_id===photo.id);
const executable=resolve(root,'browsers/chromium_headless_shell-1254/chrome-headless-shell-linux64/chrome-headless-shell');
const browser=await chromium.launch({headless:true,executablePath:executable,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
const results={schema_version:'assisted_mvp1_browser_v1',physical_smartphone_tested:false,real_recognition_requests:0,checks:[],page_errors:[]};
function record(name){results.checks.push(name);console.log('PASS',name);}
let context;
try{
 context=await browser.newContext({viewport:{width:1280,height:960},permissions:['clipboard-read','clipboard-write','camera']});
 // Read a clone in the page, not the DevTools response cache (no-store bodies
 // can be evicted by Chromium). This does not replace any network response.
 await context.addInitScript(()=>{window.__mvpResponses=[];const original=window.fetch.bind(window);window.fetch=async(...args)=>{const response=await original(...args);if(String(args[0]).startsWith('/api/photo/'))void response.clone().json().then(body=>window.__mvpResponses.push({url:String(args[0]),status:response.status,body}));return response;};});
 const page=await context.newPage();page.on('pageerror',e=>results.page_errors.push(e.message));
 await page.goto(base+'/?mode=photo');
 await page.getByRole('heading',{name:'Chess Vision'}).waitFor();
 assert.equal(await page.locator('.digital-board').count(),0);record('No mocked initial board');
 await page.getByLabel('Upload photograph',{exact:true}).setInputFiles({name:'bad.heic',mimeType:'image/heic',buffer:Buffer.from('bad')});
 await page.getByRole('alert').filter({hasText:'HEIC'}).waitFor();record('Unsupported upload is rejected');
 await page.getByLabel('Upload photograph',{exact:true}).setInputFiles(resolve(photo.path));
 await page.getByAltText('Selected physical chessboard').waitFor();
 const started=performance.now(),pending=page.waitForResponse(r=>r.url().endsWith('/api/photo/recognize'));
 await page.getByRole('button',{name:'Recognize photograph',exact:true}).click();
 await page.getByRole('progressbar',{name:'Recognizing photograph'}).waitFor();
 const response=await pending;assert.equal(response.status(),200);await page.waitForFunction(()=>window.__mvpResponses.some(r=>r.url.endsWith('/recognize')&&r.status===200));const actual=await page.evaluate(()=>window.__mvpResponses.find(r=>r.url.endsWith('/recognize')&&r.status===200).body);results.real_recognition_requests++;
 writeFileSync(output+'/recognition_c10_position_01.json',JSON.stringify(actual,null,2)+'\n');
 await page.locator('.digital-board button').first().waitFor();results.browser_recognition_seconds=(performance.now()-started)/1000;
 assert.equal(await page.locator('.digital-board button').count(),64);
 let maximum=0;
 for(const o of original.observations){const p=actual.squares[o.square];for(let i=0;i<13;i++)maximum=Math.max(maximum,Math.abs(p.logits[i]-o.raw_logits[i]));}
 const classSymbols=[null,'P','N','B','R','Q','K','p','n','b','r','q','k'];
 const mismatches=original.observations.filter(o=>actual.squares[o.square].symbol!==classSymbols[o.prediction]).map(o=>o.square);
 assert.deepEqual(mismatches,[]);assert.ok(maximum<.000027);
 assert.deepEqual(actual.packed_inputs_sha256,photo.inputs.packed_sha256);
 results.parity={frame_id:photo.id,square_disagreements:mismatches,max_abs_logit_error:maximum,packed_pixels_identical:true};
 record('Real ConvNeXt request matches all frozen predictions and crop/mask hashes');
 await page.screenshot({path:output+'/desktop-draft.png',fullPage:true});
 assert.equal(actual.squares.b8.symbol,'b');
 await page.getByRole('button',{name:/^b8, Black bishop/}).click();
 await page.getByRole('button',{name:'Flip digital board'}).click();
 assert.match(await page.locator('.digital-board button').first().getAttribute('aria-label'),/^h1,/);
 await page.getByRole('button',{name:'Set b8 to Black knight',exact:true}).click();
 await page.getByRole('button',{name:/^b8, Black knight, edited/}).waitFor();record('Board flip preserves canonical square for editing');
 await page.getByRole('button',{name:'Undo piece edit'}).click();await page.getByRole('button',{name:/^b8, Black bishop/}).waitFor();
 await page.getByRole('button',{name:/^a3, Empty square/}).click();await page.getByRole('button',{name:'Set a3 to White queen',exact:true}).click();
 await page.getByRole('button',{name:/^a3, White queen, edited/}).waitFor();
 await page.getByRole('button',{name:'Reset to AI prediction'}).click();await page.getByRole('button',{name:/^a3, Empty square/}).waitFor();record('Add/undo/reset operate without another scan');
 await page.getByRole('button',{name:/^b8, Black bishop/}).click();await page.getByRole('button',{name:'Set b8 to Black knight',exact:true}).click();
 assert.equal(await page.getByTestId('placement-preview').textContent(),photo.fen.split(' ')[0]);
 assert.ok(await page.getByRole('button',{name:'Copy FEN',exact:true}).isDisabled());assert.ok(await page.getByRole('button',{name:'Confirm Position',exact:true}).isDisabled());
 for(const label of ['White kingside','White queenside','Black kingside','Black queenside'])await page.getByLabel(label+' castling',{exact:true}).check();
 for(const name of ['En-passant target','Halfmove clock','Fullmove number'])assert.equal(await page.getByLabel(name,{exact:true}).count(),0);
 await page.getByLabel('I have checked the board and game state.', {exact:false}).check();
 const confirmation=page.waitForResponse(r=>r.url().endsWith('/api/photo/confirm'));
 await page.getByRole('button',{name:'Confirm Position',exact:true}).click();assert.equal((await confirmation).status(),200);await page.getByLabel('Confirmed FEN',{exact:true}).waitFor();const receipt=await page.evaluate(()=>window.__mvpResponses.findLast(r=>r.url.endsWith('/confirm')).body);
 assert.equal(receipt.full_fen,photo.fen);assert.equal(receipt.user_confirmed,true);
 assert.equal(await page.getByLabel('Confirmed FEN',{exact:true}).inputValue(),photo.fen);
 await page.getByRole('button',{name:'Copy FEN',exact:true}).click();
 assert.equal(await page.evaluate(()=>navigator.clipboard.readText()),photo.fen);record('Explicit confirmation validates and copies the corrected full FEN');
 await page.screenshot({path:output+'/desktop-confirmed.png',fullPage:true});
 await page.getByLabel('Side to move',{exact:true}).selectOption('b');assert.equal(await page.getByLabel('Confirmed FEN',{exact:true}).count(),0);assert.ok(await page.getByRole('button',{name:'Copy FEN',exact:true}).isDisabled());record('Game-state edit invalidates confirmation');
 await page.getByLabel('Side to move',{exact:true}).selectOption('w');await page.getByLabel('I have checked the board and game state.',{exact:false}).check();
 let release,arrived;const atRequest=new Promise(r=>arrived=r),held=new Promise(r=>release=r);
 await page.route('**/api/photo/confirm',async route=>{arrived();await held;await route.continue();});
 await page.getByRole('button',{name:'Confirm Position',exact:true}).click();await atRequest;
 await page.getByLabel('Side to move',{exact:true}).selectOption('b');release();
 await page.getByRole('status').filter({hasText:'Position changed. Review and confirm again.'}).waitFor();assert.equal(await page.getByLabel('Confirmed FEN',{exact:true}).count(),0);
 await page.unroute('**/api/photo/confirm');record('Late confirmation cannot certify an edited revision');
 await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Permission denied','NotAllowedError');};});
 await page.getByRole('button',{name:'Open camera',exact:true}).click();await page.getByRole('alert').filter({hasText:'Permission denied'}).waitFor();record('Camera permission denial is actionable');
 // A real invalid-board photograph, not a mocked geometry success.
 const blank=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=640;c.height=640;c.getContext('2d').fillRect(0,0,640,640);return c.toDataURL('image/jpeg');});
 await page.getByLabel('Upload photograph',{exact:true}).setInputFiles({name:'blank.jpg',mimeType:'image/jpeg',buffer:Buffer.from(blank.split(',')[1],'base64')});
 const rejected=page.waitForResponse(r=>r.url().endsWith('/api/photo/recognize'));await page.getByRole('button',{name:'Recognize photograph',exact:true}).click();
 const failure=await rejected;assert.equal(failure.status(),422);await page.waitForFunction(()=>window.__mvpResponses.some(r=>r.url.endsWith('/recognize')&&r.status===422));assert.equal(await page.evaluate(()=>window.__mvpResponses.findLast(r=>r.url.endsWith('/recognize')&&r.status===422).body.code),'localization');assert.equal(await page.locator('.digital-board').count(),0);record('Real localization failure is not presented as recognition success');
 await page.getByLabel('Upload photograph',{exact:true}).setInputFiles(resolve(photo.path));
 let releaseDraft,intercepted;const hit=new Promise(r=>intercepted=r),hold=new Promise(r=>releaseDraft=r);
 await page.route('**/api/photo/recognize',async route=>{intercepted();await hold;try{await route.fulfill({json:actual});}catch{}});
 await page.getByRole('button',{name:'Recognize photograph',exact:true}).click();await hit;
 await page.getByRole('button',{name:'Cancel recognition',exact:true}).click();releaseDraft();await page.waitForTimeout(200);assert.equal(await page.locator('.digital-board').count(),0);record('Cancelled recognition cannot install a late draft');
 await context.close();context=null;
 // Narrow viewport/touch emulation is not a physical smartphone test.
 context=await browser.newContext({viewport:{width:390,height:844},isMobile:true,hasTouch:true,deviceScaleFactor:1,permissions:['camera']});
 const mobile=await context.newPage();mobile.on('pageerror',e=>results.page_errors.push(e.message));
 await mobile.route('**/api/photo/recognize',route=>route.fulfill({json:actual}));
 await mobile.goto(base+'/?mode=photo');await mobile.getByLabel('Upload photograph',{exact:true}).setInputFiles(resolve(photo.path));
 await mobile.getByRole('button',{name:'Recognize photograph',exact:true}).click();await mobile.locator('.digital-board button').first().waitFor();
 assert.equal(await mobile.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await mobile.getByRole('button',{name:/^b8, Black bishop/}).click();await mobile.getByRole('button',{name:'Set b8 to Black knight',exact:true}).click();
 await mobile.screenshot({path:output+'/mobile-review.png',fullPage:true});record('390px touch viewport supports canonical editing without horizontal overflow');
 await mobile.addInitScript(()=>{const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async options=>{const s=await original(options);window.__mvpCamera=s;return s;};});
 await mobile.reload();await mobile.getByRole('button',{name:'Open camera',exact:true}).click();
 await mobile.waitForFunction(()=>document.querySelector('video')?.videoWidth>0);
 await mobile.getByRole('button',{name:'Capture photograph',exact:true}).click();await mobile.getByAltText('Selected physical chessboard').waitFor();
 assert.equal(await mobile.evaluate(()=>window.__mvpCamera.getTracks().every(t=>t.readyState==='ended')),true);record('Browser fake-camera capture releases every media track');
 assert.deepEqual(results.page_errors,[]);results.success=true;
}catch(error){results.success=false;results.error=String(error.stack||error);throw error;}
finally{if(context)await context.close();await browser.close();writeFileSync(output+'/results_v1.json',JSON.stringify(results,null,2)+'\n');}
