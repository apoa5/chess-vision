import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
const require=createRequire(import.meta.url),ts=require('typescript'),React=require('react');
require.extensions['.ts']=(module,filename)=>module._compile(ts.transpileModule(readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,filename);
test('retry waits for an explicit fresh capture, runs once, and preserves confirmation and stale-response guards',async()=>{
 const originalHooks={useRef:React.useRef,useState:React.useState,useEffect:React.useEffect},globals={window:globalThis.window,document:globalThis.document,fetch:globalThis.fetch};
 const slots=[];let cursor=0,requests=0,piece='N',delay=false,release,fail=false;
 React.useRef=value=>{const i=cursor++;return slots[i]??(slots[i]={current:value});};React.useState=value=>{const i=cursor++;if(!(i in slots))slots[i]=value;return [slots[i],next=>{slots[i]=typeof next==='function'?next(slots[i]):next;}];};React.useEffect=()=>{};
 globalThis.window={setTimeout,clearTimeout};globalThis.document={createElement:()=>({width:0,height:0,getContext:()=>({drawImage:()=>{}}),toBlob:cb=>cb(new Blob(['frame']))})};
 globalThis.fetch=async(url,options)=>{
 if(url.endsWith('/session'))return {ok:true,json:async()=>({session_id:'session'})};
 requests++;if(fail)return new Response('Image could not be classified',{status:422});const meta=JSON.parse(options.headers['X-Scan-Observation']);const squares=Object.fromEntries(Array.from({length:64},(_,i)=>[String.fromCharCode(97+i%8)+(Math.floor(i/8)+1),{symbol:piece,class_name:piece==='N'?'white_knight':'white_pawn',confidence:1,probabilities:Array.from({length:13},(_,j)=>j===(piece==='N'?2:1)?1:0),logits:Array(13).fill(0),visibility:'unknown',context_valid_fraction:1}]));
 if(delay)await new Promise(resolve=>{release=resolve;});
 return {ok:true,json:async()=>({...meta,squares,elapsed_ms:10,completion_enabled:false,resolved_squares:0})};
 };
 const {useRecognition}=require('../src/useRecognition.ts');const render=()=>{cursor=0;return useRecognition();};
 const frame=()=>({image:{width:640,height:480},corners:[[100,400],[420,400],[420,80],[100,80]],polygons:null,timestamp:performance.now()+1});
 try{
 let hook=render();await hook.start();hook=render();assert.equal(hook.mode,'scanning');assert.equal(hook.canCheckOrientation(),false);await hook.offer(frame());hook=render();assert.equal(hook.mode,'review');assert.equal(hook.canCheckOrientation(),false);assert.equal(hook.enabled,false);assert.equal(hook.result.squares.a1.symbol,'N');
 await hook.offer(frame());await hook.resume();assert.equal(requests,1);hook=render();assert.equal(hook.mode,'review');hook.invalidate();hook=render();assert.equal(hook.result.squares.a1.symbol,'N');
 hook.confirm();hook=render();assert.equal(hook.mode,'confirmed');assert.equal(hook.canCheckOrientation(),true);hook.invalidate();await hook.resume();await hook.offer(frame());assert.equal(requests,1);hook=render();assert.equal(hook.result.squares.a1.symbol,'N');assert.equal(hook.mode,'confirmed');
 piece='P';delay=true;hook.retry();hook=render();assert.equal(hook.result,null);assert.equal(hook.mode,'aiming');assert.equal(hook.enabled,false);assert.equal(hook.canCheckOrientation(),false);
 for(let i=0;i<10;i++)await hook.offer(frame());await hook.resume();assert.equal(requests,1,'retry and automatic frames must not classify until the capture button is pressed');
 await hook.captureView(null);await hook.captureView({...frame(),timestamp:performance.now()-1000});assert.equal(requests,1,'missing or stale tracked views cannot be captured');
 const capture=hook.captureView(frame());hook=render();assert.equal(hook.mode,'capturing');while(!release)await new Promise(setImmediate);await hook.offer(frame());await hook.captureView(frame());assert.equal(requests,2,'one button press submits only one frame');hook.trackingLost();release();await capture;hook=render();assert.equal(hook.mode,'review');assert.equal(hook.result.squares.a1.symbol,'P');assert.equal(hook.result.board_epoch,1);
 hook.retry();fail=true;await hook.captureView(frame());hook=render();assert.equal(hook.mode,'aiming');assert.equal(hook.enabled,false);assert.equal(hook.result,null);assert.ok(hook.message.includes('422'));for(let i=0;i<5;i++)await hook.offer(frame());assert.equal(requests,3,'a failed manual capture must wait for another button press');fail=false;
 delay=true;release=null;const pending=hook.captureView(frame());while(!release)await new Promise(setImmediate);hook.newPosition();release();await pending;hook=render();assert.equal(hook.mode,'scanning');assert.equal(hook.result,null,'late response after position reset is discarded');
 hook.stop();hook=render();assert.equal(hook.mode,'idle');assert.equal(hook.result,null);
 }finally{Object.assign(React,originalHooks);for(const [key,value] of Object.entries(globals)){if(value===undefined)delete globalThis[key];else globalThis[key]=value;}}
});
