import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
const require=createRequire(import.meta.url),ts=require('typescript');
require.extensions['.ts']=(module,filename)=>module._compile(ts.transpileModule(readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,filename);
const {CandidateBank,SCAN_LIMITS,captureScanFrame,stopMedia,scanRequest,scanJson}=require('../src/scanCapture.ts');
test('candidate bank retains only server-owned bounded compressed frames and releases URLs',()=>{
 const revoked=[];let counter=0;const urls={createObjectURL:()=>`blob:${++counter}`,revokeObjectURL:url=>revoked.push(url)};const bank=new CandidateBank(urls);
 for(let i=0;i<9;i++)bank.accept(String(i),new File(['frame'],'frame.jpg'),Array.from({length:i+1},(_,j)=>String(j)));
 assert.equal(bank.entries.size,9);
 assert.throws(()=>bank.accept('overflow',new File(['x'],'x.jpg'),[...bank.entries.keys(),'overflow']));
 assert.throws(()=>bank.accept('x',{size:SCAN_LIMITS.bytes+1},['x']));
 assert.throws(()=>bank.accept('x',new File(['x'],'x.jpg'),['not-this-session']));
 bank.keep(['1','4','7']);assert.equal(bank.entries.size,3);assert.equal(revoked.length,6);bank.clear();assert.equal(revoked.length,9);assert.equal(bank.entries.size,0);
});
test('pruned/duplicate candidates do not consume a new preview URL',()=>{
 let created=0;const bank=new CandidateBank({createObjectURL:()=>String(++created),revokeObjectURL:()=>{}});
 bank.accept('a',new File(['a'],'a.jpg'),['a']);bank.accept('b',new File(['b'],'b.jpg'),['a']);assert.equal(created,1);assert.equal(bank.entries.size,1);
});
test('camera frames are bounded in resolution/bytes and temporary canvas is freed',async()=>{
 const previous=globalThis.document;let dimensions;const canvas={width:0,height:0,getContext:()=>({drawImage:()=>{dimensions=[canvas.width,canvas.height];}}),toBlob:cb=>cb(new Blob(['jpeg'],{type:'image/jpeg'}))};globalThis.document={createElement:()=>canvas};
 try{const frame=await captureScanFrame({videoWidth:4000,videoHeight:3000});assert.deepEqual(dimensions,[2560,1920]);assert.equal(frame.type,'image/jpeg');assert.equal(canvas.width,0);assert.equal(canvas.height,0);
 canvas.toBlob=cb=>cb(new Blob([new Uint8Array(SCAN_LIMITS.bytes+1)]));await assert.rejects(captureScanFrame({videoWidth:100,videoHeight:100}),/3 MiB/);assert.equal(canvas.width,0);
 await assert.rejects(captureScanFrame({videoWidth:0,videoHeight:0}),/not ready/);
 }finally{globalThis.document=previous;}
});
test('all camera tracks stop on finish/cancel and null cleanup is safe',()=>{
 let stopped=0;stopMedia({getTracks:()=>[{stop:()=>stopped++},{stop:()=>stopped++}]});stopMedia(null);assert.equal(stopped,2);
});
test('session metadata and raw candidate request preserve identity and signal',async()=>{
 const old=globalThis.fetch;const calls=[];globalThis.fetch=async(url,init)=>{calls.push([url,init]);return new Response(JSON.stringify({session_id:'s'}));};
 try{const controller=new AbortController();await scanJson('session',{unchanged_position:true},controller.signal);assert.equal(calls[0][0],'/api/photo/scan/session');assert.deepEqual(JSON.parse(calls[0][1].body),{unchanged_position:true});
 const file=new File(['jpeg'],'frame.jpg');await scanRequest('candidate',{method:'POST',body:file,headers:{'X-Scan-Frame':'f1'}},controller.signal);assert.equal(calls[1][1].body,file);assert.ok(calls[1][1].signal instanceof AbortSignal);
 }finally{globalThis.fetch=old;}
});
test('cancel and timeout abort requests rather than installing late results',async()=>{
 const old=globalThis.fetch;globalThis.fetch=async(url,options)=>new Promise((resolve,reject)=>options.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true}));
 try{const controller=new AbortController(),pending=scanRequest('status?session_id=s',{},controller.signal,1000);controller.abort();await assert.rejects(pending,/Aborted/);
 await assert.rejects(scanRequest('status?session_id=s',{},new AbortController().signal,5),/Aborted/);
 }finally{globalThis.fetch=old;}
});
test('aggregated probabilities may explicitly omit raw logits; raw photo responses may not',async()=>{
 const {validatePhotoResult,PIECES,CLASS_NAMES,SQUARES,placement}=require('../src/photoReview.ts');
 const board=Array(64).fill('');board[4]='k';board[60]='K';
 const squares=Object.fromEntries(SQUARES.map((s,i)=>{const index=PIECES.indexOf(board[i]);return [s,{symbol:board[i]||null,class_name:CLASS_NAMES[index],confidence:1,logits:null,probabilities:Array.from({length:13},(_,j)=>j===index?1:0)}];}));
 const value={schema_version:'assisted_photo_result_v1',recognition_id:'r',status:'draft',confirmed:false,squares,provisional_placement:placement(board),geometry:{status:'completed_unverified'}};
 assert.throws(()=>validatePhotoResult(value));assert.equal(validatePhotoResult({...value,aggregation:{method:'mean_probabilities'}}).squares.a8.logits,null);
});
