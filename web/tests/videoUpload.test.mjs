import test from 'node:test';import assert from 'node:assert/strict';import {createRequire} from 'node:module';import {readFileSync} from 'node:fs';
const require=createRequire(import.meta.url),ts=require('typescript');require.extensions['.ts']=(m,f)=>m._compile(ts.transpileModule(readFileSync(f,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,f);
const {videoType,sampleTimes,mediaEvent,loadVideo,extractVideoFrame,releaseVideo,VIDEO_LIMITS}=require('../src/videoUpload.ts');
test('source file size is independent of frame upload size; type is not codec certification',()=>{
 assert.equal(videoType(new File([new Uint8Array(4*1024*1024)],'scan.mp4',{type:'video/mp4'})),'video/mp4');assert.equal(videoType({name:'scan.webm',type:'',size:123}),'video/webm');
 for(const f of [{name:'x.avi',type:'video/x-msvideo',size:1},{name:'x.mp4',type:'video/mp4',size:0},{name:'x.mp4',type:'video/mp4',size:VIDEO_LIMITS.bytes+1}])assert.throws(()=>videoType(f));
});
test('timeline sampling is deterministic, distributed, strictly increasing and capped',()=>{
 for(const duration of [.05,2,12,75,1800]){const a=sampleTimes(duration);assert.deepEqual(a,sampleTimes(duration));assert.ok(a.length<=24);assert.ok(a.every((v,i)=>v>0&&v<duration&&(!i||v>a[i-1])));assert.ok(Math.abs(a.reduce((p,c)=>p+c,0)/a.length-duration/2)<.001);}
 assert.equal(sampleTimes(75).length,24);assert.equal(sampleTimes(1800).length,24);assert.equal(sampleTimes(2).length,1);
 for(const d of [0,-1,Infinity,NaN,1801])assert.throws(()=>sampleTimes(d));
});
test('metadata event waits asynchronously, and abort/error/timeout clean listeners',async()=>{
 const v=new EventTarget(),signal=new AbortController();let done=false;const p=mediaEvent(v,'loadedmetadata',signal.signal,100,()=>{}).then(()=>done=true);await Promise.resolve();assert.equal(done,false);v.dispatchEvent(new Event('loadedmetadata'));await p;
 const a=new AbortController();const cancelled=mediaEvent(v,'seeked',a.signal,100,()=>{});a.abort();await assert.rejects(cancelled,{name:'AbortError'});
 await assert.rejects(mediaEvent(v,'seeked',signal.signal,5,()=>{}),/timed out/);
 await assert.rejects(mediaEvent(v,'seeked',signal.signal,100,()=>v.dispatchEvent(new Event('error'))),/could not be decoded/);
});
function fakeVideo(){const v=new EventTarget();Object.assign(v,{duration:75,videoWidth:1920,videoHeight:1080,readyState:2,seeking:false,canPlayType:()=> 'maybe',pause(){this.paused=true;},removeAttribute(){this.src='';},load(){}});let time=0;Object.defineProperty(v,'currentTime',{get:()=>time,set:x=>{time=x;v.seeking=true;setTimeout(()=>{v.seeking=false;v.dispatchEvent(new Event('seeked'));},5);}});return v;}
test('actual seek completion precedes capture and timestamps refer to source timeline',async()=>{
 const old=globalThis.document,v=fakeVideo();let capturedTime;globalThis.document={createElement:()=>({width:0,height:0,getContext:()=>({drawImage:()=>{assert.equal(v.seeking,false);capturedTime=v.currentTime;}}),toBlob:cb=>cb(new Blob(['jpeg'],{type:'image/jpeg'}))})};
 try{const f=await extractVideoFrame(v,40,new AbortController().signal);assert.equal(f.timestamp_ms,40000);assert.equal(f.requested_timestamp_ms,40000);assert.equal(capturedTime,40);assert.equal(f.file.type,'image/jpeg');}finally{globalThis.document=old;}
});
test('cancellation prevents capture of a late seek result',async()=>{
 const v=fakeVideo(),c=new AbortController(),p=extractVideoFrame(v,20,c.signal);c.abort();await assert.rejects(p,{name:'AbortError'});releaseVideo(v);assert.equal(v.src,'');assert.equal(v.paused,true);
});
test('container probe cannot bypass actual corrupt stream rejection',async()=>{
 const old=globalThis.document,v=fakeVideo();v.load=()=>{if(v.src)setTimeout(()=>v.dispatchEvent(new Event('error')),0);};globalThis.document={createElement:()=>v};
 try{await assert.rejects(loadVideo(new File(['bad'],'bad.mp4',{type:'video/mp4'}),'blob:bad',new AbortController().signal),/could not be decoded/);assert.equal(v.src,'');}finally{globalThis.document=old;}
});

test('MOV picker types are accepted but actual decoding is still required',()=>{assert.equal(videoType({name:'recording.MOV',type:'',size:123}),'video/quicktime');assert.equal(videoType({name:'recording.mov',type:'video/quicktime',size:123}),'video/quicktime');});
