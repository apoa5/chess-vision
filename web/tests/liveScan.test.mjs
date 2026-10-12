import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
const require=createRequire(import.meta.url),ts=require('typescript'),React=require('react');
const original={useRef:React.useRef,useState:React.useState,useEffect:React.useEffect};
require.extensions['.ts']=(module,filename)=>module._compile(ts.transpileModule(readFileSync(filename,'utf8').replaceAll('import.meta.url',JSON.stringify(`file://${filename}`)),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,filename);
test('live acquisition needs no frozen corners and verifies a new frame before emitting geometry',async()=>{
 const globals={window:globalThis.window,document:globalThis.document,Worker:globalThis.Worker,fetch:globalThis.fetch,clearInterval:globalThis.clearInterval,performance:globalThis.performance};
 const intervals=new Map(),workers=[],emitted=[],states=[];let next=0,resolveRequest,now=1000,allowCheck=false,requests=0;
 class Canvas {width=640;height=480;label='blank';getContext(){return {drawImage:source=>{this.label=source.label;},getImageData:()=>({data:new Uint8ClampedArray(this.width*this.height*4)})};}toBlob(callback){callback(new Blob([this.label]));}}
 class Worker {messages=[];constructor(){workers.push(this);}postMessage(message){this.messages.push(message);}terminate(){this.terminated=true;}}
 React.useRef=value=>({current:value});React.useState=value=>[value,next=>states.push(next)];React.useEffect=()=>{};
 globalThis.window={setInterval:callback=>{const id=++next;intervals.set(id,callback);return id;},setTimeout,clearTimeout};globalThis.clearInterval=id=>intervals.delete(id);
 globalThis.document={createElement:()=>new Canvas()};globalThis.Worker=Worker;
 globalThis.performance={now:()=>now};
 globalThis.fetch=async()=>{requests++;return new Promise(resolve=>{resolveRequest=resolve;});};
 const {useTracking}=require('../src/useTracking.ts'),{homography,squarePolygon}=require('../src/geometry.ts');
 const video={videoWidth:640,videoHeight:480,readyState:2,label:'request-frame'},canvas=new Canvas();
 const tracking=useTracking({current:video},{current:canvas},{onFrame:f=>emitted.push(f),onInvalid:()=>{},canCheckOrientation:()=>allowCheck});
 try{
 tracking.begin([],null,true);const worker=workers[0];assert.equal(worker.messages.length,0);assert.ok(states.includes('recovering'));
 intervals.values().next().value();await new Promise(setImmediate);
 const corners=[[100,400],[420,400],[420,80],[100,80]],h=homography(corners),polygons=Object.fromEntries(Array.from({length:64},(_,i)=>[String.fromCharCode(97+i%8)+(1+Math.floor(i/8)),squarePolygon(h,i%8,Math.floor(i/8))]));
 video.label='camera-moved';resolveRequest({ok:true,json:async()=>({status:'candidate',orientation_source:'learned_border_markings',learned_marking_evidence:{inliers:16,distinct_files:8,distinct_ranks:8},geometry:{corners},square_polygons:polygons})});await new Promise(setImmediate);
 assert.equal(canvas.label,'request-frame');assert.equal(worker.messages[0].kind,'initialize');assert.equal(emitted.length,0);assert.equal(tracking.captureVerifiedFrame(),null);
 worker.onmessage({data:{kind:'ready',id:worker.messages[0].id}});intervals.values().next().value();const message=worker.messages.at(-1);assert.equal(message.kind,'frame');
 worker.onmessage({data:{kind:'result',id:message.id,processing_ms:1,result:{state:'tracking',corners,transform:[1,0,0,0,1,0,0,0,1],inliers:20}}});
 assert.equal(emitted.length,1);assert.equal(canvas.label,'camera-moved');assert.equal(Object.keys(emitted[0].polygons).length,64);assert.equal(tracking.captureVerifiedFrame(),emitted[0]);
 // A long-lived tracked board must not submit background OCR while a
 // piece scan/review owns the interaction. Resume checks when it is allowed.
 now=32001;
 const accepted=()=>{intervals.values().next().value();const frame=worker.messages.at(-1);worker.onmessage({data:{kind:'result',id:frame.id,processing_ms:1,result:{state:'tracking',corners,transform:[1,0,0,0,1,0,0,0,1],inliers:20}}});};
 accepted();await new Promise(setImmediate);assert.equal(requests,1,'background check must yield to piece recognition/review');
 allowCheck=true;accepted();await new Promise(setImmediate);assert.equal(requests,2,'orientation checks resume after review/recognition');
 resolveRequest({ok:true,json:async()=>({status:'uncertain'})});await new Promise(setImmediate);
 tracking.end();assert.equal(worker.terminated,true);assert.equal(intervals.size,0);assert.equal(tracking.captureVerifiedFrame(),null);
 }finally{tracking.end();Object.assign(React,original);for(const [key,value] of Object.entries(globals)){if(value===undefined)delete globalThis[key];else globalThis[key]=value;}}
});
