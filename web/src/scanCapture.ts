import {responseJson} from './photoApi';
export const SCAN_LIMITS={candidates:9,attempts:24,bytes:3*1024*1024,longEdge:2560,cadenceMs:3000,captureMs:90000,processingMs:120000};
export class CandidateBank{
 entries=new Map<string,{file:File;url:string}>();
 constructor(private urls=URL){}
 accept(id:string,file:File,retained:string[]){
  if(retained.length>SCAN_LIMITS.candidates||new Set(retained).size!==retained.length||file.size>SCAN_LIMITS.bytes)throw Error('Scan candidate bounds exceeded.');
  if(retained.some(key=>key!==id&&!this.entries.has(key)))throw Error('Candidate response belongs to another scan.');
  for(const [key,value] of this.entries)if(!retained.includes(key)){this.urls.revokeObjectURL(value.url);this.entries.delete(key);}
  if(retained.includes(id)&&!this.entries.has(id))this.entries.set(id,{file,url:this.urls.createObjectURL(file)});
 }
 keep(ids:string[]){for(const [key,value] of this.entries)if(!ids.includes(key)){this.urls.revokeObjectURL(value.url);this.entries.delete(key);}}
 clear(){this.keep([]);}
}
export async function scanRequest(path:string,init:RequestInit,signal:AbortSignal,timeout=30000){
 const controller=new AbortController(),abort=()=>controller.abort();
 if(signal.aborted)controller.abort();else signal.addEventListener('abort',abort,{once:true});
 const timer=setTimeout(abort,timeout);
 try{return await responseJson(await fetch('/api/photo/scan/'+path,{...init,signal:controller.signal}));}
 finally{clearTimeout(timer);signal.removeEventListener('abort',abort);}
}
export function scanJson(path:string,payload:object,signal:AbortSignal){return scanRequest(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)},signal);}
export async function captureScanFrame(video:HTMLVideoElement):Promise<File>{
 if(!video.videoWidth||!video.videoHeight)throw Error('Camera preview is not ready.');
 const factor=Math.min(1,SCAN_LIMITS.longEdge/Math.max(video.videoWidth,video.videoHeight));
 const canvas=document.createElement('canvas');canvas.width=Math.round(video.videoWidth*factor);canvas.height=Math.round(video.videoHeight*factor);
 try{canvas.getContext('2d')!.drawImage(video,0,0,canvas.width,canvas.height);const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('Frame capture failed.')),'image/jpeg',.95));
  if(blob.size>SCAN_LIMITS.bytes)throw Error('Frame exceeds 3 MiB. Use a lower camera resolution or single-photo mode.');
  return new File([blob],'scan-frame.jpg',{type:'image/jpeg'});
 }finally{canvas.width=0;canvas.height=0;}
}
export function stopMedia(stream:MediaStream|null){stream?.getTracks().forEach(t=>t.stop());}
