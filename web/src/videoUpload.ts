/** Bounded local decoding. The original File is never read into an ArrayBuffer. */
import {SCAN_LIMITS,captureScanFrame} from './scanCapture';
export const VIDEO_LIMITS={bytes:1024*1024*1024,durationSeconds:1800,metadataMs:15000,seekMs:15000,minSpacingSeconds:3,decodedPixels:32000000};
export type VideoInfo={filename:string;size_bytes:number;duration_ms:number;width:number;height:number;mime:string};
export function videoType(file:File){
 const type=file.type||(/\.mp4$/i.test(file.name)?'video/mp4':/\.webm$/i.test(file.name)?'video/webm':/\.mov$/i.test(file.name)?'video/quicktime':'');
 if(!['video/mp4','video/webm','video/quicktime'].includes(type))throw Error('Choose MP4, WebM or MOV. Codec support also depends on your browser.');
 if(!file.size)throw Error('The video file is empty.');
 if(file.size>VIDEO_LIMITS.bytes)throw Error('Choose a local video no larger than 1 GiB. The original video will not be uploaded.');
 return type;
}
export function sampleTimes(duration:number):number[]{
 if(!Number.isFinite(duration)||duration<=0||duration>VIDEO_LIMITS.durationSeconds)throw Error('Video duration must be finite, greater than zero and at most 30 minutes.');
 const count=Math.min(SCAN_LIMITS.attempts,Math.max(1,Math.floor(duration/VIDEO_LIMITS.minSpacingSeconds)));
 // Midpoints of equal timeline bins avoid fragile exact end-of-stream seeks.
 return Array.from({length:count},(_,i)=>duration*(i+.5)/count);
}
function aborted(){return new DOMException('Video processing cancelled.','AbortError');}
export function mediaEvent(video:HTMLVideoElement,event:string,signal:AbortSignal,timeout:number,action:()=>void){
 return new Promise<void>((resolve,reject)=>{
  let timer:ReturnType<typeof setTimeout>;
  const cleanup=()=>{clearTimeout(timer);video.removeEventListener(event,done);video.removeEventListener('error',error);signal.removeEventListener('abort',abort);};
  const done=()=>{cleanup();resolve();},error=()=>{cleanup();reject(Error('This video could not be decoded. Try an MP4 with H.264 or a browser-supported WebM.'));},abort=()=>{cleanup();reject(aborted());};
  if(signal.aborted){reject(aborted());return;}
  video.addEventListener(event,done,{once:true});video.addEventListener('error',error,{once:true});signal.addEventListener('abort',abort,{once:true});
  timer=setTimeout(()=>{cleanup();reject(Error('Video decoding/seeking timed out. Try a shorter, seekable MP4 or WebM.'));},timeout);
  try{action();}catch(e){cleanup();reject(e);}
 });
}
export function releaseVideo(video:HTMLVideoElement|null){if(video){video.pause();video.removeAttribute('src');video.load();}}
export async function loadVideo(file:File,url:string,signal:AbortSignal){
 const mime=videoType(file),video=document.createElement('video');video.preload='metadata';video.muted=true;video.playsInline=true;
 try{
  // MOV MIME probing can be empty even for decodable MP4-compatible MOV bytes.
  // Actual metadata and seek decoding remain authoritative.
  if(mime!=='video/quicktime'&&!video.canPlayType(mime))throw Error('Your browser does not support this video container. Try MP4/H.264 or WebM.');
  await mediaEvent(video,'loadedmetadata',signal,VIDEO_LIMITS.metadataMs,()=>{video.src=url;video.load();});
  sampleTimes(video.duration);
  if(!video.videoWidth||!video.videoHeight||video.videoWidth*video.videoHeight>VIDEO_LIMITS.decodedPixels)throw Error('Video dimensions are missing or exceed the 32-megapixel decode limit.');
  return {video,info:{filename:file.name,size_bytes:file.size,duration_ms:video.duration*1000,width:video.videoWidth,height:video.videoHeight,mime} as VideoInfo};
 }catch(e){releaseVideo(video);throw e;}
}
export async function extractVideoFrame(video:HTMLVideoElement,seconds:number,signal:AbortSignal){
 if(signal.aborted)throw aborted();
 if(!Number.isFinite(seconds)||seconds<0||seconds>=video.duration)throw Error('Invalid source-video timestamp.');
 // seeking+seeked gates decoded current data; do not capture immediately after
 // assigning currentTime. No playback or camera permission is required.
 if(Math.abs(video.currentTime-seconds)>1e-7||video.readyState<2){
  await mediaEvent(video,'seeked',signal,VIDEO_LIMITS.seekMs,()=>{video.currentTime=seconds;});
 }
 if(signal.aborted)throw aborted();
 if(video.seeking||video.readyState<2||Math.abs(video.currentTime-seconds)>.1)throw Error('The browser did not decode the requested seek position reliably.');
 const timestamp_ms=video.currentTime*1000,file=await captureScanFrame(video);
 if(signal.aborted)throw aborted();
 return {file,timestamp_ms,requested_timestamp_ms:seconds*1000};
}
