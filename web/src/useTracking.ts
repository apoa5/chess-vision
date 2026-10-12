import {useEffect,useRef,useState} from 'react';
import {verifiedMarkingGeometry,markingDisagrees} from './trackingChecks';
import type {Point} from './geometry';
import type {Gray,TrackResult} from './tracking';
import type {AlignedFrame} from './useRecognition';
import {imageBytes} from './recording';
const MAX_EDGE=640;
function gray(c:HTMLCanvasElement):Gray {const {data}=c.getContext('2d')!.getImageData(0,0,c.width,c.height),pixels=new Uint8Array(c.width*c.height);for(let i=0;i<pixels.length;i++)pixels[i]=(data[i*4]*77+data[i*4+1]*150+data[i*4+2]*29)>>8;return {width:c.width,height:c.height,pixels};}
export function useTracking(video:React.RefObject<HTMLVideoElement|null>,canvas:React.RefObject<HTMLCanvasElement|null>,callbacks:{onFrame:(frame:AlignedFrame)=>void;onInvalid:(discardObservation?:boolean)=>void;canCheckOrientation?:()=>boolean}){
 const handlers=useRef(callbacks);handlers.current=callbacks;
 const [status,setStatus]=useState('uninitialized'),[reason,setReason]=useState(''),[geometry,setGeometry]=useState<{corners:Point[];polygons:Record<string,Point[]>|null}|null>(null),[latency,setLatency]=useState(0);
 const verifiedFrame=useRef<AlignedFrame|null>(null);
 const active=useRef<{worker:Worker;dispose:()=>void}|null>(null);
 const end=()=>{verifiedFrame.current=null;handlers.current.onInvalid();if(active.current){active.current.dispose();active.current=null;}setGeometry(null);setStatus('uninitialized');};
 useEffect(()=>()=>{active.current?.dispose();},[]);
 const begin=(corners:Point[],polygons:Record<string,Point[]>|null,automatic=false)=>{
 end();const v=video.current,c=canvas.current;if(!v||!c||!v.videoWidth)throw Error('Camera is no longer available. Restart the camera to continue.');
 if(automatic){c.width=v.videoWidth;c.height=v.videoHeight;}
 const width=c.width,height=c.height,scale=Math.min(1,MAX_EDGE/Math.max(width,height)),small=document.createElement('canvas');small.width=Math.round(width*scale);small.height=Math.round(height*scale);const sx=small.width/width,sy=small.height/height;
 const snapshot=document.createElement('canvas');snapshot.width=width;snapshot.height=height;
 const worker=new Worker(new URL('./tracking.worker.ts',import.meta.url),{type:'module'});
 let busy=true,ready=false,recovering=false,id=0,sent=0,lastResult=performance.now(),lastCheck=performance.now(),nextRecovery=0,pending=false,referencePolys=polygons;
 let serverBusy=false,controller:AbortController|null=null,displayCorners:Point[]=corners;
 const alive=()=>active.current?.worker===worker;
 const initialize=(image:HTMLCanvasElement,pts:Point[])=>{small.getContext('2d')!.drawImage(image,0,0,small.width,small.height);const frame=gray(small);ready=false;busy=true;lastResult=performance.now();worker.postMessage({kind:'initialize',frame,corners:pts.map(p=>[p[0]*sx,p[1]*sy]),timestamp:performance.now(),id:++id},[frame.pixels.buffer]);};
 const fail=(text:string)=>{if(!alive())return;verifiedFrame.current=null;handlers.current.onInvalid();active.current!.dispose();active.current=null;setGeometry(null);setStatus('lost');setReason(text+' Freeze a fresh frame to realign.');};
 const recover=(text:string)=>{if(!alive())return;verifiedFrame.current=null;handlers.current.onInvalid(text.includes('disagreed'));recovering=true;id++;ready=false;busy=false;pending=false;setGeometry(null);setStatus('recovering');setReason(text+' Hold the entire board in view with readable markings.');};
 // Only image-based semantic marking evidence may automatically restore orientation.
 const check=async(recovery:boolean)=>{
 if(serverBusy||!alive())return;let retryDelay=1000;serverBusy=true;controller=new AbortController();const requestController=controller,timeout=window.setTimeout(()=>requestController.abort(),60000);
 const image=document.createElement('canvas');image.width=width;image.height=height;
 const expected=displayCorners.map(p=>[...p] as Point);
 if(recovery)image.getContext('2d')!.drawImage(v,0,0);else image.getContext('2d')!.drawImage(c,0,0);
 try{
 const bytes=await imageBytes(image);if(!alive())return;
 const response=await fetch('/api/bootstrap',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:new Uint8Array(bytes).buffer,signal:requestController.signal});
 if(!response.ok){retryDelay=3000;throw Error(`Board check returned ${response.status}`);}const result=await response.json();if(!alive())return;
 const verified=verifiedMarkingGeometry(result,width,height);if(!verified){if(recovery)setReason(`Board not yet verified: ${result.reason??'markings or square boundaries are unclear'}. Keep the whole board in view.`);return;}
 const pts=verified.corners;
 if(recovery){
 // The server result belongs to the saved request image, not the current live video.
 referencePolys=verified.polygons;displayCorners=pts;recovering=false;c.getContext('2d')!.drawImage(image,0,0);initialize(image,pts);setStatus('initializing');setReason('Markings found. Verifying the current camera view.');
 }else if(!recovering){if(markingDisagrees(expected,pts))recover('Marking check disagreed with tracking.');}
 }catch(error){retryDelay=3000;if(alive()&&recovery)setReason(`Reacquiring: ${(error as Error).message}. Keep markings visible.`);}finally{clearTimeout(timeout);image.width=0;image.height=0;if(alive()){serverBusy=false;controller=null;nextRecovery=performance.now()+retryDelay;}}
 };
 worker.onerror=()=>fail('Tracking worker failed.');
 worker.onmessage=event=>{
 if(!alive())return;const data=event.data;if(data.id!==id)return;
 if(data.kind==='error'){recover(data.reason);return;}if(data.kind==='ready'){busy=false;ready=true;lastResult=performance.now();return;}
 busy=false;lastResult=performance.now();const result=data.result as TrackResult;
 if(result.state==='lost'){recover(result.reason);return;}
 if(result.state==='degraded'){verifiedFrame.current=null;handlers.current.onInvalid();setGeometry(null);setStatus('degraded');setReason(result.reason);pending=false;return;}
 if(lastResult-sent>500){recover('Tracking could not keep up with the camera.');return;}
 if(!pending||!result.transform||!result.corners)return;
 c.getContext('2d')!.drawImage(snapshot,0,0);
 const transform=(p:Point):Point=>{const [x,y]=p;const m=result.transform!,d=m[6]*x*sx+m[7]*y*sy+m[8];return [(m[0]*x*sx+m[1]*y*sy+m[2])/d/sx,(m[3]*x*sx+m[4]*y*sy+m[5])/d/sy];};
 displayCorners=result.corners.map(p=>[p[0]/sx,p[1]/sy]);
 const aligned={corners:displayCorners,polygons:referencePolys?Object.fromEntries(Object.entries(referencePolys).map(([key,poly])=>[key,poly.map(transform)])):null};setGeometry(aligned);verifiedFrame.current={image:c,...aligned,timestamp:sent};handlers.current.onFrame(verifiedFrame.current);setStatus('tracking');setReason(`${result.inliers} board-plane features; move slowly and keep borders visible.`);setLatency(Math.round(data.processing_ms));pending=false;
 if(performance.now()-lastCheck>30000&&!serverBusy&&(handlers.current.canCheckOrientation?.()??true)){lastCheck=performance.now();void check(false);}
 };
 const timer=window.setInterval(()=>{
 if(!alive())return;
 if(v.videoWidth!==width||v.videoHeight!==height){fail('Camera dimensions changed.');return;}
 if(v.readyState<2)return;
 if(recovering){if(performance.now()>nextRecovery)void check(true);return;}
 if(!ready||busy)return;
 snapshot.getContext('2d')!.drawImage(v,0,0);small.getContext('2d')!.drawImage(snapshot,0,0,small.width,small.height);const frame=gray(small);pending=true;busy=true;sent=performance.now();worker.postMessage({kind:'frame',frame,timestamp:sent,id:++id},[frame.pixels.buffer]);
 },80);
 const watchdog=window.setInterval(()=>{if(!recovering&&performance.now()-lastResult>700)recover('No fresh verified tracking frame.');},100);
 active.current={worker,dispose:()=>{worker.terminate();worker.onmessage=null;worker.onerror=null;controller?.abort();clearInterval(timer);clearInterval(watchdog);snapshot.width=0;snapshot.height=0;small.width=0;small.height=0;pending=false;}};
 if(automatic){recover('Finding board coordinates and orientation.');}else{setStatus('initializing');setReason('Checking the current camera view against your reviewed frame.');initialize(c,corners);}
 };
 const captureVerifiedFrame=()=>{const frame=verifiedFrame.current;if(!active.current||!frame||performance.now()-frame.timestamp>500)return null;return frame;};
 return {begin,end,status,reason,geometry,latency,captureVerifiedFrame};
}
