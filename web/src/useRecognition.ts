import {useEffect,useRef,useState} from 'react';
import {openRecognitionSession,recognitionError} from './recognitionSession';
import {responseIsCurrent} from './recognitionChecks';
import {homography,squarePolygon} from './geometry';
import type {Point} from './geometry';
import {imageBytes,sha256,sessionZip} from './recording';
export type AlignedFrame={image:HTMLCanvasElement;corners:Point[];polygons:Record<string,Point[]>|null;timestamp:number};
export type PieceObservation={session_id:string;frame_id:string;board_epoch:number;geometry_version:number;elapsed_ms:number;resolved_squares:number;completion_enabled:boolean;squares:Record<string,{symbol:string|null;class_name:string;confidence:number;probabilities:number[];logits:number[];visibility:string;context_valid_fraction:number}>};
export function useRecognition(){
 const [enabled,setEnabled]=useState(false),[result,setResult]=useState<PieceObservation|null>(null),[message,setMessage]=useState('Piece recognition is off.'),[accepted,setAccepted]=useState(0);
 const [mode,setMode]=useState<'idle'|'scanning'|'paused'|'aiming'|'capturing'|'review'|'confirmed'>('idle');
 const latest=useRef<{jpeg:Uint8Array;meta:object;observation:PieceObservation}|null>(null);
 const state=useRef({mode:'idle' as 'idle'|'scanning'|'paused'|'aiming'|'capturing'|'review'|'confirmed',enabled:false,session:'',epoch:0,version:0,start:0,generation:0,busy:false,last:-Infinity,controller:null as AbortController|null});
 const invalidate=()=>{const s=state.current;s.generation++;s.controller?.abort();s.controller=null;s.busy=false;if(s.mode==='capturing'){s.mode='aiming';setMode('aiming');setMessage('Capture cancelled. Adjust your view, then tap Recognize this view.');}};
 // An already copied keyframe remains paired with its own verified geometry even if live tracking is interrupted.
 const canCheckOrientation=()=>{const s=state.current;return !s.enabled&&!s.busy&&s.mode!=='review'&&s.mode!=='aiming'&&s.mode!=='capturing';};
 const trackingLost=(discardObservation=false)=>{if(discardObservation)invalidate();const s=state.current;if(s.enabled&&!s.busy)setMessage('Waiting for board tracking to recover before reading pieces.');};
 const stop=()=>{invalidate();latest.current=null;setResult(null);state.current.mode='idle';setMode('idle');state.current.enabled=false;state.current.session='';setEnabled(false);setMessage('Piece recognition stopped.');};
 useEffect(()=>()=>{state.current.enabled=false;state.current.generation++;state.current.controller?.abort();},[]);
 const start=async()=>{
 stop();const s=state.current,token=s.generation,controller=new AbortController();s.controller=controller;setMessage('Opening recognition session.');const timeout=window.setTimeout(()=>controller.abort(),10000);
 try{const session=await openRecognitionSession(controller.signal,()=>{if(s.generation===token)setMessage('Board analysis is busy. Retrying recognition session…');});if(s.generation!==token)return;s.session=session.session_id;s.enabled=true;s.mode='scanning';setMode('scanning');s.epoch=0;s.version=0;s.start=performance.now();s.last=-Infinity;setEnabled(true);setAccepted(0);setMessage('Waiting for a verified tracking frame. Keep pieces still.');}catch(error){if(s.generation===token)setMessage((error as Error).message);}finally{clearTimeout(timeout);if(s.generation===token)s.controller=null;}
 };
 const pause=()=>{if(state.current.mode!=='scanning')return;invalidate();state.current.enabled=false;state.current.mode='paused';setMode('paused');setEnabled(false);setMessage('Piece recognition paused. Camera tracking continues.');};
 const resume=async()=>{const s=state.current;if(s.mode==='review'||s.mode==='confirmed'||s.mode==='aiming'||s.mode==='capturing')return;if(!s.session){await start();return;}invalidate();s.enabled=true;s.mode='scanning';setMode('scanning');s.last=-Infinity;setEnabled(true);setMessage('Waiting for a verified tracking frame. Keep pieces still.');};
 const confirm=()=>{const s=state.current;if(s.mode!=='review'||!latest.current)return;invalidate();s.mode='confirmed';s.enabled=false;setMode('confirmed');setEnabled(false);setMessage('Position confirmed by you. Piece recognition is locked; camera tracking continues.');};
 const newPosition=()=>{invalidate();latest.current=null;setResult(null);const s=state.current;s.epoch++;s.last=-Infinity;s.enabled=!!s.session;s.mode=s.enabled?'scanning':'idle';setEnabled(s.enabled);setMode(s.mode);setAccepted(0);setMessage(s.enabled?'Waiting for a fresh view. Hold the board still and keep all squares visible.':'Position reset. Start scanning to continue.');};
 const retry=()=>{if(state.current.mode!=='review'&&state.current.mode!=='confirmed')return;newPosition();const s=state.current;s.enabled=false;s.mode='aiming';setEnabled(false);setMode('aiming');setMessage('Adjust your view, hold the board still, then tap Recognize this view.');};
 const captureView=async(frame:AlignedFrame|null)=>{if(state.current.mode!=='aiming')return;if(!frame||performance.now()-frame.timestamp>500){setMessage('Waiting for a fresh tracked view. Keep the full board visible, then tap Recognize this view.');return;}await offer(frame,true);};
 const offer=async(frame:AlignedFrame,manual=false)=>{
 const s=state.current;if((manual?s.mode!=='aiming':!s.enabled)||s.busy||frame.timestamp<s.start||(!manual&&performance.now()-s.last<2500))return;
 if(manual){s.mode='capturing';setMode('capturing');}
 s.busy=true;s.last=performance.now();const token=s.generation,epoch=s.epoch,session=s.session,version=++s.version,frameID=crypto.randomUUID();
 const copy=document.createElement('canvas');copy.width=frame.image.width;copy.height=frame.image.height;copy.getContext('2d')!.drawImage(frame.image,0,0);
 const h=homography(frame.corners),polygons=frame.polygons??Object.fromEntries(Array.from({length:64},(_,i)=>[String.fromCharCode(97+i%8)+(Math.floor(i/8)+1),squarePolygon(h,i%8,Math.floor(i/8))]));
 const meta={schema_version:'scan_piece_observation_v1',session_id:session,frame_id:frameID,board_epoch:epoch,geometry_version:version,orientation_version:epoch,width:copy.width,height:copy.height,timestamp_ms:frame.timestamp-s.start,geometry:{source:'tracking',version,corners:frame.corners,h_board_to_frame:h,square_polygons:polygons}};
 const controller=new AbortController();s.controller=controller;const timeout=window.setTimeout(()=>controller.abort(),60000);
 try{setMessage('Recognizing a selected frame…');const bytes=await imageBytes(copy);if(s.generation!==token)return;const hash=await sha256(bytes);if(s.generation!==token)return;const response=await fetch('/api/scan/keyframe',{method:'POST',headers:{'Content-Type':'image/jpeg','X-Scan-Observation':JSON.stringify({...meta,source_sha256:hash})},body:new Uint8Array(bytes).buffer,signal:controller.signal});if(!response.ok)throw Error(await recognitionError(response,'Piece recognition'));const observation=await response.json() as PieceObservation;
 if(s.generation!==token)return;if(!responseIsCurrent(observation,session,s.epoch,frameID,version,hash))throw Error(manual?'Piece recognition returned an incompatible response. Adjust your view and capture again.':'Piece recognition returned an incompatible response. Retrying with a fresh tracked frame.');
 latest.current={jpeg:new Uint8Array(bytes),meta:{...meta,source_sha256:hash},observation};s.enabled=false;s.mode='review';setEnabled(false);setMode('review');setResult(observation);setAccepted(n=>n+1);setMessage('Check the digital board. Recognition is paused until you confirm or retry.');
 }catch(error){if(s.generation===token)setMessage((error as Error).message);}finally{clearTimeout(timeout);copy.width=0;copy.height=0;if(s.generation===token){s.busy=false;s.controller=null;if(manual&&s.mode==='capturing'){s.mode='aiming';setMode('aiming');}}}
 };
 const exportLatest=()=>{const frame=latest.current;if(!frame)return;const blob=sessionZip([{name:'frame.jpg',data:frame.jpeg},{name:'keyframe.json',data:new TextEncoder().encode(JSON.stringify(frame.meta,null,2))},{name:'observation.json',data:new TextEncoder().encode(JSON.stringify(frame.observation,null,2))}]);const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='recognition-observation.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);};
 return {exportLatest,start,stop,pause,resume,confirm,retry,captureView,newPosition,offer,invalidate,trackingLost,canCheckOrientation,enabled,result,mode,message,accepted};
}
