import React,{useEffect,useRef,useState} from 'react';
import {homography,Point} from './geometry';
import {CapturedFrame,CaptureEvent,SESSION_VERSION,MAX_FRAMES,MAX_BYTES,imageBytes,sha256,sessionZip} from './recording';

type Props={video:React.RefObject<HTMLVideoElement|null>;canvas:React.RefObject<HTMLCanvasElement|null>;phase:string;corners:Point[];source:'camera'|'demo';frozenAt:number;refinedPolygons:Record<string,Point[]>|null;onBoardChange:()=>void};
export function Recorder({video,canvas,phase,corners,source,frozenAt,refinedPolygons,onBoardChange}:Props){
 const [position,setPosition]=useState(''),[fen,setFen]=useState(''),[lighting,setLighting]=useState(''),[view,setView]=useState('overhead'),[count,setCount]=useState(0),[running,setRunning]=useState(false),[exists,setExists]=useState(false),[status,setStatus]=useState('No scan recorded yet.');
 const session=useRef<{id:string;created:string;start:number;metadata:object;frames:CapturedFrame[];files:{name:string;data:Uint8Array}[];events:CaptureEvent[];epoch:number;bytes:number}|null>(null);
 const active=useRef(false),busy=useRef(false),version=useRef(0),viewRef=useRef(view),sourceRef=useRef(source);viewRef.current=view;sourceRef.current=source;
 const event=(type:string)=>{const s=session.current;if(s&&s.events.length<1000)s.events.push({type,timestamp_ms:performance.now()-s.start,board_epoch:s.epoch});};
 const pause=(reason='capture_paused')=>{active.current=false;setRunning(false);event(reason);};
 useEffect(()=>{if(phase!=='live'&&active.current)pause('camera_not_live');},[phase]);
 useEffect(()=>{const hidden=()=>{if(document.hidden&&active.current)pause('page_hidden');};document.addEventListener('visibilitychange',hidden);return()=>{active.current=false;document.removeEventListener('visibilitychange',hidden);};},[]);
 const capture=async(frozen=false)=>{
   const s=session.current;if(!s||busy.current)return;
   if(s.frames.length>=MAX_FRAMES){pause('frame_limit');setStatus('60-frame limit reached. Export this session.');return;}
   busy.current=true;const timestamp=(frozen?frozenAt:performance.now())-s.start,epoch=s.epoch,group=viewRef.current;
   try{
     let image:HTMLCanvasElement;let geometry:CapturedFrame['geometry']=null;
     if(frozen){if(timestamp<0)throw Error('This image predates the session. Capture a fresh frame first.');if(s.events.some(e=>e.type==='new_board_epoch'&&e.timestamp_ms>timestamp))throw Error('The board changed after this frame. Capture a fresh frame.');if(!canvas.current)throw Error('No frozen frame');image=canvas.current;geometry={h_board_to_frame:homography(corners),corners:corners.map(p=>[...p]),source:'manual',version:++version.current,...(refinedPolygons?{square_polygons:refinedPolygons}:{})};}
     else{const v=video.current;if(!v||!v.videoWidth||v.readyState<2)return;image=document.createElement('canvas');image.width=v.videoWidth;image.height=v.videoHeight;image.getContext('2d')!.drawImage(v,0,0);}
     const width=image.width,height=image.height,data=await imageBytes(image),hash=await sha256(data);
     if(session.current!==s)return;
     if(s.bytes+data.length>MAX_BYTES){pause('memory_limit');setStatus('Memory limit reached. Export this session.');return;}
     const id=`frame_${String(s.frames.length+1).padStart(4,'0')}`,path=`frames/${id}.jpg`;
     s.frames.push({frame_id:id,path,timestamp_ms:timestamp,width,height,source_sha256:hash,board_epoch:epoch,view_group:group,geometry,visibility:'unknown',source:frozen?sourceRef.current:'camera'});
     s.files.push({name:path,data});s.bytes+=data.length;setCount(s.frames.length);setStatus(`${s.frames.length} frames saved in browser memory (${(s.bytes/1048576).toFixed(1)} MB). Export before closing this page.`);
   }catch(e){pause('capture_error');setStatus((e as Error).message);}finally{busy.current=false;}
 };
 const captureRef=useRef(capture);captureRef.current=capture;
 useEffect(()=>{if(!running)return;const v=video.current;if(!v)return;let cancelled=false,last=-Infinity,id=0;
   const next=()=>{if(cancelled)return;if('requestVideoFrameCallback' in v)id=v.requestVideoFrameCallback(tick);else id=requestAnimationFrame(tick);};
   const tick=(now:number)=>{if(cancelled||!active.current)return;if(now-last>=1000){last=now;void captureRef.current();}next();};next();
   return()=>{cancelled=true;if('cancelVideoFrameCallback' in v)v.cancelVideoFrameCallback(id);else cancelAnimationFrame(id);};
 },[running]);
 const create=()=>{if(busy.current)return;if(!position.trim()){setStatus('Enter a position ID first.');return;}
   pause();session.current={id:`scan_${crypto.randomUUID()}`,created:new Date().toISOString(),start:performance.now(),metadata:{position_id:position.trim(),fen:fen.trim()||null,lighting:lighting.trim(),board_id:'wooden_tournament_39cm',equipment:'wooden tournament set',browser:navigator.userAgent,category:'assisted',ground_truth_source:fen.trim()?'user':'not_supplied',split:'development'},frames:[],files:[],events:[],epoch:0,bytes:0};setCount(0);setExists(true);setStatus('Session ready. Start the camera and record, or save an aligned frozen frame.');event('session_created');};
 const resume=()=>{if(phase!=='live'){setStatus('Start camera before recording.');return;}active.current=true;setRunning(true);event('capture_started');};
 const exportScan=async()=>{const s=session.current;if(!s||!s.frames.length||busy.current)return;pause('export');
   const manifest={schema_version:SESSION_VERSION,session_id:s.id,created_at:s.created,metadata:s.metadata,board:{square_cm:4,border_cm:3.5},capture:{format:'jpeg_sequence',interval_ms:1000,max_frames:MAX_FRAMES,automatic_tracking:false,automatic_recognition:false},frames:s.frames,events:s.events};
   const blob=sessionZip([{name:'manifest.json',data:new TextEncoder().encode(JSON.stringify(manifest,null,2))},...s.files]);
   const file=new File([blob],`${s.id}.zip`,{type:'application/zip'});
   try{if(navigator.canShare?.({files:[file]})){await navigator.share({files:[file],title:'Chess scan'});setStatus('Export shared. Save the ZIP before closing.');return;}
     const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=file.name;a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);setStatus('ZIP download requested. Check your Downloads folder.');
   }catch(e){setStatus(`Export not completed: ${(e as Error).message}. Your frames remain in memory.`);}
 };
 return <section><h2>Capture a scan</h2><p>Keep pieces stationary. Record about 15–30 seconds, moving slowly from overhead to an oblique view, then the opposite side. Frames remain local until you export them.</p>
 <label>Position ID <input value={position} disabled={exists} onChange={e=>setPosition(e.target.value)} placeholder="position_001"/></label>
 <label>Known FEN (optional) <input value={fen} disabled={exists} onChange={e=>setFen(e.target.value)} placeholder="Paste the ground-truth FEN"/></label>
 <label>Lighting/location <input value={lighting} disabled={exists} onChange={e=>setLighting(e.target.value)} placeholder="Living room, overhead lights"/></label>
 <button disabled={exists} onClick={create}>Create session</button>
 <label>Current view <select value={view} onChange={e=>{setView(e.target.value);event(`view:${e.target.value}`);}}>{['overhead','white-side','black-side','left-side','right-side','other'].map(v=><option key={v}>{v}</option>)}</select></label>
 <div className="buttons"><button disabled={!exists||running||phase!=='live'} onClick={resume}>Record scan</button><button disabled={!running} onClick={()=>pause()}>Pause recording</button><button disabled={!exists||phase!=='frozen'||corners.length!==4} onClick={()=>void capture(true)}>Save aligned frame</button><button disabled={!count} onClick={()=>void exportScan()}>Export ZIP</button></div>
 <button disabled={!exists} onClick={()=>event('hand_occlusion')}>Mark hand/occlusion</button><button disabled={!exists} onClick={()=>event('blur_or_board_loss')}>Mark blur/board loss</button><button disabled={!exists} onClick={()=>{pause('board_change');onBoardChange();if(session.current){session.current.epoch++;event('new_board_epoch');}setStatus('New board epoch. Old observations will not be combined with this position.');}}>Mark board/piece change</button>
 <button disabled={!exists||running} onClick={()=>{if(confirm('Discard frames in memory? Export first if you need them.')){session.current=null;setExists(false);setCount(0);setStatus('Session discarded.');}}}>Discard / new session</button>
 <p role="status">{status}</p><p>Continuous captures have no alignment yet. “Save aligned frame” stores corners only for that frozen image. View labels are user notes, not proof of independent evidence. No completion score is calculated.</p></section>;
}
