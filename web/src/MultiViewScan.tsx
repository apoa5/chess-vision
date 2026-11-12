import React,{useEffect,useRef,useState} from 'react';
import {loadVideo,extractVideoFrame,releaseVideo,sampleTimes,videoType,type VideoInfo} from './videoUpload';
import {openPhotoCamera} from './photoApi';
import {validatePhotoResult,type PhotoResult} from './photoReview';
import {CandidateBank,SCAN_LIMITS,captureScanFrame,scanJson,scanRequest,stopMedia} from './scanCapture';
type ScanStatus={phase:string;attempts:number;retained_ids:string[];selected_ids:string[];candidate_count:number;inputs_prepared:number;completed_inputs:number;total_inputs:number;warnings:string[];error?:{message:string};result?:PhotoResult&{multiview:{reference_frame_id:string;accepted_view_count:number;fusion_policy:string;warnings:string[]}}};
export function MultiViewScan({onStart,onResult,onMessage,inputMode='live'}:{inputMode?:'live'|'video';onStart:()=>void;onResult:(result:PhotoResult,file:File)=>void;onMessage:(message:string)=>void}){
 const [phase,setPhase]=useState('idle'),[unchanged,setUnchanged]=useState(false),[error,setError]=useState(''),[note,setNote]=useState('Keep the pieces still.');
 const [attempts,setAttempts]=useState(0),[previews,setPreviews]=useState<{id:string;url:string}[]>([]),[status,setStatus]=useState<ScanStatus|null>(null);
 const [videoFile,setVideoFile]=useState<File|null>(null),[videoInfo,setVideoInfo]=useState<VideoInfo|null>(null),[videoUrl,setVideoUrl]=useState('');
 const [sampleCount,setSampleCount]=useState(0),[decisions,setDecisions]=useState<string[]>([]);
 const uploadedPreview=useRef<HTMLVideoElement>(null);
 const video=useRef<HTMLVideoElement>(null),bank=useRef(new CandidateBank()),props=useRef({onStart,onResult,onMessage});props.current={onStart,onResult,onMessage};
 const state=useRef({generation:0,active:false,finishing:false,sid:'',controller:null as AbortController|null,stream:null as MediaStream|null,pending:null as Promise<void>|null,started:0,decoder:null as HTMLVideoElement|null,sourceUrl:'',metrics:{metadata_ms:0,extraction_ms:0,candidate_request_ms:0,quality_check_ms:0,analysis_ms:0}});
 const preview=()=>setPreviews([...bank.current.entries].map(([id,v])=>({id,url:v.url})));
 const stop=()=>{stopMedia(state.current.stream);state.current.stream=null;if(video.current)video.current.srcObject=null;};
 const sendCancel=(sid:string)=>{if(sid)void fetch('/api/photo/scan/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid}),keepalive:true}).catch(()=>{});};
 const cancel=(notify=true)=>{const s=state.current;s.generation++;s.active=false;s.finishing=false;s.controller?.abort();s.controller=null;sendCancel(s.sid);s.sid='';stop();releaseVideo(s.decoder);s.decoder=null;if(s.sourceUrl)URL.revokeObjectURL(s.sourceUrl);s.sourceUrl='';bank.current.clear();if(notify){setVideoFile(null);setVideoInfo(null);setVideoUrl('');preview();setPhase('cancelled');setNote('Scan cancelled.');props.current.onStart();}};
 useEffect(()=>{const hide=()=>{if(document.hidden&&state.current.stream)cancel();};document.addEventListener('visibilitychange',hide);return()=>{cancel(false);document.removeEventListener('visibilitychange',hide);};},[]);
 const updateNote=(text:string)=>{setNote(text);props.current.onMessage(text);};
 const collect=async(token:number)=>{
  const s=state.current;let count=0;
  while(s.active&&token===s.generation&&count<SCAN_LIMITS.attempts&&performance.now()-s.started<SCAN_LIMITS.captureMs){
   count++;setAttempts(count);
   const attempt=(async()=>{
    try{
     if(!video.current)throw Error('Camera preview is unavailable.');
     const timestamp=performance.now()-s.started,file=await captureScanFrame(video.current);
     if(token!==s.generation||!s.active)return;
     const id='frame_'+String(count).padStart(3,'0');
     const result=await scanRequest('candidate',{method:'POST',headers:{'Content-Type':'image/jpeg','X-Scan-Session':s.sid,'X-Scan-Frame':id,'X-Scan-Timestamp':String(timestamp)},body:file},s.controller!.signal,45000);
     if(token!==s.generation)return;
     bank.current.accept(id,file,result.retained_ids);preview();
     if(!s.finishing)updateNote(result.status==='retained'?`${bank.current.entries.size} views collected. Try another angle.`:result.message||'Similar view skipped. Try another angle.');
    }catch(e){if(token===s.generation){setError((e as Error).message);updateNote('Frame skipped. Keep the board visible and try another angle.');}}
   })();s.pending=attempt;await attempt;if(s.pending===attempt)s.pending=null;
   if(!s.active||token!==s.generation)break;
   await new Promise(resolve=>setTimeout(resolve,SCAN_LIMITS.cadenceMs));
  }
  if(s.active&&token===s.generation){s.active=false;stop();setPhase('paused_limit');updateNote('Scan paused. Tap Finish Scan or retry.');}
 };
 const chooseVideo=async(file:File)=>{
  cancel(false);props.current.onStart();setVideoFile(null);setVideoInfo(null);setVideoUrl('');preview();setError('');setUnchanged(false);setDecisions([]);setStatus(null);setPhase('reading_metadata');
  const s=state.current,token=s.generation;s.controller=new AbortController();
  try{videoType(file);s.sourceUrl=URL.createObjectURL(file);const started=performance.now();const loaded=await loadVideo(file,s.sourceUrl,s.controller.signal);
   if(token!==s.generation){releaseVideo(loaded.video);return;}s.decoder=loaded.video;s.metrics={metadata_ms:performance.now()-started,extraction_ms:0,candidate_request_ms:0,quality_check_ms:0,analysis_ms:0};setVideoFile(file);setVideoInfo(loaded.info);setVideoUrl(s.sourceUrl);setPhase('video_selected');updateNote('Video ready.');
  }catch(e){if(token===s.generation){releaseVideo(s.decoder);s.decoder=null;if(s.sourceUrl)URL.revokeObjectURL(s.sourceUrl);s.sourceUrl='';setPhase('failed');setError((e as Error).message);}}
 };
 const startVideo=async()=>{
  if(!videoFile||!videoInfo||!state.current.decoder)return;
  uploadedPreview.current?.pause();props.current.onStart();const s=state.current,token=s.generation;s.controller?.abort();s.controller=new AbortController();const signal=s.controller.signal;setError('');setAttempts(0);setDecisions([]);setStatus(null);bank.current.clear();preview();s.started=performance.now();s.active=true;setPhase('extracting');const runController=s.controller,deadlineTimer=setTimeout(()=>runController.abort(),8*60*1000);
  try{
   const {mime,...source}=videoInfo;void mime;
   const session=await scanJson('session',{unchanged_position:true,source:{kind:'uploaded_video',...source}},signal);
   if(token!==s.generation){sendCancel(session.session_id);return;}s.sid=session.session_id;
   const times=sampleTimes(videoInfo.duration_ms/1000);setSampleCount(times.length);let previousTimestamp=-1,previousPixels:Uint8Array|null=null;
   for(let i=0;i<times.length;i++){
    if(token!==s.generation||signal.aborted)return;setPhase('extracting');updateNote(`Reading frame ${i+1} of ${times.length}…`);
    const begun=performance.now(),frame=await extractVideoFrame(s.decoder!,times[i],signal);s.metrics.extraction_ms+=performance.now()-begun;
    if(token!==s.generation||signal.aborted)return;
    const bytes=new Uint8Array(await frame.file.arrayBuffer()),duplicate=previousPixels?.length===bytes.length&&bytes.every((v,i)=>v===previousPixels![i]);
    if(frame.timestamp_ms<=previousTimestamp||duplicate){setDecisions(v=>[...v,`${times[i].toFixed(2)} s: repeated decoded frame skipped`]);continue;}previousTimestamp=frame.timestamp_ms;previousPixels=bytes;
    const id='frame_'+String(i+1).padStart(3,'0');setAttempts(i+1);setPhase('checking_quality');updateNote(`Checking frame ${i+1} of ${times.length}…`);
    const before=performance.now(),response=await scanRequest('candidate',{method:'POST',headers:{'Content-Type':'image/jpeg','X-Scan-Session':s.sid,'X-Scan-Frame':id,'X-Scan-Timestamp':String(frame.timestamp_ms)},body:frame.file},signal,45000);s.metrics.candidate_request_ms+=performance.now()-before;s.metrics.quality_check_ms+=(response.quality_check_seconds||0)*1000;
    if(token!==s.generation)return;bank.current.accept(id,frame.file,response.retained_ids);preview();setDecisions(v=>[...v,`${(frame.timestamp_ms/1000).toFixed(2)} s: ${response.status}${response.reason?' — '+response.reason:''}`]);
   }
   if(token!==s.generation)return;s.active=false;releaseVideo(s.decoder);s.decoder=null;await finish();
  }catch(e){if(token===s.generation){s.active=false;releaseVideo(s.decoder);s.decoder=null;sendCancel(s.sid);setPhase('failed');setError(signal.aborted?'Video analysis stopped at its bounded processing deadline. Choose a shorter recording or retry.':(e as Error).message);}}finally{clearTimeout(deadlineTimer);}
 };
 const start=async()=>{
  cancel(false);props.current.onStart();const s=state.current,token=s.generation;s.controller=new AbortController();setError('');setStatus(null);setAttempts(0);preview();setPhase('opening');
  try{
   const session=await scanJson('session',{unchanged_position:true},s.controller.signal);if(token!==s.generation){sendCancel(session.session_id);return;}s.sid=session.session_id;
   const stream=await openPhotoCamera();if(token!==s.generation||document.hidden){stopMedia(stream);sendCancel(s.sid);return;}s.stream=stream;
   setPhase('scanning');await new Promise(resolve=>setTimeout(resolve,0));
   if(!video.current)throw Error('Camera preview could not open.');video.current.srcObject=stream;await video.current.play();
   if(token!==s.generation){stopMedia(stream);return;}
   s.active=true;s.started=performance.now();updateNote('Move slowly and keep the whole board visible.');void collect(token);
  }catch(e){if(token===s.generation){stop();sendCancel(s.sid);s.sid='';setPhase('failed');setError((e as Error).message);}}
 };
 const finish=async()=>{
  const s=state.current;if(!s.sid||s.finishing)return;s.finishing=true;s.active=false;stop();const token=s.generation;setPhase('finishing');setError('');updateNote('Finishing scan…');
  try{
   await s.pending;if(token!==s.generation)return;
   await scanJson('finish',{session_id:s.sid},s.controller!.signal);setPhase('selecting');const deadline=performance.now()+SCAN_LIMITS.processingMs;
   while(token===s.generation){
    if(performance.now()>deadline)throw Error('Recognition timed out. Cancel/retry the scan or use single-photo mode.');
    const value=await scanRequest('status?session_id='+encodeURIComponent(s.sid),{},s.controller!.signal,15000) as ScanStatus;
    if(token!==s.generation)return;setStatus(value);setPhase(value.phase);
    if(value.selected_ids.length){bank.current.keep(value.selected_ids);preview();}
    const text=value.phase==='queued'?'Waiting to process…':value.phase==='selecting'?'Selecting views…':value.phase==='processing_inputs'?`Preparing ${value.selected_ids.length} views…`:value.phase==='recognizing'?`Reading board: ${value.completed_inputs}/${value.total_inputs}`:value.phase==='combining'?'Preparing your board…':'Waiting for review.';
    updateNote(text);
    if(value.phase==='failed')throw Error(value.error?.message||'Recognition failed. Retry the scan.');
    if(value.phase==='cancelled')throw Error('Scan cancelled.');
    if(value.phase==='review_ready'&&value.result){
     const result=validatePhotoResult(value.result),reference=bank.current.entries.get(value.result.multiview.reference_frame_id);
     if(!reference)throw Error('Selected source frame is missing; retry the scan.');
     if(inputMode==='video'){s.metrics.analysis_ms=performance.now()-s.started;(result as PhotoResult&{video_upload:unknown}).video_upload={source:videoInfo,timings:{...s.metrics},sampling_policy:'timeline_bin_midpoints_max24_min3s_v1'};}
     props.current.onResult(result,reference.file);updateNote('Your board is ready to review.');return;
    }
    await new Promise(resolve=>setTimeout(resolve,500));
   }
  }catch(e){if(token===s.generation){sendCancel(s.sid);setPhase('failed');setError((e as Error).message);}}
  finally{if(token===s.generation)s.finishing=false;}
 };
 const busy=['reading_metadata','extracting','checking_quality','opening','scanning','finishing','queued','selecting','processing_inputs','recognizing','combining'].includes(phase);
 return <section className="photo-panel multi-scan" aria-labelledby="multi-title"><h2 id="multi-title">{inputMode==='video'?'1. Video':'1. Live scan'}</h2>
  {inputMode==='video'&&<><p>MP4, WebM or MOV · up to 1 GB / 30 minutes</p><label className="photo-upload">Choose video<input aria-label="Upload video" type="file" accept="video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov,.MOV" onChange={e=>{const f=e.target.files?.[0];if(f)void chooseVideo(f);e.target.value='';}}/></label>{videoInfo&&<><p>{videoInfo.filename} · {(videoInfo.size_bytes/1048576).toFixed(1)} MiB · {(videoInfo.duration_ms/1000).toFixed(2)} seconds · {videoInfo.width}×{videoInfo.height}</p>{videoUrl&&<video ref={uploadedPreview} src={videoUrl} controls={!busy} playsInline preload="metadata" aria-label="Uploaded video preview"/>}<button onClick={()=>cancel()}>Remove video</button></>}</>}
  {inputMode==='live'&&<p>Move slowly and keep the whole board visible.</p>}
  <label className="photo-attestation"><input type="checkbox" disabled={busy} checked={unchanged} onChange={e=>setUnchanged(e.target.checked)}/>{inputMode==='video'?'The chess pieces remain in the same positions throughout this video.':'The pieces will stay in place during this scan.'}</label>
  {inputMode==='live'?<div className="photo-actions"><button className="primary-button" disabled={!unchanged||busy} onClick={()=>void start()}>{phase==='idle'?'Start Scan':'Retry Scan'}</button><button disabled={!['scanning','paused_limit'].includes(phase)||attempts===0} onClick={()=>void finish()}>Finish Scan</button><button disabled={!busy&&phase!=='paused_limit'} onClick={()=>cancel()}>Cancel Scan</button></div>:<div className="photo-actions"><button className="primary-button" disabled={!unchanged||!videoFile||!state.current.decoder||busy} onClick={()=>void startVideo()}>Analyze video</button><button disabled={!busy} onClick={()=>cancel()}>Cancel video analysis</button></div>}
  {['opening','scanning'].includes(phase)&&<video ref={video} autoPlay muted playsInline aria-label="Multi-view camera preview"/>}
  <p role="status">{note}</p>
  {busy&&<p>{inputMode==='video'?`${attempts} / ${sampleCount} frames checked`:`${previews.length} views collected`}</p>}
  {status?.phase==='recognizing'&&<><progress aria-label="Square inputs processed" max={status.total_inputs||1} value={status.completed_inputs}/><p>{status.completed_inputs}/{status.total_inputs} inputs processed</p></>}
  {['finishing','queued','selecting','processing_inputs','combining'].includes(phase)&&<progress aria-label="Preparing scan review"/>}
  {error&&<p role="alert" className="photo-error">{error} Try again or use Upload Photo.</p>}

  {decisions.length>0&&<details><summary>Frame details</summary><ul>{decisions.map((d,i)=><li key={i}>{d}</li>)}</ul></details>}
  <div className="scan-contact">{previews.map(p=><figure key={p.id}><img src={p.url} alt={'Candidate '+p.id}/><figcaption>{p.id}</figcaption></figure>)}</div>
  {status?.result?.multiview.warnings.length? <details><summary>Scan details</summary>{status.result.multiview.warnings.map((warning,i)=><p key={i}>{warning}</p>)}</details>:null}

 </section>;
}
