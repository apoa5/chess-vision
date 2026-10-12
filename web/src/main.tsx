import React,{useEffect,useRef,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {homography,project,squarePolygon,contains,Point} from './geometry';
import './style.css';
import {BoardPreviewBoundary} from './BoardPreviewBoundary';
import {PositionReview} from './PositionReview';
import {DigitalBoard} from './DigitalBoard';
import {Recorder} from './Recorder';
import {imageBytes,sessionZip} from './recording';
const ALIGNMENT_VERSION='board_reader_multi_seed_v5';
import {useRecognition} from './useRecognition';
import {useTracking} from './useTracking';
function App(){
 const video=useRef<HTMLVideoElement>(null),canvas=useRef<HTMLCanvasElement>(null),stream=useRef<MediaStream|null>(null),generation=useRef(0);
 const recognition=useRecognition();
 const tracking=useTracking(video,canvas,{onFrame:frame=>{void recognition.offer(frame);},onInvalid:recognition.trackingLost,canCheckOrientation:recognition.canCheckOrientation});
 const [view,setView]=useState<'camera'|'board'>('camera');
 const actionMenu=useRef<HTMLDetailsElement>(null);
 const cameraSurface=useRef<HTMLDivElement>(null);
 const [surfaceSize,setSurfaceSize]=useState<Point>([320,480]);
 useEffect(()=>{const el=cameraSurface.current;if(!el)return;const resize=()=>{const r=el.getBoundingClientRect();setSurfaceSize([r.width,r.height]);};resize();if(typeof ResizeObserver==='undefined'){window.addEventListener('resize',resize);return()=>window.removeEventListener('resize',resize);}const observer=new ResizeObserver(resize);observer.observe(el);return()=>observer.disconnect();},[]);
 const [phase,setPhase]=useState('idle'),[message,setMessage]=useState('Start the camera, hold the board in view, scanning will find it automatically.'),[corners,setCorners]=useState<Point[]>([]),[selected,setSelected]=useState(''),[size,setSize]=useState<Point>([800,600]);
 const [frozenAt,setFrozenAt]=useState(0);
 const [source,setSource]=useState<'camera'|'demo'>('camera');
 const [autoPolys,setAutoPolys]=useState<Record<string,Point[]>|null>(null),[alignBusy,setAlignBusy]=useState(false);
 const [diagnostic,setDiagnostic]=useState<{jpeg:Uint8Array;report:object}|null>(null);
 const currentFrame=useRef(frozenAt);currentFrame.current=frozenAt;
 const analyze=async(assisted=false)=>{
 const c=canvas.current;if(!c||alignBusy)return;
 const frame=currentFrame.current,token=generation.current;setAlignBusy(true);setDiagnostic(null);setMessage('Analyzing board geometry. This may take several seconds.');
 let jpeg:Uint8Array|undefined,saved=false;const controller=new AbortController(),timeout=window.setTimeout(()=>controller.abort(),30000);
 const saveDiagnostic=(response:object)=>{if(jpeg&&frame===currentFrame.current&&token===generation.current){saved=true;setDiagnostic({jpeg,report:{schema_version:'alignment_diagnostic_v1',frontend_version:ALIGNMENT_VERSION,created_at:new Date().toISOString(),width:c.width,height:c.height,assisted,response}});}};
 try{
 jpeg=new Uint8Array(await imageBytes(c));
 const response=await fetch('/api/bootstrap',{method:'POST',headers:{'Content-Type':'image/jpeg',...(assisted?{'X-Board-Corners':JSON.stringify(corners)}:{})},body:new Uint8Array(jpeg).buffer,signal:controller.signal});
 if(!response.ok){const detail=(await response.text()).replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim().slice(0,250);saveDiagnostic({http_status:response.status,error:detail});throw Error(response.status===429?'Another board analysis is running. Wait a moment and retry.':`Board analysis failed (${response.status}): ${detail}`);}
 const result=await response.json();if(frame!==currentFrame.current||token!==generation.current)return;
 saveDiagnostic(result);
 if(result.implementation_version!==ALIGNMENT_VERSION){setMessage('The running Python server is using older recognition code. Stop it with Ctrl+C, restart it with the same command, then retry.');return;}
 if(result.status!=='candidate'){const attempts=(result.marking_attempts??[]).map((a:{path:string;reason?:string;status:string})=>`${a.path}: ${a.reason??a.status}`).join('; ');setMessage(`Alignment uncertain: ${result.reason}.${attempts?` Reader checks: ${attempts}.`:''} Download the alignment diagnostic to inspect this exact frame.`);return;}
 setCorners(result.geometry.corners);setAutoPolys(result.square_polygons);setMessage(`${assisted?'Assisted':'Automatic'} alignment candidate${result.orientation_source==='learned_border_markings'?` from ${result.learned_marking_evidence.inliers} consistent markings`:''}. Check a1, h8 and the square boundaries before saving. Piece predictions become available after tracking starts.`);
 }catch(e){if(frame===currentFrame.current&&token===generation.current){if(!saved)saveDiagnostic({error:(e as Error).message});setMessage((e as Error).message);}}finally{clearTimeout(timeout);setAlignBusy(false);}
 };
 const exportDiagnostic=()=>{if(!diagnostic)return;const blob=sessionZip([{name:'frame.jpg',data:diagnostic.jpeg},{name:'diagnostic.json',data:new TextEncoder().encode(JSON.stringify(diagnostic.report,null,2))}]);const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='alignment-diagnostic.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);};
 const stop=()=>{recognition.stop();tracking.end();generation.current++;stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;if(video.current)video.current.srcObject=null;};
 useEffect(()=>{const hide=()=>{if(document.hidden){stop();setPhase('idle');setCorners([]);setMessage('Camera stopped while this page was hidden. Restart to capture a fresh frame.');}};document.addEventListener('visibilitychange',hide);return()=>{stop();document.removeEventListener('visibilitychange',hide);};},[]);
 const start=async()=>{setView('camera');setDiagnostic(null);setSource('camera');stop();const token=generation.current;setAutoPolys(null);setCorners([]);setSelected('');setPhase('requesting');try{
 if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('Camera access requires HTTPS (or localhost on this device).');
 const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false});
 if(token!==generation.current){s.getTracks().forEach(t=>t.stop());return;}stream.current=s;
 s.getVideoTracks()[0].addEventListener('ended',()=>{stop();setPhase('idle');setMessage('Camera disconnected. Restart to continue.');},{once:true});
 if(video.current){video.current.srcObject=s;await video.current.play();}if(token!==generation.current)return;
 if(video.current?.videoWidth)setSize([video.current.videoWidth,video.current.videoHeight]);setPhase('live');setMessage('Opening scan session. Keep the whole board and its markings visible.');await recognition.start();if(token!==generation.current)return;tracking.begin([],null,true);setPhase('tracking');
 }catch(e){if(token===generation.current){stop();setPhase('idle');setMessage(e instanceof Error?e.message:'Unable to open camera');}}};
 const freeze=()=>{generation.current++;recognition.stop();setView('camera');setDiagnostic(null);tracking.end();setAutoPolys(null);setFrozenAt(performance.now());const v=video.current,c=canvas.current;if(!v||!c||!v.videoWidth)return;c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d')!.drawImage(v,0,0);setSize([c.width,c.height]);setCorners([]);setPhase('frozen');setMessage('Select the playing-area corners in the order below. Do not select the outside border.');};
 const demo=()=>{setView('camera');setDiagnostic(null);setAutoPolys(null);setFrozenAt(performance.now());setSource('demo');stop();setPhase('frozen');setSize([800,600]);setCorners([]);setSelected('');const c=canvas.current!;c.width=800;c.height=600;const ctx=c.getContext('2d')!;ctx.fillStyle='#654b38';ctx.fillRect(0,0,800,600);ctx.fillStyle='#bc9569';ctx.fillRect(115,15,570,570);for(let f=0;f<8;f++)for(let r=0;r<8;r++){ctx.fillStyle=(f+r)%2?'#eadac1':'#91613e';ctx.fillRect(160+f*60,60+(7-r)*60,60,60);ctx.fillStyle='#111';ctx.font='16px sans-serif';ctx.fillText(String.fromCharCode(97+f)+(r+1),166+f*60,80+(7-r)*60);}setMessage('Demo: click the four playing-area corners. The a1 corner is bottom left.');};
 const displayCorners=phase==='tracking'?(tracking.geometry?.corners??[]):corners;
 const displayPolys=phase==='tracking'?tracking.geometry?.polygons:autoPolys;
 let h:number[]|undefined,error='';if(displayCorners.length===4)try{h=homography(displayCorners);}catch(e){error=(e as Error).message;}
 const click=(e:React.MouseEvent<SVGSVGElement>)=>{if(alignBusy||phase==='tracking')return;const rect=e.currentTarget.getBoundingClientRect();const p:Point=[(e.clientX-rect.left)*size[0]/rect.width,(e.clientY-rect.top)*size[1]/rect.height];if(corners.length<4){setCorners([...corners,p]);return;}if(h)for(let f=0;f<8;f++)for(let r=0;r<8;r++)if(contains(autoPolys?.[String.fromCharCode(97+f)+(r+1)]??squarePolygon(h,f,r),p))setSelected(String.fromCharCode(97+f)+(r+1));};
 const boardObservation=recognition.result;
 const boardStatus=recognition.mode==='confirmed'?'Confirmed by you · locked':'Awaiting your confirmation';
 const supported=recognition.mode==='confirmed'&&boardObservation?Object.keys(boardObservation.squares):[];
 const recognitionStatus=recognition.message.startsWith('Latest frame classified')?'Collecting piece evidence. Move slowly to a different view; check the digital board.':recognition.message.startsWith('Recognizing a selected frame')?'Reading pieces…':recognition.message;
 const cameraActive=['live','frozen','tracking'].includes(phase);
 const trackingLabel=phase==='tracking'?(tracking.status==='tracking'?'Tracking':tracking.status==='recovering'?'Reacquiring':'Checking alignment'):phase==='frozen'?'Review alignment':phase==='live'?'Camera live':'Camera off';
 const retryRecognition=()=>{setView('camera');recognition.retry();};
 const stopCamera=()=>{stop();setPhase('idle');setCorners([]);setMessage('Camera stopped.');};
 return <main>
 <div className="scan-workspace">
 <header className="app-header"><div><span className="eyebrow">CHESS VISION</span><h1>Scan your board</h1></div><span className={`tracking-badge ${tracking.status==='tracking'&&phase==='tracking'?'is-tracking':''}`} role="status">{trackingLabel}</span></header>
 <nav className="view-switch" aria-label="Scan view"><button aria-pressed={view==='camera'} onClick={()=>setView('camera')}>Camera</button><button aria-pressed={view==='board'} disabled={phase==='frozen'} onClick={()=>setView('board')}>Digital board</button></nav>
 <div className={`scan-views view-${view} ${phase==='frozen'?'is-reviewing':''}`}>
 <div className="camera-pane">
 <div className="camera-heading"><span>Live view</span>{view==='board'&&<button className="quiet-button return-camera" onClick={()=>setView('camera')}>Expand camera ↗</button>}</div>
 <div className="camera-surface" ref={cameraSurface}><div className="stage" style={{aspectRatio:`${size[0]}/${size[1]}`,width:Math.max(1,Math.min(surfaceSize[0],surfaceSize[1]*size[0]/size[1]))}}><video ref={video} muted playsInline style={{display:source==='camera'&&['live','frozen','tracking'].includes(phase)?'block':'none',...((phase==='frozen'||phase==='tracking')?{position:'absolute',inset:0}: {})}}/><canvas ref={canvas} style={{display:phase==='frozen'||(phase==='tracking'&&['initializing','tracking'].includes(tracking.status))?'block':'none'}}/>{(phase==='frozen'||phase==='tracking')&&<svg viewBox={`0 0 ${size[0]} ${size[1]}`} onClick={click} aria-label="Select board corners or inspect a square">{h&&<><polygon className="border" points={([[-3.5,-3.5],[35.5,-3.5],[35.5,35.5],[-3.5,35.5]] as Point[]).map(p=>project(h!,p).join(',')).join(' ')}/>{Array.from({length:64},(_,i)=>{const f=i%8,r=Math.floor(i/8),id=String.fromCharCode(97+f)+(r+1),poly=displayPolys?.[id]??squarePolygon(h!,f,r),center=poly.reduce((a,p)=>[a[0]+p[0]/4,a[1]+p[1]/4] as Point,[0,0] as Point);return <g key={id}><polygon className={selected===id?'chosen':supported.includes(id)?'square piece-supported':'square'} points={poly.map(p=>p.join(',')).join(' ')}/><text x={center[0]} y={center[1]} fontSize={Math.max(12,size[0]/65)}>{id}</text></g>;})}</>}{displayCorners.map(([x,y],i)=><g key={i}><circle cx={x} cy={y} r={size[0]/80}/><text x={x+size[0]/70} y={y}>{i+1}</text></g>)}</svg>}</div>{!cameraActive&&<div className="camera-placeholder"><span className="camera-symbol" aria-hidden="true">◎</span><h2>Your physical board</h2><p>Start the rear camera and keep<br/>the board and borders in view.</p></div>}</div>
 </div>
 <div className="board-pane"><div className="full-board"><BoardPreviewBoundary resetKey={recognition.result?.frame_id??'none'}><DigitalBoard observation={boardObservation} statusLabel={boardStatus}/></BoardPreviewBoundary></div><button className="board-peek" onClick={()=>setView('board')} aria-label="Expand digital board"><BoardPreviewBoundary resetKey={recognition.result?.frame_id??'none'}><DigitalBoard observation={boardObservation} statusLabel={boardStatus} compact/></BoardPreviewBoundary><span>{boardObservation?'Digital board ↗':'Waiting for pieces'}</span></button></div>
 </div>
 <div className="scan-feedback"><p role="status">{phase==='tracking'?(tracking.status==='tracking'?recognitionStatus:`${tracking.reason} ${recognitionStatus}`):message}</p>{phase==='frozen'&&<p className="corner-guide">{corners.length<4?`Tap playing-area corners: a1 → h1 → h8 → a8. Next: ${['a1','h1','h8','a8'][corners.length]}`:'Check square identities and boundaries, then start tracking.'}</p>}</div>
 <div className="scan-progress" aria-label="Scan evidence"><span>Coordinates <strong>{phase==='tracking'&&tracking.status==='tracking'?'64 / 64':phase==='tracking'?'Finding…':'Not tracked'}</strong></span><span>Position <strong>{recognition.mode==='confirmed'?'Confirmed':recognition.mode==='review'?'Review needed':recognition.mode==='aiming'?'Choose a view':recognition.mode==='capturing'?'Reading pieces…':recognition.mode==='paused'?'Paused':'Scanning…'}</strong></span></div>
 <div className="scan-dock">
 {['idle','requesting'].includes(phase)&&<button className="primary-button" onClick={start} disabled={phase==='requesting'}>{phase==='requesting'?'Opening camera…':'Start scanning'}</button>}
 {phase==='live'&&<button disabled className="primary-button">Starting scan…</button>}
 {phase==='frozen'&&<><button className="primary-button" disabled={alignBusy} onClick={()=>void analyze()}>{alignBusy?'Finding board…':'Find board automatically'}</button><button disabled={alignBusy||!h||source!=='camera'} onClick={()=>{try{tracking.begin(corners,autoPolys);setPhase('tracking');setSelected('');}catch(e){setMessage((e as Error).message);}}}>Track alignment</button></>}
 {phase==='tracking'&&recognition.mode==='confirmed'&&<button className="primary-button" onClick={retryRecognition}>Rescan position</button>}
 {phase==='tracking'&&recognition.mode!=='confirmed'&&recognition.mode!=='review'&&recognition.mode!=='aiming'&&recognition.mode!=='capturing'&&<><button className="primary-button" disabled={!recognition.enabled&&tracking.status!=='tracking'} onClick={()=>recognition.enabled?recognition.pause():void recognition.resume()}>{recognition.enabled?'Pause scanning':'Resume scanning'}</button></>}
 {phase==='tracking'&&recognition.mode==='aiming'&&<button className="primary-button" disabled={tracking.status!=='tracking'} onClick={()=>void recognition.captureView(tracking.captureVerifiedFrame())}>Recognize this view</button>}
 {phase==='tracking'&&recognition.mode==='capturing'&&<button className="primary-button" disabled>Reading pieces…</button>}
 <details ref={actionMenu} className="action-menu" onKeyDown={e=>{if(e.key==='Escape')actionMenu.current?.removeAttribute('open');}}><summary>More ···</summary><div className="action-menu-items" onClick={()=>actionMenu.current?.removeAttribute('open')}>
 {cameraActive&&<button onClick={stopCamera}>Stop camera</button>}
 {['live','tracking'].includes(phase)&&<button onClick={freeze}>Manual alignment / freeze</button>}
 <button onClick={demo}>Try demo</button>
 <button disabled={!recognition.enabled&&!recognition.result} onClick={recognition.newPosition}>New position / reset predictions</button>
 <button disabled={!recognition.result} onClick={recognition.exportLatest}>Download latest recognition</button>
 {phase==='frozen'&&<><button disabled={alignBusy||corners.length!==4} onClick={()=>void analyze(true)}>Refine selected corners</button><button disabled={alignBusy} onClick={()=>{setAutoPolys(null);setCorners(corners.slice(0,-1));setSelected('');}}>Undo corner</button><button disabled={alignBusy} onClick={()=>{setAutoPolys(null);setCorners([]);setSelected('');}}>Reset alignment</button></>}
 {diagnostic&&<button onClick={exportDiagnostic}>Download alignment diagnostic</button>}
 </div></details>
 </div>
 <p className="scan-note">{recognition.mode==='confirmed'?'Position locked. Rescan after moving pieces or changing boards.':recognition.mode==='aiming'?'Choose a clear, steady view with every square visible. Recognition starts when you tap the button.':'Keep pieces still. The first successful result opens for your confirmation.'}</p>
 {error&&<p className="alignment-error" role="alert">{error}</p>}
 </div>
 <PositionReview observation={recognition.mode==='review'?recognition.result:null} onConfirm={recognition.confirm} onRetry={retryRecognition}/>
 <div className="support-panels"><details><summary>Recording and scan export</summary><Recorder video={video} canvas={canvas} phase={phase==='tracking'?'live':phase} corners={corners} source={source} frozenAt={frozenAt} refinedPolygons={autoPolys} onBoardChange={recognition.newPosition}/></details><details><summary>Alignment help</summary><p>Select the playing-area corners in semantic order: a1, h1, h8, a8. Select the square grid, not the outer border. The dashed outline shows the expected physical board border.</p><p>In a frozen frame, tap a square to inspect its identity. Selected: {selected||'none'}. {autoPolys?'Boundaries were refined using image evidence.':'Manual corners project the square boundaries.'}</p><p>Keep the camera approximately still during frozen alignment review. Automatic alignment sends that frame to your laptop.</p></details><details><summary>Device capabilities</summary><pre>{JSON.stringify({secureContext:window.isSecureContext,camera:!!navigator.mediaDevices?.getUserMedia,videoFrameCallback:'requestVideoFrameCallback' in HTMLVideoElement.prototype,dimensions:size,worker:typeof Worker!=='undefined',wasm:typeof WebAssembly!=='undefined',browser:navigator.userAgent,tracking:tracking.status,trackingWorkerMs:tracking.latency,recognition:recognition.enabled?'selected-frame':'off',acceptedRecognitionFrames:recognition.accepted},null,2)}</pre></details><details><summary>Privacy and processing</summary><p>Recording stays in this browser until export. Enabling recognition sends selected images and their matching geometry to your laptop. Automatic alignment sends selected live frames to your local laptop server; manual alignment sends a frozen frame. Tracking runs locally in a browser worker. Periodic orientation checks and automatic recovery send selected frames to your laptop. Recording does not yet attach tracking geometry to continuous frames.</p></details></div>
 </main>;
}
createRoot(document.getElementById('root')!).render(<App/>);
