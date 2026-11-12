import React,{useEffect,useRef,useState} from 'react';
import {DigitalBoard} from './DigitalBoard';
import {ChessPiece,pieceName} from './ChessPiece';
import {BoardPreviewBoundary} from './BoardPreviewBoundary';
import {MultiViewScan} from './MultiViewScan';
import {PIECES,SQUARES,loadReview,pieceEdit,undoPiece,resetPieces,editGame,previewFen,placement,structuralErrors,acceptConfirmation,type Review,type PhotoResult,type Piece} from './photoReview';
import {photoType,recognizePhoto,confirmDraft,openPhotoCamera,capturePhoto} from './photoApi';

export function AssistedPhotoApp(){
 const [captureMode,setCaptureMode]=useState<'single'|'multi'|'video'>('single');
 const [file,setFile]=useState<File|null>(null),[url,setUrl]=useState(''),[phase,setPhase]=useState<'empty'|'camera'|'ready'|'recognizing'|'draft'>('empty');
 const [result,setResult]=useState<PhotoResult|null>(null),[review,setReview]=useState<Review|null>(null),[selected,setSelected]=useState('');
 const [message,setMessage]=useState('Choose a photo to begin.'),[error,setError]=useState(''),[enabled,setEnabled]=useState<boolean|null>(null),[confirmBusy,setConfirmBusy]=useState(false);
 const [camera,setCamera]=useState<MediaStream|null>(null),[manual,setManual]=useState(false),[manualOpen,setManualOpen]=useState(false),[corners,setCorners]=useState<number[][]>([]),[whiteSide,setWhiteSide]=useState(''),[showGrid,setShowGrid]=useState(true);
 const video=useRef<HTMLVideoElement>(null),image=useRef<HTMLImageElement>(null),fenBox=useRef<HTMLTextAreaElement>(null),stream=useRef<MediaStream|null>(null),controller=useRef<AbortController|null>(null),generation=useRef(0),current=useRef<Review|null>(null);
 const stopCamera=()=>{stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;setCamera(null);if(video.current)video.current.srcObject=null;};
 const cancelPending=()=>{generation.current++;controller.current?.abort();controller.current=null;setConfirmBusy(false);};
 const update=(fn:(s:Review)=>Review)=>{if(!current.current)return;const next=fn(current.current);current.current=next;setReview(next);setError('');if(!next.confirmed)setMessage('Review your changes before confirming.');};
 const clearDraft=()=>{current.current=null;setReview(null);setResult(null);setSelected('');};
 const checkServer=async()=>{try{const response=await fetch('/api/photo/status');if(!response.ok)throw Error('Server unavailable');const body=await response.json();setEnabled(body.enabled===true);}catch{setEnabled(null);setError('Cannot reach the server. Check your connection and retry.');}};
 useEffect(()=>{void checkServer();return()=>{generation.current++;controller.current?.abort();stream.current?.getTracks().forEach(t=>t.stop());};},[]);
 useEffect(()=>{if(!file){setUrl('');return;}const next=URL.createObjectURL(file);setUrl(next);return()=>URL.revokeObjectURL(next);},[file]);
 useEffect(()=>{if(camera&&video.current){video.current.srcObject=camera;void video.current.play().catch(()=>setError('Camera preview could not start. Try uploading a photograph.'));}},[camera]);
 useEffect(()=>{const hide=()=>{if(document.hidden&&stream.current){cancelPending();stopCamera();setPhase(file?'ready':'empty');setMessage('Camera stopped while this page was hidden.');}};document.addEventListener('visibilitychange',hide);return()=>document.removeEventListener('visibilitychange',hide);},[file]);
 const choose=(next:File)=>{cancelPending();stopCamera();clearDraft();setCorners([]);setManual(false);setManualOpen(false);setWhiteSide('');setError('');try{photoType(next);setFile(next);setPhase('ready');setMessage('Photo ready.');}catch(e){setFile(null);setPhase('empty');setError((e as Error).message);}};
 const startCamera=async()=>{cancelPending();stopCamera();clearDraft();setFile(null);setPhase('camera');setError('');setMessage('Allow camera access to take a photo.');const token=generation.current;try{const next=await openPhotoCamera();if(token!==generation.current||document.hidden){next.getTracks().forEach(t=>t.stop());return;}stream.current=next;setCamera(next);next.getVideoTracks()[0]?.addEventListener('ended',()=>{stopCamera();setPhase('empty');setMessage('Camera disconnected. Try again or upload a photo.');},{once:true});}catch(e){if(token===generation.current){setPhase('empty');setError((e as Error).message);}}};
 const capture=async()=>{try{if(!video.current)throw Error('Camera not ready.');choose(await capturePhoto(video.current));}catch(e){setError((e as Error).message);}};
 const reset=()=>{cancelPending();stopCamera();clearDraft();setFile(null);setCorners([]);setManual(false);setWhiteSide('');setPhase('empty');setError('');setMessage('Choose a new photograph.');};
 const recognize=async()=>{if(!file)return;cancelPending();clearDraft();const token=generation.current,abort=new AbortController();controller.current=abort;setPhase('recognizing');setError('');setMessage('Reading your board…');const timeout=setTimeout(()=>abort.abort(),90000);
  try{const response=await recognizePhoto(file,{...(manual&&corners.length===4?{corners}:{}),...(whiteSide?{white_side:whiteSide}:{})},abort.signal);if(token!==generation.current)return;setResult(response);const draft=loadReview(response);current.current=draft;setReview(draft);setSelected('');setPhase('draft');setMessage('Check the board and correct any mistakes.');}
  catch(e){if(token!==generation.current)return;setPhase('ready');setError(abort.signal.aborted?'Recognition timed out. Retry; the server may still be finishing the previous request.':(e as Error).message);if(['localization','orientation','grid','invalid_geometry','crop_alignment'].includes((e as Error&{code?:string}).code??''))setManualOpen(true);}
  finally{clearTimeout(timeout);if(token===generation.current)controller.current=null;}
 };
 const cancel=()=>{cancelPending();setPhase(file?'ready':'empty');setMessage('Recognition cancelled.');};
 const confirm=async()=>{const draft=current.current;if(!draft?.checked)return;let preview:string;try{preview=previewFen(draft);}catch(e){setError((e as Error).message);return;}void preview;const revision=draft.revision,token=generation.current,abort=new AbortController();controller.current=abort;setConfirmBusy(true);setError('');const timeout=setTimeout(()=>abort.abort(),15000);
  try{const receipt=await confirmDraft(draft,abort.signal);if(token!==generation.current||!current.current)return;const accepted=acceptConfirmation(current.current,receipt,revision);current.current=accepted;setReview(accepted);setMessage(accepted.confirmed?'Position confirmed.':'Position changed. Review and confirm again.');}
  catch(e){if(token===generation.current)setError(abort.signal.aborted?'Confirmation timed out. Please try again.':(e as Error).message);}
  finally{clearTimeout(timeout);if(token===generation.current){setConfirmBusy(false);controller.current=null;}}
 };
 const copy=async()=>{const confirmed=current.current?.confirmed;if(!confirmed)return;try{if(!navigator.clipboard?.writeText)throw Error('Clipboard unavailable');await navigator.clipboard.writeText(confirmed.full_fen);setMessage('FEN copied.');}catch{fenBox.current?.focus();fenBox.current?.select();setMessage('Select the FEN below to copy it.');}};
 const edited=review?SQUARES.filter((_,i)=>review.board[i]!==review.original[i]):[];
 const board=review&&result?{squares:Object.fromEntries(SQUARES.map((s,i)=>[s,{...result.squares[s],symbol:review.board[i]||null}]))}:null;
 const hints=result?SQUARES.filter(s=>result.squares[s].confidence<.9||result.multiview?.disagreement_squares.includes(s)):[];
 const errors=review?structuralErrors(review.board):[];
 let fenPreview='';if(review)try{fenPreview=previewFen(review);}catch(e){fenPreview=(e as Error).message;}
 const status=review?.confirmed?'Confirmed position':review?(review.revision?'User-edited draft':'AI-generated draft'):captureMode!=='single'?(captureMode==='video'?'Video upload':'Multi-view scan'):phase==='recognizing'?'Recognition processing':file?'Photograph ready':'No selected image';
 return <main className="photo-app">
  <header className="photo-header"><h1>Chess Vision</h1></header>
  <>{(captureMode==='single'||review)&&<p className="photo-status" role="status">{message}</p>}</>
  <nav className="view-switch" aria-label="Recognition capture mode">{([['single','Upload Photo'],['video','Upload Video'],['multi','Live Scan']] as const).map(([mode,label])=><button key={mode} aria-pressed={captureMode===mode} onClick={()=>{if(captureMode===mode)return;cancelPending();stopCamera();clearDraft();setFile(null);setPhase('empty');setError('');setCaptureMode(mode);setMessage(mode==='single'?'Choose a photo to begin.':mode==='video'?'Choose a video to begin.':'Start scanning, then tap Finish Scan.');}}>{label}</button>)}</nav>
  {error&&<p className="photo-error" role="alert">{error}</p>}
  {enabled===false&&<p role="alert">Recognition is unavailable. <button onClick={()=>void checkServer()}>Check server</button>.</p>}
  <div className="photo-layout">
   {captureMode!=='single'&&<MultiViewScan key={captureMode} inputMode={captureMode==='video'?'video':'live'} onStart={()=>{cancelPending();stopCamera();clearDraft();setFile(null);setPhase('empty');}} onMessage={setMessage} onResult={(response,reference)=>{choose(reference);setResult(response);const draft=loadReview(response);current.current=draft;setReview(draft);setPhase('draft');setMessage('Check the board and correct any mistakes.');}}/>}
   <section className="photo-panel" hidden={captureMode!=='single'} aria-labelledby="photo-input-title"><h2 id="photo-input-title">1. Photo</h2>
    <div className="photo-actions"><label className="photo-upload">Choose photo<input aria-label="Upload photograph" type="file" accept="image/jpeg,image/png" onChange={e=>{const f=e.target.files?.[0];if(f)choose(f);e.target.value='';}}/></label><button onClick={()=>void startCamera()}>Open camera</button>{file&&<button onClick={reset}>Remove photograph</button>}</div>
    {phase==='camera'&&<div className="photo-camera"><video ref={video} muted playsInline autoPlay aria-label="Camera preview"/><div className="photo-actions"><button disabled={!camera} onClick={()=>void capture()}>Capture photograph</button><button onClick={()=>{cancelPending();stopCamera();setPhase('empty');setMessage('Camera closed.');}}>Close camera</button></div></div>}
    {url&&<div className={`photo-source ${manual?'manual-selection':''}`}><img ref={image} src={url} alt="Selected physical chessboard" onError={()=>{setError('The browser could not decode this photograph. Try a JPEG or PNG.');setPhase('empty');setFile(null);clearDraft();}} onClick={e=>{if(!manual||phase==='recognizing'||corners.length>=4)return;const box=e.currentTarget.getBoundingClientRect();setCorners([...corners,[(e.clientX-box.left)*e.currentTarget.naturalWidth/box.width,(e.clientY-box.top)*e.currentTarget.naturalHeight/box.height]]);}}/>
     {(result||manual)&&<svg viewBox={`0 0 ${result?.image.width||image.current?.naturalWidth||1} ${result?.image.height||image.current?.naturalHeight||1}`} aria-label="Board geometry overlay">
      {showGrid&&!manual&&result&&Object.entries(result.geometry.square_polygons).map(([s,p])=><g key={s}><polygon points={p.map(v=>v.join(',')).join(' ')} fill="none" stroke="#89ffc2" strokeWidth="2"/>{['a1','h1','a8','h8'].includes(s)&&<text x={p.reduce((a,v)=>a+v[0],0)/4} y={p.reduce((a,v)=>a+v[1],0)/4} fill="white" stroke="#102323" paintOrder="stroke" strokeWidth="3" fontSize="35" textAnchor="middle">{s}</text>}</g>)}
      {manual&&corners.map((p,i)=><g key={i}><circle cx={p[0]} cy={p[1]} r="12" fill="#ffbd6e"/><text x={p[0]+20} y={p[1]} fill="white" fontSize="32">{i+1}</text></g>)}
     </svg>}
    </div>}
    {result&&<p><label><input type="checkbox" checked={showGrid} onChange={e=>setShowGrid(e.target.checked)}/> Show grid</label></p>}
    {file&&<><details open={manualOpen} onToggle={e=>setManualOpen(e.currentTarget.open)}><summary>Adjust alignment</summary><p>Check the board edges and square labels.</p><label>White home edge (rank 1)<select aria-label="White home edge" value={whiteSide} disabled={phase==='recognizing'} onChange={e=>{cancelPending();clearDraft();setPhase('ready');setWhiteSide(e.target.value);}}><option value="">Automatic border markings</option>{['bottom','left','top','right'].map(side=><option key={side} value={side}>{side} of photograph — supplied by you</option>)}</select></label><button disabled={phase==='recognizing'} onClick={()=>{cancelPending();clearDraft();setPhase('ready');setManual(!manual);setCorners([]);}}>{manual?'Use automatic localization':'Select four playing-area corners'}</button>{manual&&<><p>Tap corners clockwise from top-left: {corners.length}/4 selected.</p><button onClick={()=>setCorners(corners.slice(0,-1))} disabled={phase==='recognizing'||!corners.length}>Undo corner</button></>}</details>
    <div className="photo-actions"><button className="primary-button" disabled={enabled===false||phase==='recognizing'||(manual&&corners.length!==4)} onClick={()=>void recognize()}>{result?'Recognize again (discard draft)':'Recognize photograph'}</button>{phase==='recognizing'&&<><progress aria-label="Recognizing photograph"/><button onClick={cancel}>Cancel recognition</button></>}</div></>}
    <p className="photo-note">JPEG or PNG · up to 12 MB</p>
   </section>
   <section className="photo-panel" aria-labelledby="photo-review-title"><h2 id="photo-review-title">2. Review board</h2>
    {!review&&<p>Your board will appear here.</p>}
    {review&&result&&<><p>Check every square before confirming.</p>
     {result.multiview&&url&&<details><summary>Check alignment</summary><p>Check that the grid matches the board.</p><div className="photo-source"><img src={url} alt="Selected multi-view reference photograph"/><svg viewBox={`0 0 ${result.image.width} ${result.image.height}`} aria-label="Multi-view reference square grid">{Object.entries(result.geometry.square_polygons).map(([s,p])=><g key={s}><polygon points={p.map(v=>v.join(',')).join(' ')} fill="none" stroke="#89ffc2" strokeWidth="2"/>{['a1','h1','a8','h8'].includes(s)&&<text x={p.reduce((a,v)=>a+v[0],0)/4} y={p.reduce((a,v)=>a+v[1],0)/4} fill="white" stroke="#102323" paintOrder="stroke" strokeWidth="3" fontSize="35" textAnchor="middle">{s}</text>}</g>)}</svg></div></details>}
     <div className="photo-board-area"><BoardPreviewBoundary resetKey={result.recognition_id}><DigitalBoard observation={board} selectedSquare={selected} onSquareSelect={setSelected} editedSquares={edited} suggestedSquares={hints} statusLabel={status}/></BoardPreviewBoundary></div>
     <p>{edited.length} square{edited.length===1?'':'s'} edited. Dot: edited · Dashed border: review suggested.</p>
     {result.multiview&&<details><summary>View disagreement: {result.multiview.disagreement_squares.length} squares</summary><p>Views may disagree. Check the highlighted squares.</p><p>{result.multiview.disagreement_squares.join(', ')||'No top-class disagreements.'}</p>{selected&&<ul>{result.multiview.per_view.map(v=><li key={v.frame_id}>{v.frame_id} · {(v.timestamp_ms/1000).toFixed(1)} s · {pieceName((v.squares[selected].symbol||'') as Piece)} · {(100*v.squares[selected].confidence).toFixed(1)}%</li>)}</ul>}</details>}
     <fieldset className="photo-palette"><legend>{selected?`Replace ${selected} · ${pieceName(review.board[SQUARES.indexOf(selected)])}`:'Select any board square to edit'}</legend><div>{PIECES.map(p=><button key={p||'empty'} disabled={!selected} aria-label={`Set ${selected||'selected square'} to ${pieceName(p)}`} aria-pressed={!!selected&&review.board[SQUARES.indexOf(selected)]===p} onClick={()=>update(s=>pieceEdit(s,selected,p as Piece))}>{p?<ChessPiece symbol={p}/>:<span>Empty</span>}<small>{p?pieceName(p):'Remove piece'}</small></button>)}</div></fieldset>
     <div className="photo-actions"><button disabled={!review.history.length} onClick={()=>update(undoPiece)}>Undo piece edit</button><button onClick={()=>update(resetPieces)}>Reset to AI prediction</button></div>
     <h2>3. Game state</h2><p>Set the game state before confirming.</p>
     <div className="photo-game"><label>Side to move<select aria-label="Side to move" value={review.game.side_to_move} onChange={e=>update(s=>editGame(s,{...s.game,side_to_move:e.target.value as 'w'|'b'}))}><option value="w">White</option><option value="b">Black</option></select></label>
      <fieldset><legend>Castling rights</legend>{[['K','White kingside'],['Q','White queenside'],['k','Black kingside'],['q','Black queenside']].map(([right,label])=><label key={right}><input type="checkbox" aria-label={label+' castling'} checked={review.game.castling.includes(right)} onChange={e=>{const checked=e.target.checked;update(s=>{const rights=new Set(s.game.castling.replace('-',''));if(checked)rights.add(right);else rights.delete(right);return editGame(s,{...s.game,castling:['K','Q','k','q'].filter(r=>rights.has(r)).join('')||'-'});});}}/>{label}</label>)}</fieldset>
     </div>
     {errors.length>0&&<ul className="photo-errors">{errors.map(e=><li key={e}>{e}</li>)}</ul>}
     <details><summary>FEN preview</summary><code className="photo-fen" data-testid="placement-preview">{placement(review.board)}</code><code className="photo-fen">{fenPreview}</code></details>
     <label className="photo-attestation"><input type="checkbox" checked={review.checked} onChange={e=>{const checked=e.target.checked;update(s=>({...s,checked,confirmed:null,revision:s.revision+1}));}}/>I have checked the board and game state.</label>
     <div className="photo-actions"><button className="primary-button" disabled={!review.checked||errors.length>0||confirmBusy} onClick={()=>void confirm()}>{confirmBusy?'Validating position…':'Confirm Position'}</button><button disabled={!review.confirmed} onClick={()=>void copy()}>Copy FEN</button></div>
     {review.confirmed&&<div className="photo-confirmed"><strong>Position confirmed</strong><textarea ref={fenBox} aria-label="Confirmed FEN" readOnly value={review.confirmed.full_fen}/></div>}
    </>}
   </section>
  </div>
  <footer><details><summary>About</summary><p>Noncommercial use · pretrained model license: CC BY-NC 4.0.</p></details></footer>
 </main>;
}
