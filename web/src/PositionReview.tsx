import React,{useEffect,useRef} from 'react';
import {DigitalBoard} from './DigitalBoard';
import {BoardPreviewBoundary} from './BoardPreviewBoundary';
import type {PieceObservation} from './useRecognition';
export function PositionReview({observation,onConfirm,onRetry}:{observation:PieceObservation|null;onConfirm:()=>void;onRetry:()=>void}){
 const dialog=useRef<HTMLDialogElement>(null);
 useEffect(()=>{const el=dialog.current;if(!el)return;const previous=document.activeElement as HTMLElement|null;if(observation&&!el.open)el.showModal();if(!observation&&el.open)el.close();return()=>{if(el.open)el.close();previous?.focus();};},[observation?.frame_id]);
 return <dialog ref={dialog} className="position-review" aria-labelledby="position-review-title" aria-describedby="position-review-help" onCancel={event=>event.preventDefault()}>
 <div className="review-content"><h2 id="position-review-title">Is this position correct?</h2><p id="position-review-help">Check every piece and empty square against your physical board. Piece recognition is paused while you review.</p>
 <div className="review-board"><BoardPreviewBoundary resetKey={observation?.frame_id??'none'}><DigitalBoard observation={observation}/></BoardPreviewBoundary></div>
 <div className="review-actions"><button onClick={onRetry}>No, try another view</button><button className="primary-button" onClick={onConfirm} autoFocus>Yes, confirm position</button></div><p className="review-note">Confirming keeps this position fixed while camera tracking continues.</p></div>
 </dialog>;
}
