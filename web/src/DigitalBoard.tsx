import React,{useEffect,useRef,useState} from 'react';
import {ChessPiece,pieceName} from './ChessPiece';
import type {PieceObservation} from './useRecognition';
export function boardSquare(index:number,flipped=false):string {const i=flipped?63-index:index;return String.fromCharCode(97+i%8)+(8-Math.floor(i/8));}
export function DigitalBoard({observation,compact=false,statusLabel="Scan result · awaiting confirmation"}:{observation:PieceObservation|null;compact?:boolean;statusLabel?:string}){
 const [flipped,setFlipped]=useState(false),[selected,setSelected]=useState('');
 const container=useRef<HTMLDivElement>(null),heading=useRef<HTMLDivElement>(null),inspection=useRef<HTMLParagraphElement>(null);
 const [boardSize,setBoardSize]=useState(320);
 useEffect(()=>{const el=container.current;if(!el||compact)return;const resize=()=>{const box=el.getBoundingClientRect(),reserved=(heading.current?.getBoundingClientRect().height??40)+(inspection.current?.getBoundingClientRect().height??32)+16;setBoardSize(Math.max(1,Math.min(box.width,box.height-reserved,560)));};resize();if(typeof ResizeObserver==='undefined'){window.addEventListener('resize',resize);return()=>window.removeEventListener('resize',resize);}const observer=new ResizeObserver(resize);observer.observe(el);if(heading.current)observer.observe(heading.current);if(inspection.current)observer.observe(inspection.current);return()=>observer.disconnect();},[compact]);
 const p=selected?observation?.squares[selected]:undefined;
 return <div ref={container} className={`board-display ${compact?'compact-board':''}`}>
 {!compact&&<div ref={heading} className="board-heading"><div><strong>Digital board</strong><span>{observation?statusLabel:'Waiting for predictions'}</span></div><button className="quiet-button" onClick={()=>setFlipped(!flipped)} aria-label="Flip digital board" aria-pressed={flipped}>Flip board</button></div>}
 <div className={`digital-board ${observation?'':'unobserved-board'}`} style={compact?undefined:{width:boardSize,height:boardSize}} role="group" aria-label={flipped?'Digital board, Black at bottom':'Digital board, White at bottom'}>
 {Array.from({length:64},(_,i)=>{const id=boardSquare(i,flipped),prediction=observation?.squares[id],row=Math.floor(i/8),col=i%8,content=<><ChessPiece symbol={prediction?.symbol??''}/>{col===0&&<span className="rank-coordinate">{id[1]}</span>}{row===7&&<span className="file-coordinate">{id[0]}</span>}</>,className=`board-cell ${(row+col)%2?'dark-cell':'light-cell'} ${selected===id?'selected-cell':''}`;
 return compact?<div className={className} key={id}>{content}</div>:<button className={className} key={id} onClick={()=>setSelected(id)} aria-pressed={selected===id} aria-label={`${id}, ${prediction?pieceName(prediction.symbol):'not observed'}${prediction?`, model score ${Math.round(prediction.confidence*100)} percent`:''}`}>{content}</button>;})}
 {!observation&&!compact&&<div className="board-placeholder">Start scanning<br/>to see the scanned position.</div>}
 </div>
 {!compact&&<p ref={inspection} className="board-inspection" aria-live="polite">{p?<><strong>{selected} · {pieceName(p.symbol)}</strong><span>Model score {Math.round(p.confidence*100)}% · uncalibrated</span></>:observation?'Tap a square to inspect its piece and score.':'Keep the pieces still while scanning.'}</p>}
 </div>;
}
