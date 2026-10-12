import React from 'react';
// Original SVG artwork: locally rendered, no icon font or external image requests.
const names:Record<string,string>={p:'pawn',n:'knight',b:'bishop',r:'rook',q:'queen',k:'king'};
export function pieceName(symbol:string|null|undefined):string {if(typeof symbol!=='string'||!symbol)return 'Empty square';const name=names[symbol.toLowerCase()];return name?`${symbol===symbol.toUpperCase()?'White':'Black'} ${name}`:'Empty square';}
export function ChessPiece({symbol}:{symbol:string|null|undefined}){
 if(typeof symbol!=='string'||!symbol)return null;
 const type=symbol.toLowerCase();if(!names[type])return null;
 const white=symbol===symbol.toUpperCase();
 return <svg className={`chess-piece ${white?'white-piece':'black-piece'}`} viewBox="0 0 60 60" role="img" aria-label={pieceName(symbol)}><title>{pieceName(symbol)}</title><g fill={white?'#fff7e7':'#28313b'} stroke={white?'#35404a':'#d6dee4'} strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round">
 {type==='p'&&<><circle cx="30" cy="16" r="7"/><path d="M25 23h10l-1 7c0 6 4 10 8 13H18c4-3 8-7 8-13z"/><path d="M19 43h22l3 6H16z"/></>}
 {type==='n'&&<><path d="M18 44c-1-10 5-16 12-22l-8 5-8-2 3-8 10-7 2-7 7 7c11 4 14 15 12 34z"/><path d="M21 17l7-3m8 4c6 7 6 15 3 21" fill="none"/><circle cx="26" cy="18" r="1.5" fill={white?'#35404a':'#d6dee4'}/><path d="M18 44h28l2 6H15z"/></>}
 {type==='b'&&<><circle cx="30" cy="8" r="3"/><path d="M30 11c-5 5-12 10-11 16 1 5 6 7 11 7s10-2 11-7c1-6-6-11-11-16z"/><path d="M25 18l8 10" fill="none"/><path d="M26 34h8c-1 5 3 8 7 11H19c4-3 8-6 7-11z"/><path d="M18 45h24l4 5H14z"/></>}
 {type==='r'&&<><path d="M16 10h7v7h5v-7h5v7h5v-7h7v15l-6 4 2 15H19l2-15-5-4z"/><path d="M17 24h27M22 30h16" fill="none"/><path d="M18 44h24l4 6H14z"/></>}
 {type==='q'&&<><path d="M16 17l7 8 7-13 7 13 7-8-5 17H21z"/><circle cx="15" cy="14" r="3"/><circle cx="30" cy="9" r="3"/><circle cx="45" cy="14" r="3"/><path d="M22 34h16l-2 5 6 6H18l6-6z"/><path d="M19 30h22M22 38h16" fill="none"/><path d="M17 45h26l3 5H14z"/></>}
 {type==='k'&&<><path d="M27 5h6v5h5v5h-5v6h-6v-6h-5v-5h5z"/><path d="M30 24c-12-15-23-4-16 7l7 8h18l7-8c7-11-4-22-16-7z"/><path d="M30 24v13M22 39h16l3 6H19z"/><path d="M17 45h26l3 5H14z"/></>}
 <path d="M14 50h32l3 5H11z"/></g></svg>;
}
