import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {readFileSync} from 'node:fs';
import {PIECES,CLASS_NAMES,SQUARES,DEFAULT_GAME,loadReview,pieceEdit,undoPiece,resetPieces,editGame,placement,previewFen,structuralErrors,acceptConfirmation,validatePhotoResult} from '../src/photoReview.ts';
const require=createRequire(import.meta.url),ts=require('typescript');
require.extensions['.ts']=(module,filename)=>module._compile(ts.transpileModule(readFileSync(filename,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,filename);
const api=require('../src/photoApi.ts');
function result(){const pieces=Array(64).fill('');pieces[4]='k';pieces[60]='K';return {schema_version:'assisted_photo_result_v1',recognition_id:'photo1',status:'draft',confirmed:false,geometry:{status:'completed_unverified'},provisional_placement:placement(pieces),squares:Object.fromEntries(SQUARES.map((s,i)=>{const n=PIECES.indexOf(pieces[i]);return [s,{symbol:pieces[i]||null,class_name:CLASS_NAMES[n],confidence:1,logits:Array.from({length:13},(_,j)=>j===n?10:0),probabilities:Array.from({length:13},(_,j)=>j===n?1:0)}];}))};}
function checked(s){return {...s,checked:true};}
function receipt(s){return {schema_version:'confirmed_position_v1',status:'confirmed',user_confirmed:true,recognition_id:s.recognitionId,revision:s.revision,placement_fen:placement(s.board),full_fen:previewFen(s),board:s.board.map(p=>p||null),game_state:s.game};}
test('canonical coordinates and piece placement remain a8..h1',()=>{
 assert.equal(SQUARES[0],'a8');assert.equal(SQUARES[63],'h1');const s=loadReview(result());assert.equal(placement(s.board),'4k3/8/8/8/8/8/8/4K3');assert.equal(previewFen(s),'4k3/8/8/8/8/8/8/4K3 w - - 0 1');assert.equal(s.confirmed,null);assert.equal(s.checked,false);
});
test('add, replace, recolor, remove, undo and reset preserve the immutable AI draft',()=>{
 const start=loadReview(result());let s=pieceEdit(start,'b7','N');assert.equal(s.board[9],'N');assert.equal(s.original[9],'');s=pieceEdit(s,'b7','n');assert.equal(s.board[9],'n');s=pieceEdit(s,'b7','');assert.equal(s.board[9],'');s=undoPiece(s);assert.equal(s.board[9],'n');s=resetPieces(s);assert.deepEqual(s.board,start.board);assert.ok(Object.isFrozen(s.original));assert.throws(()=>pieceEdit(s,'z9','P'));
});
test('confirmation requires explicit checking and exact matching FEN, revision, image and game state',()=>{
 let s=loadReview(result());assert.equal(acceptConfirmation(s,receipt(s),s.revision).confirmed,null);s=checked(s);const r=receipt(s);assert.equal(acceptConfirmation(s,r,s.revision).confirmed,r);
 for(const bad of [{...r,full_fen:'wrong'},{...r,game_state:{...r.game_state,side_to_move:'b'}},{...r,board:r.board.slice(1)}])assert.throws(()=>acceptConfirmation(s,bad,s.revision));
 assert.equal(acceptConfirmation(s,{...r,recognition_id:'another'},s.revision).confirmed,null);
});
test('piece/game edits, undo/reset and new image invalidate confirmation and stale replies',()=>{
 let s=checked(loadReview(result()));const r=receipt(s);s=acceptConfirmation(s,r,s.revision);const edited=pieceEdit(s,'a2','P');assert.equal(edited.confirmed,null);assert.equal(edited.checked,false);assert.equal(acceptConfirmation(checked(edited),r,s.revision).confirmed,null);
 assert.equal(editGame(s,{...s.game,side_to_move:'b'}).confirmed,null);assert.equal(resetPieces(s).confirmed,null);assert.equal(undoPiece(edited).confirmed,null);assert.equal(loadReview({...result(),recognition_id:'new'}).confirmed,null);
});
test('defaults never grant castling rights; FEN stays synchronized with game-state edits',()=>{
 let s=loadReview(result());assert.equal(DEFAULT_GAME.castling,'-');s=editGame(s,{...s.game,side_to_move:'b',halfmove_clock:17,fullmove_number:42});assert.ok(previewFen(s).endsWith(' b - - 17 42'));assert.throws(()=>previewFen(editGame(s,{...s.game,halfmove_clock:NaN})));assert.throws(()=>previewFen(editGame(s,{...s.game,en_passant:'b5'})));
});
test('structural checks reject missing kings/back-rank pawns but permit promoted queens',()=>{
 let s=loadReview(result());assert.deepEqual(structuralErrors(s.board),[]);s=pieceEdit(s,'a8','P');assert.ok(structuralErrors(s.board).some(e=>e.includes('Pawns')));s=pieceEdit(s,'a8','Q');s=pieceEdit(s,'b2','Q');assert.deepEqual(structuralErrors(s.board),[]);assert.ok(structuralErrors(pieceEdit(s,'e1','').board).some(e=>e.includes('White king')));
});
test('recognition validation refuses unresolved, reordered classes, incomplete and inconsistent drafts',()=>{
 const good=result();assert.equal(validatePhotoResult(good),good);
 for(const bad of [{...good,geometry:{status:'unresolved'}},{...good,confirmed:true},{...good,provisional_placement:'wrong'},{...good,squares:{...good.squares,a8:{...good.squares.a8,class_name:'white_king'}}},{...good,squares:{...good.squares,a8:{...good.squares.a8,logits:[NaN]}}}])assert.throws(()=>validatePhotoResult(bad));
});
test('uploads are bounded JPEG/PNG and requests carry raw file bytes, manual geometry and cancellation',async()=>{
 const previous=globalThis.fetch;const file=new File([new Uint8Array([255,216,255,224])],'board.jpg',{type:'image/jpeg'});assert.equal(api.photoType(file),'image/jpeg');assert.throws(()=>api.photoType(new File(['x'],'board.heic',{type:'image/heic'})));assert.throws(()=>api.photoType(new File([],'empty.png',{type:'image/png'})));
 let received;globalThis.fetch=async(url,options)=>{received={url,options};return new Response(JSON.stringify(result()),{headers:{'Content-Type':'application/json'}});};
 try{const controller=new AbortController();await api.recognizePhoto(file,{white_side:'left'},controller.signal);assert.equal(received.options.body,file);assert.equal(received.options.signal,controller.signal);assert.deepEqual(JSON.parse(received.options.headers['X-Photo-Geometry']),{white_side:'left'});}finally{globalThis.fetch=previous;}
});
test('error responses preserve localization/orientation status instead of claiming success',async()=>{
 await assert.rejects(api.responseJson(new Response(JSON.stringify({code:'orientation',message:'Choose rank 1'}),{status:422})),e=>e.code==='orientation'&&e.message==='Choose rank 1');await assert.rejects(api.responseJson(new Response('<html>error</html>',{status:500})),/unreadable/);
});
test('camera uses secure context and capture frees its temporary canvas',async()=>{
 const oldWindow=globalThis.window,oldNavigator=Object.getOwnPropertyDescriptor(globalThis,'navigator'),oldDocument=globalThis.document;
 let request,canvas;
 Object.defineProperty(globalThis,'navigator',{value:{mediaDevices:{getUserMedia:async options=>{request=options;return {tracks:[]};}}},configurable:true});globalThis.window={isSecureContext:false};
 try{await assert.rejects(api.openPhotoCamera(),/HTTPS/);window.isSecureContext=true;await api.openPhotoCamera();assert.equal(request.audio,false);assert.equal(request.video.facingMode.ideal,'environment');
 globalThis.document={createElement:()=>canvas={width:0,height:0,getContext:()=>({drawImage:()=>{}}),toBlob:cb=>cb(new Blob(['capture'],{type:'image/jpeg'}))}};
 const captured=await api.capturePhoto({videoWidth:1920,videoHeight:1080});assert.equal(captured.type,'image/jpeg');assert.equal(canvas.width,0);assert.equal(canvas.height,0);
 }finally{globalThis.window=oldWindow;globalThis.document=oldDocument;if(oldNavigator)Object.defineProperty(globalThis,'navigator',oldNavigator);else delete globalThis.navigator;}
});

test('upload MIME follows JPEG/PNG bytes without reencoding, and unknown bytes never upload',async()=>{
 const previous=globalThis.fetch;const calls=[];globalThis.fetch=async(url,options)=>{calls.push(options);return new Response(JSON.stringify(result()));};
 try{
  const png=new File([new Uint8Array([137,80,78,71,13,10,26,10])],'mislabeled.jpg',{type:'image/jpeg'});
  await api.recognizePhoto(png,{},new AbortController().signal);assert.equal(calls[0].headers['Content-Type'],'image/png');assert.equal(calls[0].body,png);
  const jpeg=new File([new Uint8Array([255,216,255,224])],'mislabeled.png',{type:'image/png'});
  await api.recognizePhoto(jpeg,{},new AbortController().signal);assert.equal(calls[1].headers['Content-Type'],'image/jpeg');assert.equal(calls[1].body,jpeg);
  await assert.rejects(api.recognizePhoto(new File(['not jpeg'],'wrong.jpg',{type:'image/jpeg'}),{},new AbortController().signal),/not a JPEG or PNG/);assert.equal(calls.length,2);
  const cancel=new AbortController();cancel.abort();await assert.rejects(api.recognizePhoto(jpeg,{},cancel.signal),{name:'AbortError'});assert.equal(calls.length,2);
 }finally{globalThis.fetch=previous;}
});
