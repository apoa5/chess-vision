// Canonical draft/confirmation contract. No model score can confirm a position.
export const SQUARES=Array.from({length:64},(_,i)=>`${'abcdefgh'[i%8]}${8-Math.floor(i/8)}`);
export const PIECES=['','P','N','B','R','Q','K','p','n','b','r','q','k'] as const;
export const CLASS_NAMES=['empty','white_pawn','white_knight','white_bishop','white_rook','white_queen','white_king','black_pawn','black_knight','black_bishop','black_rook','black_queen','black_king'];
export type Piece=typeof PIECES[number];
export type GameState={side_to_move:'w'|'b';castling:string;en_passant:string;halfmove_clock:number;fullmove_number:number};
export const DEFAULT_GAME:GameState={side_to_move:'w',castling:'-',en_passant:'-',halfmove_clock:0,fullmove_number:1};
export type PhotoResult={schema_version:'assisted_photo_result_v1';recognition_id:string;status:'draft';confirmed:false;provisional_placement:string;elapsed_seconds:number;model:{backend:string;calibration:string};aggregation?:{method:string;score_kind:string;raw_logits_kind:string};multiview?:{disagreement_squares:string[];per_view:Array<{frame_id:string;timestamp_ms:number;squares:Record<string,{symbol:string|null;confidence:number}>}>};squares:Record<string,{symbol:string|null;class_name:string;confidence:number;logits:number[]|null;probabilities:number[];visibility:string;context_valid_fraction:number}>;geometry:{status:'completed_unverified';camera_orientation:string;orientation_source:string;square_polygons:Record<string,number[][]>;source_corners:number[][]};image:{width:number;height:number}};
export type Confirmed={schema_version:'confirmed_position_v1';status:'confirmed';user_confirmed:true;recognition_id:string;revision:number;full_fen:string;placement_fen:string;confirmation_id:string;board:(string|null)[];game_state:GameState};
export type Review={original:readonly Piece[];board:Piece[];history:Piece[][];game:GameState;revision:number;checked:boolean;confirmed:Confirmed|null;recognitionId:string};
export function placement(board:readonly Piece[]):string {
 if(board.length!==64||board.some(p=>!PIECES.includes(p)))throw Error('Expected 64 canonical squares.');
 return Array.from({length:8},(_,r)=>{let text='',empty=0;for(const p of board.slice(r*8,r*8+8)){if(!p){empty++;continue;}if(empty){text+=empty;empty=0;}text+=p;}return text+(empty||'');}).join('/');
}
export function loadReview(result:PhotoResult):Review {
 const original=SQUARES.map(s=>(result.squares[s].symbol??'') as Piece);
 return {original:Object.freeze([...original]),board:[...original],history:[],game:{...DEFAULT_GAME},revision:0,checked:false,confirmed:null,recognitionId:result.recognition_id};
}
export function pieceEdit(state:Review,square:string,piece:Piece):Review {
 const i=SQUARES.indexOf(square);if(i<0||!PIECES.includes(piece))throw Error('Invalid square/piece.');
 if(state.board[i]===piece)return state;
 const board=[...state.board];board[i]=piece;
 return {...state,board,history:[...state.history.slice(-127),[...state.board]],revision:state.revision+1,checked:false,confirmed:null};
}
export function undoPiece(state:Review):Review {
 if(!state.history.length)return state;
 return {...state,board:[...state.history[state.history.length-1]],history:state.history.slice(0,-1),revision:state.revision+1,checked:false,confirmed:null};
}
export function resetPieces(state:Review):Review {
 return {...state,board:[...state.original],history:[...state.history.slice(-127),[...state.board]],revision:state.revision+1,checked:false,confirmed:null};
}
export function editGame(state:Review,game:GameState):Review {
 return {...state,game:{...game},revision:state.revision+1,checked:false,confirmed:null};
}
export function structuralErrors(board:readonly Piece[]):string[] {
 const errors:string[]=[];
 for(const [king,color,pieces,pawn] of [['K','White','PNBRQK','P'],['k','Black','pnbrqk','p']]){
  if(board.filter(p=>p===king).length!==1)errors.push(`Place exactly one ${color} king.`);
  if(board.filter(p=>p&&pieces.includes(p)).length>16)errors.push(`${color} has more than 16 pieces.`);
  if(board.filter(p=>p===pawn).length>8)errors.push(`${color} has more than 8 pawns.`);
 }
 if([...board.slice(0,8),...board.slice(-8)].some(p=>p==='p'||p==='P'))errors.push('Pawns cannot be on rank 1 or rank 8.');
 return errors;
}
export function previewFen(state:Review):string {
 const g=state.game;
 if(!Number.isInteger(g.halfmove_clock)||g.halfmove_clock<0||!Number.isInteger(g.fullmove_number)||g.fullmove_number<1)throw Error('Use whole move counters: halfmove ≥ 0, fullmove ≥ 1.');
 if(!/^(-|[a-h][36])$/.test(g.en_passant))throw Error('En passant must be - or a square on rank 3 or 6.');
 return `${placement(state.board)} ${g.side_to_move} ${g.castling} ${g.en_passant} ${g.halfmove_clock} ${g.fullmove_number}`;
}
export function acceptConfirmation(state:Review,receipt:Confirmed,requestedRevision:number):Review {
 if(!state.checked||state.revision!==requestedRevision||receipt.revision!==requestedRevision||receipt.recognition_id!==state.recognitionId)return state;
 if(receipt.schema_version!=='confirmed_position_v1'||receipt.status!=='confirmed'||receipt.user_confirmed!==true||receipt.full_fen!==previewFen(state)||receipt.placement_fen!==placement(state.board)||!Array.isArray(receipt.board)||receipt.board.some((p,i)=>(p??'')!==state.board[i])||receipt.board.length!==64||Object.keys(DEFAULT_GAME).some(k=>receipt.game_state?.[k as keyof GameState]!==state.game[k as keyof GameState]))throw Error('Confirmation does not match the current draft.');
 return {...state,confirmed:receipt};
}
export function validatePhotoResult(value:PhotoResult):PhotoResult {
 if(value.schema_version!=='assisted_photo_result_v1'||value.status!=='draft'||value.confirmed!==false||typeof value.recognition_id!=='string'||value.geometry?.status!=='completed_unverified'||!value.squares||Object.keys(value.squares).length!==64)throw Error('Incompatible recognition response.');
 for(const s of SQUARES){const p=value.squares[s],index=p?PIECES.indexOf((p.symbol??'') as Piece):-1,probabilityMean=value.aggregation?.method==='mean_probabilities';if(!p||index<0||p.class_name!==CLASS_NAMES[index]||!Number.isFinite(p.confidence)||p.confidence<0||p.confidence>1||(p.logits===null?!probabilityMean:p.logits?.length!==13)||p.probabilities?.length!==13||[...(p.logits??[]),...p.probabilities].some(n=>!Number.isFinite(n))||p.probabilities.some(n=>n<0||n>1)||Math.abs(p.probabilities.reduce((a,b)=>a+b,0)-1)>.00001)throw Error('Incomplete square predictions.');}
 if(placement(SQUARES.map(s=>(value.squares[s].symbol??'') as Piece))!==value.provisional_placement)throw Error('Prediction/placement mismatch.');
 return value;
}
