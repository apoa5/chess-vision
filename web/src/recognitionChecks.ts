export function responseIsCurrent(value:unknown,session:string,epoch:number,frame:string,version:number,hash:string):boolean {
 try{
 const r=value as {session_id:string;frame_id:string;board_epoch:number;geometry_version:number;orientation_version:number;source_sha256:string;completion_enabled:boolean;squares:Record<string,{symbol:string|null;confidence:number;class_name:string;probabilities:number[];logits:number[]}>};
 if(r.session_id!==session||r.board_epoch!==epoch||r.orientation_version!==epoch||r.frame_id!==frame||r.geometry_version!==version||r.source_sha256!==hash||r.completion_enabled!==false||Object.keys(r.squares).length!==64)return false;
 for(let rank=1;rank<=8;rank++)for(const file of 'abcdefgh'){const square=r.squares[file+rank];if(!square||(square.symbol!==null&&(typeof square.symbol!=='string'||!['','P','N','B','R','Q','K','p','n','b','r','q','k'].includes(square.symbol)))||typeof square.class_name!=='string'||!Number.isFinite(square.confidence)||square.confidence<0||square.confidence>1||square.probabilities.length!==13||square.logits.length!==13||square.logits.some(x=>!Number.isFinite(x))||square.probabilities.some(x=>!Number.isFinite(x)||x<0||x>1)||Math.abs(square.probabilities.reduce((a,b)=>a+b,0)-1)>1e-5)return false;}
 return true;
 }catch{return false;}
}
