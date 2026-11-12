import {validatePhotoResult,type PhotoResult,type Review,type Confirmed} from './photoReview';
export const MAX_PHOTO_BYTES=12*1024*1024;
export function photoType(file:File):string {
 const type=file.type||(/\.jpe?g$/i.test(file.name)?'image/jpeg':/\.png$/i.test(file.name)?'image/png':'');
 if(!['image/jpeg','image/png'].includes(type))throw Error('Use JPEG or PNG. Convert HEIC before uploading.');
 if(!file.size||file.size>MAX_PHOTO_BYTES)throw Error('Choose a photograph up to 12 MiB.');
 return type;
}
export async function responseJson(response:Response){
 let body;try{body=await response.json();}catch{throw Error(`The server returned an unreadable response (${response.status}). Check the Python server and retry.`);}
 if(!response.ok){const error=new Error(body.message||`Request failed (${response.status}).`) as Error&{code?:string};error.code=body.code;throw error;}
 return body;
}
export async function recognizePhoto(file:File,manual:object,signal:AbortSignal):Promise<PhotoResult>{
 photoType(file);
 // Read only the signature; preserve the complete original bytes and EXIF.
 const signature=new Uint8Array(await file.slice(0,16).arrayBuffer());
 const type=signature[0]===255&&signature[1]===216&&signature[2]===255?'image/jpeg':
  [137,80,78,71,13,10,26,10].every((v,i)=>signature[i]===v)?'image/png':null;
 if(!type)throw Error('This file is not a JPEG or PNG image. Export HEIC/HEIF as JPEG or PNG; changing the filename is not a conversion.');
 if(signal.aborted)throw new DOMException('Recognition cancelled.','AbortError');
 const response=await fetch('/api/photo/recognize',{method:'POST',headers:{'Content-Type':type,'X-Photo-Geometry':JSON.stringify(manual)},body:file,signal});
 return validatePhotoResult(await responseJson(response));
}
export async function confirmDraft(state:Review,signal:AbortSignal):Promise<Confirmed>{
 return responseJson(await fetch('/api/photo/confirm',{method:'POST',headers:{'Content-Type':'application/json'},signal,
  body:JSON.stringify({recognition_id:state.recognitionId,revision:state.revision,user_confirmed:state.checked,board:state.board.map(p=>p||null),game_state:state.game})}));
}
export async function openPhotoCamera(){
 if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia)throw Error('Camera access needs trusted HTTPS or localhost. You can also upload a photograph.');
 return navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1920},height:{ideal:1080}},audio:false});
}
export async function capturePhoto(video:HTMLVideoElement):Promise<File>{
 if(!video.videoWidth||!video.videoHeight)throw Error('Wait for the camera preview before capturing.');
 const canvas=document.createElement('canvas');canvas.width=video.videoWidth;canvas.height=video.videoHeight;
 try{canvas.getContext('2d')!.drawImage(video,0,0);const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('Camera capture failed.')),'image/jpeg',.95));return new File([blob],'camera.jpg',{type:'image/jpeg'});}
 finally{canvas.width=0;canvas.height=0;}
}
