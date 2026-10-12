export async function recognitionError(response:Response,operation:string):Promise<string>{
 const detail=(await response.text()).replace(/<[^>]*>/g,' ').replace(/\s+/g,' ').trim().slice(0,450);
 if(response.status===404||response.status===501)return `${operation} endpoint unavailable (${response.status}). Restart the Python server from this project using the updated script.`;
 if(response.status===429)return `${operation}: server busy (${response.status}). Try again when board analysis finishes.`;
 return `${operation} failed (${response.status})${detail?`: ${detail}`:'. Check the Python server terminal for details.'}`;
}
function delay(signal:AbortSignal){return new Promise<void>((resolve,reject)=>{const aborted=()=>{clearTimeout(timer);reject(new DOMException('Request cancelled','AbortError'));};const timer=setTimeout(()=>{signal.removeEventListener('abort',aborted);resolve();},1000);signal.addEventListener('abort',aborted,{once:true});if(signal.aborted)aborted();});}
export async function openRecognitionSession(signal:AbortSignal,onBusy:()=>void){
 for(let attempt=0;attempt<4;attempt++){
 const response=await fetch('/api/scan/session',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}',signal});
 if(response.status===429&&attempt<3){await response.text();onBusy();await delay(signal);continue;}
 if(!response.ok)throw Error(await recognitionError(response,'Recognition session'));
 const session=await response.json();if(typeof session.session_id!=='string'||!session.session_id)throw Error('Recognition session returned an invalid session ID. Check the server response.');return session;
 }
 throw Error('Unable to open recognition session.');
}
