import {BorderTracker} from './tracking';
import type {Gray} from './tracking';
import type {Point} from './geometry';
let tracker:BorderTracker|null=null;
self.onmessage=(event:MessageEvent<{kind:string;frame:Gray;corners:Point[];timestamp:number;id:number}>)=>{const {kind,frame,corners,timestamp,id}=event.data;try{if(kind==='initialize'){tracker=new BorderTracker(frame,corners,timestamp);postMessage({kind:'ready',id});}else {const start=performance.now();const result=tracker?.update(frame,timestamp)??{state:'lost',reason:'Tracker not initialized',inliers:0,features:0};postMessage({kind:'result',id,timestamp,result,processing_ms:performance.now()-start});}}catch(error){postMessage({kind:'error',id,reason:(error as Error).message});}};
