// Run the same tracker used by the worker; fixture images are prepared from a validated scan ZIP.
import fs from 'node:fs';
import {BorderTracker} from '../src/tracking.ts';
const fixture=JSON.parse(fs.readFileSync(0,'utf8'));
const image=f=>({...f,pixels:new Uint8Array(f.pixels)});
const tracker=new BorderTracker(image(fixture.frames[0]),fixture.corners,0),records=[];
for(let i=1;i<fixture.frames.length;i++){
 const start=performance.now(),result=tracker.update(image(fixture.frames[i]),i*100);
 const error=result.corners?Math.max(...result.corners.map((p,j)=>Math.hypot(p[0]-fixture.frames[i].corners[j][0],p[1]-fixture.frames[i].corners[j][1]))):null;
 records.push({step:i,...result,processing_ms:performance.now()-start,max_corner_error_processing_px:error});
}
console.log(JSON.stringify({schema_version:'controlled_border_tracking_v1',source_sha256:fixture.source_sha256,processing_dimensions:[fixture.frames[0].width,fixture.frames[0].height],kind:'synthetic_motion_of_real_phone_image',motion:'four small projective steps at simulated 100ms intervals',live_device_performance_measured:false,full_viewpoint_accuracy_measured:false,records},null,2));
