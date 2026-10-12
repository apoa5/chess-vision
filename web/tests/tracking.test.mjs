import test from 'node:test';
import assert from 'node:assert/strict';
import {BorderTracker,fitTransform,robustTransform,multiply,inverse} from '../src/tracking.ts';
import {project,homography} from '../src/geometry.ts';
const corners=[[60,260],[260,260],[260,60],[60,60]];
function scene(dx=0,dy=0,occluded=false){const width=320,height=320,pixels=new Uint8Array(width*height);for(let y=0;y<height;y++)for(let x=0;x<width;x++){const u=x-dx,v=y-dy;let value=30;if(u>=38&&u<=282&&v>=38&&v<=282){if(u>=60&&u<=260&&v>=60&&v<=260)value=((Math.floor((u-60)/25)+Math.floor((v-60)/25))%2)?180:70;else {let seed=(Math.imul(u+19,374761393)+Math.imul(v+31,668265263))>>>0;seed=Math.imul(seed^(seed>>>13),1274126177)>>>0;value=50+(seed%180);}}pixels[y*width+x]=occluded?100:value;}return {width,height,pixels};}
test('projective fit rejects degenerate features and preserves arbitrary square orientation',()=>{const points=[[20,30],[200,30],[200,220],[20,220],[100,100],[150,180]],h=[1.01,.02,4,-.01,.99,-3,.0001,-.0001,1],fit=fitTransform(points,points.map(p=>project(h,p)));for(const p of points){const a=project(fit,p),b=project(h,p);assert.ok(Math.hypot(a[0]-b[0],a[1]-b[1])<1e-6);}assert.throws(()=>fitTransform([[1,1],[2,2],[3,3],[4,4]],[[1,1],[2,2],[3,3],[4,4]]));});
test('RANSAC ignores inconsistent feature matches rather than moving the board onto them',()=>{const from=Array.from({length:30},(_,i)=>[20+(i%6)*35,30+Math.floor(i/6)*40]),to=from.map(([x,y],i)=>i<7?[y+90,x-40]:[x+4,y-3]);const result=robustTransform(from,to);assert.ok(result);assert.equal(result.indices.length,23);assert.ok(result.residual<1e-6);assert.equal(robustTransform(from,from.map((_,i)=>[(i*97)%300,(i*43)%290])),null);});
test('border tracking follows small translations and keeps semantic corners',()=>{const tracker=new BorderTracker(scene(),corners,0);for(let i=1;i<=4;i++){const result=tracker.update(scene(i*2,i),i*100);assert.equal(result.state,'tracking',result.reason);for(let j=0;j<4;j++)assert.ok(Math.hypot(result.corners[j][0]-corners[j][0]-i*2,result.corners[j][1]-corners[j][1]-i)<1.5);homography(result.corners);}});
test('occlusion immediately hides geometry; persistent failure, frame gaps and dimension changes lose tracking',()=>{
 const tracker=new BorderTracker(scene(),corners,0);
 for(let i=1;i<=3;i++){const result=tracker.update(scene(0,0,true),i*100);assert.equal(result.state,i<3?'degraded':'lost');assert.equal(result.corners,undefined);}
 assert.equal(tracker.update(scene(),400).state,'lost');
 for(const [image,timestamp] of [[scene(),900],[{width:300,height:320,pixels:new Uint8Array(300*320)},100]])assert.equal(new BorderTracker(scene(),corners,0).update(image,timestamp).state,'lost');
});
test('a textureless border cannot initialize from the pieces or checker pattern',()=>{const image=scene();for(let y=0;y<320;y++)for(let x=0;x<320;x++)if(x<60||x>260||y<60||y>260)image.pixels[y*320+x]=110;const tracker=new BorderTracker(image,corners,0);assert.equal(tracker.update(image,100).state,'lost');});
test('initialization preserves a1/h1/h8/a8 for every quarter-turn view',()=>{for(let turn=0;turn<4;turn++){const transform=turn===0?[1,0,0,0,1,0,0,0,1]:turn===1?[0,-1,319,1,0,0,0,0,1]:turn===2?[-1,0,319,0,-1,319,0,0,1]:[0,1,0,-1,0,319,0,0,1],base=scene(),rotated={width:320,height:320,pixels:new Uint8Array(320*320)};for(let y=0;y<320;y++)for(let x=0;x<320;x++){const [u,v]=project(transform,[x,y]);rotated.pixels[v*320+u]=base.pixels[y*320+x];}const semantic=corners.map(p=>project(transform,p)),tracker=new BorderTracker(rotated,semantic,0),result=tracker.update(rotated,100);assert.equal(result.state,'tracking',result.reason);result.corners.forEach((p,i)=>assert.ok(Math.hypot(p[0]-semantic[i][0],p[1]-semantic[i][1])<1));}});
test('one visible edge cannot support a whole-board alignment',()=>{const base=scene(),partial={...base,pixels:new Uint8Array(base.pixels)};for(let y=0;y<320;y++)for(let x=0;x<320;x++)if(y<260)partial.pixels[y*320+x]=100;const result=new BorderTracker(base,corners,0).update(partial,100);assert.equal(result.state,'degraded');assert.equal(result.corners,undefined);});

test('a single bad camera frame does not permanently lose the reviewed anchor',()=>{const tracker=new BorderTracker(scene(),corners,0);const bad=tracker.update(scene(0,0,true),100);assert.equal(bad.state,'degraded');assert.equal(bad.transform,undefined);const recovered=tracker.update(scene(),200);assert.equal(recovered.state,'tracking',recovered.reason);recovered.corners.forEach((p,i)=>assert.ok(Math.hypot(p[0]-corners[i][0],p[1]-corners[i][1])<1));});
test('stationary camera tolerates exposure/noise changes without gradually deleting features',()=>{const base=scene(),tracker=new BorderTracker(base,corners,0);let features;for(let frame=1;frame<=20;frame++){const pixels=new Uint8Array(base.pixels.length);for(let i=0;i<pixels.length;i++)pixels[i]=Math.min(255,Math.max(0,base.pixels[i]*.85+15+((i*17+frame*13)%13-6)*2));const result=tracker.update({...base,pixels},frame*100);assert.equal(result.state,'tracking',result.reason);features??=result.features;assert.equal(result.features,features);result.corners.forEach((p,i)=>assert.ok(Math.hypot(p[0]-corners[i][0],p[1]-corners[i][1])<1));}});
test('clockwise camera walk preserves square identities through every diagonal transition',()=>{
 const width=512,height=512,base={width,height,pixels:new Uint8Array(width*height).fill(30)},small=scene();
 for(let y=0;y<320;y++)for(let x=0;x<320;x++)base.pixels[(y+96)*width+x+96]=small.pixels[y*320+x];
 const semantic=corners.map(([x,y])=>[x+96,y+96]),tracker=new BorderTracker(base,semantic,0);let maximumError=0;
 for(let step=1;step<=120;step++){
 const angle=-step*3*Math.PI/180,cos=Math.cos(angle),sin=Math.sin(angle),h=multiply(multiply(multiply([1,0,256,0,1,256,0,0,1],[cos,-sin,0,sin,cos,0,0,0,1]),[1,0,0,0,.85+.15*cos,0,.0007*sin,.0004*(1-cos),1]),[1,0,-256,0,1,-256,0,0,1]),pixels=new Uint8Array(width*height),inv=inverse(h);
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const d=inv[6]*x+inv[7]*y+inv[8],u=(inv[0]*x+inv[1]*y+inv[2])/d,v=(inv[3]*x+inv[4]*y+inv[5])/d,ix=Math.floor(u),iy=Math.floor(v);if(ix<0||iy<0||ix>=width-1||iy>=height-1)continue;const dx=u-ix,dy=v-iy,i=iy*width+ix;pixels[y*width+x]=(1-dy)*((1-dx)*base.pixels[i]+dx*base.pixels[i+1])+dy*((1-dx)*base.pixels[i+width]+dx*base.pixels[i+width+1]);}
 const result=tracker.update({width,height,pixels},step*100);assert.equal(result.state,'tracking',`step ${step}: ${result.reason}`);
 result.corners.forEach((p,i)=>{const truth=project(h,semantic[i]),error=Math.hypot(p[0]-truth[0],p[1]-truth[1]);maximumError=Math.max(maximumError,error);assert.ok(error<5,`step ${step}, corner ${i}: ${error}px`);});
 }
 assert.ok(maximumError<5);console.log(`Full-circle synthetic perspective test: maximum corner error ${maximumError.toFixed(2)} px`);
});
import {verifiedMarkingGeometry,markingDisagrees} from '../src/trackingChecks.ts';
import {squarePolygon} from '../src/geometry.ts';
test('automatic recovery requires semantic evidence and rejects shifted square allocations',()=>{
 const h=homography(corners),polygons=Object.fromEntries(Array.from({length:64},(_,i)=>[String.fromCharCode(97+i%8)+(Math.floor(i/8)+1),squarePolygon(h,i%8,Math.floor(i/8))]));
 const candidate={status:'candidate',orientation_source:'learned_border_markings',learned_marking_evidence:{inliers:9,distinct_files:4,distinct_ranks:5},geometry:{corners},square_polygons:polygons};
 assert.ok(verifiedMarkingGeometry(candidate,320,320));
 for(const bad of [{...candidate,status:'uncertain'},{...candidate,orientation_source:'manual'},{...candidate,learned_marking_evidence:{inliers:8,distinct_files:1,distinct_ranks:7}},{...candidate,square_polygons:{...polygons,a1:polygons.a2}},{...candidate,geometry:{corners:[...corners].reverse()}}])assert.equal(verifiedMarkingGeometry(bad,320,320),null);
 assert.equal(markingDisagrees(corners,corners),false);assert.equal(markingDisagrees(corners,[...corners.slice(1),corners[0]]),true);
});
import {distributedBorderSupport} from '../src/tracking.ts';
test('a temporarily obscured corner retains tracking with broad evidence in the other three regions',()=>{
 const base=scene(),partial={...base,pixels:new Uint8Array(base.pixels)};
 for(let y=160;y<320;y++)for(let x=160;x<320;x++)if(x>260||y>260)partial.pixels[y*320+x]=100;
 const tracker=new BorderTracker(base,corners,0),result=tracker.update(partial,100);
 assert.equal(result.state,'tracking',result.reason);result.corners.forEach((p,i)=>assert.ok(Math.hypot(p[0]-corners[i][0],p[1]-corners[i][1])<1));
 assert.equal(tracker.update(base,200).state,'tracking');
});
test('coverage rejects one edge and near-collinear diagonal evidence despite a large feature count',()=>{
 assert.equal(distributedBorderSupport(Array.from({length:30},(_,i)=>[20+i*9,25])),false);
 assert.equal(distributedBorderSupport(Array.from({length:30},(_,i)=>[20+i*9,20+i*9+(i%2)])),false);
});
