import {homography, project} from './geometry.ts';
import type {Point, Matrix} from './geometry.ts';
export type Gray = {width:number;height:number;pixels:Uint8Array};
export type TrackResult = {state:'tracking'|'degraded'|'lost';reason:string;corners?:Point[];transform?:Matrix;inliers:number;features:number;residual?:number};
const identity=[1,0,0,0,1,0,0,0,1];
export function inverse(m:Matrix):Matrix {
 const [a,b,c,d,e,f,g,h,i]=m,det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
 if(Math.abs(det)<1e-10)throw Error('Singular transform');
 return [(e*i-f*h),(c*h-b*i),(b*f-c*e),(f*g-d*i),(a*i-c*g),(c*d-a*f),(d*h-e*g),(b*g-a*h),(a*e-b*d)].map(v=>v/det);
}
function pyramid(image:Gray):Gray[]{const out=[image];for(let level=1;level<3;level++){const p=out[level-1],w=Math.floor(p.width/2),h=Math.floor(p.height/2),pixels=new Uint8Array(w*h);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*2*p.width+x*2;pixels[y*w+x]=(p.pixels[i]+p.pixels[i+1]+p.pixels[i+p.width]+p.pixels[i+p.width+1])/4;}out.push({width:w,height:h,pixels});}return out;}
function patch(image:Gray,p:Point):number[]|null {const x=p[0],y=p[1],r=5;if(x<r||y<r||x>=image.width-r-1||y>=image.height-r-1)return null;const v=[];for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const u=x+dx,w=y+dy,ix=Math.floor(u),iy=Math.floor(w),fx=u-ix,fy=w-iy,i=iy*image.width+ix;v.push((1-fy)*((1-fx)*image.pixels[i]+fx*image.pixels[i+1])+fy*((1-fx)*image.pixels[i+image.width]+fx*image.pixels[i+image.width+1]));}const mean=v.reduce((a,b)=>a+b,0)/v.length;return v.map(a=>a-mean);}
function ncc(a:number[]|null,b:number[]|null):number {if(!a||!b)return -1;let ab=0,aa=0,bb=0;for(let i=0;i<a.length;i++){ab+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}return aa>100&&bb>100?ab/Math.sqrt(aa*bb):-1;}
function match(a:Gray[],b:Gray[],p:Point,guess:Point):{point:Point;score:number}|null {
 const search=(level:number,center:Point,radius:number)=>{const scale=2**level,template=patch(a[level],[p[0]/scale,p[1]/scale]);let best:Point=center,score=-1;for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){const point:Point=[Math.round(center[0])+dx,Math.round(center[1])+dy],value=ncc(template,patch(b[level],point));if(value>score){score=value;best=point;}}if(level===0&&score>.5){for(let axis=0;axis<2;axis++){const left:Point=[...best],right:Point=[...best];left[axis]--;right[axis]++;const l=ncc(template,patch(b[level],left)),r=ncc(template,patch(b[level],right)),d=l-2*score+r;if(d<-.001)best[axis]+=Math.max(-.5,Math.min(.5,.5*(l-r)/d));}}return {point:best,score};};
 // A nearly stationary camera should not depend on tiny, downsampled lettering.
 const local=search(0,guess,3);if(local.score>=.88)return local;
 let q=guess,score=-1;
 for(let level=2;level>=0;level--){const scale=2**level,found=search(level,[q[0]/scale,q[1]/scale],level===2?6:3);if(found.score<.35&&level>0)continue;q=[found.point[0]*scale,found.point[1]*scale];score=found.score;}
 const best=local.score>score?local:{point:q,score};return best.score>=.78?best:null;
}
function features(image:Gray,corners:Point[]):Point[]{const inv=inverse(homography(corners)),candidates:{p:Point;score:number}[]=[];for(let y=16;y<image.height-16;y+=2)for(let x=16;x<image.width-16;x+=2){const [u,v]=project(inv,[x,y]); // Flat border only; never initialize on a piece or inside the playing area.
 if(u< -3.2||v< -3.2||u>35.2||v>35.2||(u>-.5&&u<32.5&&v>-.5&&v<32.5))continue;
 let xx=0,yy=0,xy=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){const i=(y+dy)*image.width+x+dx,gx=image.pixels[i+1]-image.pixels[i-1],gy=image.pixels[i+image.width]-image.pixels[i-image.width];xx+=gx*gx;yy+=gy*gy;xy+=gx*gy;}const score=(xx+yy-Math.sqrt((xx-yy)**2+4*xy*xy))/2;if(score>1500)candidates.push({p:[x,y],score});}
 candidates.sort((a,b)=>b.score-a.score);const points:Point[]=[],regions=[0,0,0,0];for(const c of candidates){const [u,v]=project(inv,c.p),region=(u>=16?1:0)+(v>=16?2:0);if(regions[region]>=20)continue;if(points.every(p=>Math.hypot(p[0]-c.p[0],p[1]-c.p[1])>=10)){points.push(c.p);regions[region]++;}if(points.length>=80)break;}return points;}
// Least-squares projective fit in normalized image coordinates, also used by RANSAC.
export function fitTransform(from:Point[],to:Point[]):Matrix {if(from.length<4||from.length!==to.length)throw Error('Insufficient correspondences');const scale=640,normal=Array.from({length:8},()=>Array(9).fill(0));from.forEach(([px,py],k)=>{const x=px/scale,y=py/scale,u=to[k][0]/scale,v=to[k][1]/scale;for(const row of [[x,y,1,0,0,0,-u*x,-u*y,u],[0,0,0,x,y,1,-v*x,-v*y,v]])for(let i=0;i<8;i++)for(let j=0;j<9;j++)normal[i][j]+=row[i]*row[j];});for(let k=0;k<8;k++){let p=k;for(let i=k+1;i<8;i++)if(Math.abs(normal[i][k])>Math.abs(normal[p][k]))p=i;[normal[k],normal[p]]=[normal[p],normal[k]];const d=normal[k][k];if(Math.abs(d)<1e-10)throw Error('Degenerate features');for(let j=k;j<9;j++)normal[k][j]/=d;for(let i=0;i<8;i++)if(i!==k){const f=normal[i][k];for(let j=k;j<9;j++)normal[i][j]-=f*normal[k][j];}}const h=normal.map(row=>row[8]);return [h[0],h[1],h[2]*scale,h[3],h[4],h[5]*scale,h[6]/scale,h[7]/scale,1];}
export function robustTransform(from:Point[],to:Point[]):{matrix:Matrix;indices:number[];residual:number}|null {if(from.length<12)return null;let best:number[]=[];let seed=12345;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};for(let trial=0;trial<100;trial++){const sample=new Set<number>();while(sample.size<4)sample.add(Math.floor(random()*from.length));const ids=[...sample];try{const h=fitTransform(ids.map(i=>from[i]),ids.map(i=>to[i])),inliers=from.map((p,i)=>{const q=project(h,p);return Math.hypot(q[0]-to[i][0],q[1]-to[i][1])<2.5?i:-1;}).filter(i=>i>=0);if(inliers.length>best.length)best=inliers;}catch{/* Degenerate sample. */}}
 if(best.length<12||best.length/from.length<.65)return null;const matrix=fitTransform(best.map(i=>from[i]),best.map(i=>to[i])),errors=best.map(i=>{const q=project(matrix,from[i]);return Math.hypot(q[0]-to[i][0],q[1]-to[i][1]);}).sort((a,b)=>a-b);return {matrix,indices:best,residual:errors[Math.floor(errors.length/2)]};}
export function multiply(a:Matrix,b:Matrix):Matrix {return Array.from({length:9},(_,i)=>{const r=Math.floor(i/3),c=i%3;return a[r*3]*b[c]+a[r*3+1]*b[c+3]+a[r*3+2]*b[c+6];});}
const BOARD_TO_NORMAL=[8,0,32,0,-8,288,0,0,1];
const NORMAL_CORNERS:Point[]=[[32,288],[288,288],[288,32],[32,32]];
export function rectifyForTracking(image:Gray,boardToImage:Matrix):Gray {
 const width=320,height=320,pixels=new Uint8Array(width*height),map=multiply(boardToImage,inverse(BOARD_TO_NORMAL));
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){const [u,v]=project(map,[x,y]),ix=Math.floor(u),iy=Math.floor(v);if(ix<0||iy<0||ix>=image.width-1||iy>=image.height-1)continue;const dx=u-ix,dy=v-iy,i=iy*image.width+ix;pixels[y*width+x]=(1-dy)*((1-dx)*image.pixels[i]+dx*image.pixels[i+1])+dy*((1-dx)*image.pixels[i+image.width]+dx*image.pixels[i+image.width+1]);}
 return {width,height,pixels};
}
export function distributedBorderSupport(points:Point[]):boolean {
 if(points.length<12)return false;
 const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);
 if(Math.max(...xs)-Math.min(...xs)<192||Math.max(...ys)-Math.min(...ys)<192)return false;
 if(new Set(points.map(p=>`${p[0]>=160?1:0}${p[1]>=160?1:0}`)).size<3)return false;
 // A missing corner is acceptable only when a broad two-dimensional hull remains.
 const sorted=[...points].sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 const cross=(a:Point,b:Point,c:Point)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
 const half=(ps:Point[])=>{const hull:Point[]=[];for(const p of ps){while(hull.length>=2&&cross(hull[hull.length-2],hull[hull.length-1],p)<=0)hull.pop();hull.push(p);}return hull;};
 const lower=half(sorted),upper=half([...sorted].reverse()),hull=[...lower.slice(0,-1),...upper.slice(0,-1)];
 const area=Math.abs(hull.reduce((sum,p,i)=>{const q=hull[(i+1)%hull.length];return sum+p[0]*q[1]-p[1]*q[0];},0))/2;
 return area>=.35*256*256;
}
export class BorderTracker {
 private reference:Gray[];private points:Point[];private initial:Matrix;private boardToImage:Matrix;private last:number;private dead=false;private misses=0;private updates=0;private dimensions:Point;private priorCorners:Point[]|null=null;private interval=100;
 constructor(image:Gray,corners:Point[],timestamp:number){this.initial=homography(corners);this.boardToImage=this.initial;this.reference=pyramid(rectifyForTracking(image,this.initial));this.points=features(this.reference[0],NORMAL_CORNERS);this.last=timestamp;this.dimensions=[image.width,image.height];if(this.points.length<16)this.dead=true;}
 update(image:Gray,timestamp:number):TrackResult {
 const lost=(reason:string):TrackResult=>{this.dead=true;return {state:'lost',reason,inliers:0,features:this.points.length};};
 if(this.dead)return lost('Not enough reliable border features, or tracking already lost.');
 if(image.width!==this.dimensions[0]||image.height!==this.dimensions[1])return lost('Camera dimensions changed.');
 if(timestamp-this.last>700||timestamp<=this.last)return lost('Frame gap too large.');
 const uncertain=(reason:string):TrackResult=>{this.misses++;if(this.misses>=3)return lost(reason);return {state:'degraded',reason,inliers:0,features:this.points.length};};
 // Predict the changing viewpoint before rectification, especially through diagonal views.
 const boardPoints:Point[]=[[0,0],[32,0],[32,32],[0,32]],oldCorners=boardPoints.map(p=>project(this.boardToImage,p));
 let predicted=this.boardToImage;
 if(this.priorCorners){const factor=Math.min(1.5,(timestamp-this.last)/this.interval)*.8;try{predicted=homography(oldCorners.map((p,i)=>[p[0]+factor*(p[0]-this.priorCorners![i][0]),p[1]+factor*(p[1]-this.priorCorners![i][1])]));}catch{/* Fall back to the last verified view. */}}
 let accepted:{result:NonNullable<ReturnType<typeof robustTransform>>;candidate:Matrix;corners:Point[]}|null=null,matched=0;
 for(const estimate of predicted===this.boardToImage?[predicted]:[predicted,this.boardToImage]){
 const current=pyramid(rectifyForTracking(image,estimate)),from:Point[]=[],to:Point[]=[];
 for(const p of this.points){const forward=match(this.reference,current,p,p);if(!forward)continue;const backward=match(current,this.reference,forward.point,p);if(!backward||Math.hypot(backward.point[0]-p[0],backward.point[1]-p[1])>1.5)continue;from.push(p);to.push(forward.point);}
 matched=Math.max(matched,from.length);const result=robustTransform(from,to);if(!result||!distributedBorderSupport(result.indices.map(i=>from[i])))continue;
 const candidate=multiply(multiply(multiply(estimate,inverse(BOARD_TO_NORMAL)),result.matrix),BOARD_TO_NORMAL),corners=boardPoints.map(p=>project(candidate,p));
 accepted={result,candidate,corners};break;
 }
 if(!accepted)return uncertain(`Border matches need broad board-plane coverage: ${matched} / ${this.points.length}. Hold still while checking again.`);
 const {result,candidate,corners}=accepted;
 try{homography(corners);}catch{return lost('Invalid projected board.');}
 if(corners.some(p=>p[0]<0||p[1]<0||p[0]>=image.width||p[1]>=image.height))return lost('Playing area left the camera view.');
 // Refuse implausible jumps; recognition must verify a new orientation after loss.
 if(corners.some((p,i)=>{const old=project(this.boardToImage,([[0,0],[32,0],[32,32],[0,32]] as Point[])[i]);return Math.hypot(p[0]-old[0],p[1]-old[1])>Math.max(image.width,image.height)*.12;}))return uncertain('View changed too quickly.');
 this.priorCorners=oldCorners;this.interval=timestamp-this.last;this.boardToImage=candidate;this.last=timestamp;this.misses=0;this.updates++;
 this.reference=pyramid(rectifyForTracking(image,candidate));
 // Refresh from the current flat border, keeping every point in the same physical board coordinates.
 if(this.updates%4===0){const fresh=features(this.reference[0],NORMAL_CORNERS);if(fresh.length>=16)this.points=fresh;}
 return {state:'tracking',reason:'Perspective-aware board alignment verified',corners,transform:multiply(candidate,inverse(this.initial)),inliers:result.indices.length,features:this.points.length,residual:result.residual};
 }
}
export const IDENTITY=identity;
