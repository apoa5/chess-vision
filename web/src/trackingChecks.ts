import {homography,project} from './geometry.ts';
import {inverse} from './tracking.ts';
import type {Point} from './geometry.ts';
export type VerifiedGeometry={corners:Point[];polygons:Record<string,Point[]>};
// Recovery requires independent semantic evidence and geometry consistent with all 64 identities.
export function verifiedMarkingGeometry(value:unknown,width:number,height:number):VerifiedGeometry|null {
 try{
 const r=value as {status:string;orientation_source:string;learned_marking_evidence:{inliers:number;distinct_files:number;distinct_ranks:number};geometry:{corners:Point[]};square_polygons:Record<string,Point[]>};
 if(r.status!=='candidate'||r.orientation_source!=='learned_border_markings'||r.learned_marking_evidence.inliers<8||r.learned_marking_evidence.distinct_files<3||r.learned_marking_evidence.distinct_ranks<3)return null;
 const corners=r.geometry.corners;if(corners.length!==4||corners.some(p=>p.length!==2||p.some(v=>!Number.isFinite(v))||p[0]<0||p[1]<0||p[0]>=width||p[1]>=height))return null;
 const inv=inverse(homography(corners)),polygons=r.square_polygons;if(Object.keys(polygons).length!==64)return null;
 for(let rank=0;rank<8;rank++)for(let file=0;file<8;file++){
 const id=String.fromCharCode(97+file)+(rank+1),poly=polygons[id];if(poly?.length!==4||poly.some(p=>p.length!==2||p.some(v=>!Number.isFinite(v))))return null;
 const mapped=poly.map(p=>project(inv,p)),expected:Point[]=[[file*4,rank*4],[(file+1)*4,rank*4],[(file+1)*4,(rank+1)*4],[file*4,(rank+1)*4]];
 let best=Infinity;for(const order of [mapped,[...mapped].reverse()])for(let shift=0;shift<4;shift++)best=Math.min(best,Math.max(...expected.map((p,i)=>Math.hypot(p[0]-order[(i+shift)%4][0],p[1]-order[(i+shift)%4][1]))));if(best>.8)return null;
 }
 return {corners,polygons};
 }catch{return null;}
}
export function markingDisagrees(expected:Point[],found:Point[]):boolean {const square=Math.min(...expected.map((p,i)=>Math.hypot(p[0]-expected[(i+1)%4][0],p[1]-expected[(i+1)%4][1])))/8;return found.some((p,i)=>Math.hypot(p[0]-expected[i][0],p[1]-expected[i][1])>square*.35);}
