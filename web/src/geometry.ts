export type Point = [number, number];
export type Matrix = number[];
// Semantic order: a1 outer corner, h1 outer corner, h8 outer corner, a8 outer corner.
export function homography(corners: Point[]): Matrix {
  if(corners.length !== 4 || corners.some(p => p.some(v => !Number.isFinite(v)))) throw Error('Select four valid corners');
  const cross = corners.map((p,i) => {const q=corners[(i+1)%4],r=corners[(i+2)%4];return (q[0]-p[0])*(r[1]-q[1])-(q[1]-p[1])*(r[0]-q[0]);});
  // Canonical x points right, y toward rank 8; an unmirrored camera reverses winding.
  if(cross.some(v=>v >= -1)) throw Error('Corners are folded, mirrored or too close. Check their semantic order.');
  const board:Point[]=[[0,0],[32,0],[32,32],[0,32]];
  const rows:number[][]=[];
  board.forEach(([x,y],i)=>{const [u,v]=corners[i]; rows.push([x,y,1,0,0,0,-u*x,-u*y,u],[0,0,0,x,y,1,-v*x,-v*y,v]);});
  for(let k=0;k<8;k++){let pivot=k;for(let j=k+1;j<8;j++)if(Math.abs(rows[j][k])>Math.abs(rows[pivot][k]))pivot=j;
    [rows[k],rows[pivot]]=[rows[pivot],rows[k]];const scale=rows[k][k];if(Math.abs(scale)<1e-9)throw Error('Degenerate alignment');
    rows[k]=rows[k].map(v=>v/scale);for(let j=0;j<8;j++)if(j!==k){const m=rows[j][k];rows[j]=rows[j].map((v,i)=>v-m*rows[k][i]);}}
  const h=[...rows.map(r=>r[8]),1];
  const denominators=board.map(([x,y])=>h[6]*x+h[7]*y+1);
  if(denominators.some(v=>v<=1e-8))throw Error('Board crosses the projective horizon');
  return h;
}
export function project(h:Matrix,[x,y]:Point):Point {const d=h[6]*x+h[7]*y+h[8];if(Math.abs(d)<1e-9)throw Error('Point at infinity');return [(h[0]*x+h[1]*y+h[2])/d,(h[3]*x+h[4]*y+h[5])/d];}
export function squarePolygon(h:Matrix,file:number,rank:number):Point[]{return [[file*4,rank*4],[(file+1)*4,rank*4],[(file+1)*4,(rank+1)*4],[file*4,(rank+1)*4]].map(p=>project(h,p as Point));}
export function contains(poly:Point[],p:Point):boolean {let inside=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j];if((a[1]>p[1])!==(b[1]>p[1])&&p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])inside=!inside;}return inside;}
export function displayPoint(p:Point,source:Point,display:Point):Point {return [p[0]*display[0]/source[0],p[1]*display[1]/source[1]];}
