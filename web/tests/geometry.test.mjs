import {describe,it} from 'node:test';
import assert from 'node:assert/strict';
const expect=v=>({toBe:w=>assert.equal(v,w),toEqual:w=>assert.deepEqual(v,w),toBeCloseTo:w=>assert.ok(Math.abs(v-w)<0.005),toBeGreaterThan:w=>assert.ok(v>w),toThrow:()=>assert.throws(v)});
import {homography,project,squarePolygon,contains,displayPoint} from '../src/geometry.ts';
describe('semantic board mapping',()=>{
 for(const angle of [0,90,180,270,37,143])it(`preserves square identity at ${angle} degrees`,()=>{
 const a=angle*Math.PI/180; const map=([x,y])=>[400+8*(x*Math.cos(a)+y*Math.sin(a)),400+8*(x*Math.sin(a)-y*Math.cos(a))];
 const corners=([[0,0],[32,0],[32,32],[0,32]]).map(map);const h=homography(corners);
 for(let f=0;f<8;f++)for(let r=0;r<8;r++){const center=map([4*f+2,4*r+2]);expect(contains(squarePolygon(h,f,r),center)).toBe(true);const actual=project(h,[4*f+2,4*r+2]);expect(actual[0]).toBeCloseTo(center[0]);expect(actual[1]).toBeCloseTo(center[1]);}
 });
 it('handles perspective and preserves the full border',()=>{const c=[[40,500],[600,450],[470,70],[140,100]];const h=homography(c);([[0,0],[32,0],[32,32],[0,32]]).forEach((p,i)=>{const q=project(h,p);expect(q[0]).toBeCloseTo(c[i][0]);expect(q[1]).toBeCloseTo(c[i][1]);});expect(project(h,[-3.5,-3.5])[1]).toBeGreaterThan(500);});
 it('rejects mirrored, folded and collapsed geometry',()=>{for(const c of [[[0,0],[100,0],[100,100],[0,100]],[[0,100],[100,0],[100,100],[0,0]],[[0,0],[0,0],[0,0],[0,0]]])expect(()=>homography(c)).toThrow();});
 it('maps source to CSS pixels without device-pixel scaling',()=>{expect(displayPoint([960,540],[1920,1080],[360,202.5])).toEqual([180,101.25]);});
});
