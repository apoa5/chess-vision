export const SESSION_VERSION = 'guided_capture_v1';
export const MAX_FRAMES = 60;
export const MAX_BYTES = 80 * 1024 * 1024;
export type CapturedFrame = {
  frame_id: string; path: string; timestamp_ms: number; width: number; height: number;
  source_sha256: string; board_epoch: number; view_group: string;
  geometry: null | {h_board_to_frame: number[]; corners: number[][]; source: 'manual'; version: number; square_polygons?: Record<string,number[][]>};
  visibility: 'unknown'; source: 'camera' | 'demo';
};
export type CaptureEvent = {timestamp_ms: number; type: string; board_epoch: number};
export function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const b of bytes) {c ^= b; for(let i=0;i<8;i++) c = (c >>> 1) ^ ((c & 1) ? 0xedb88320 : 0);}
  return (c ^ 0xffffffff) >>> 0;
}
// Stored ZIP: no compression dependencies, compatible with Python zipfile.
export function sessionZip(entries: {name: string; data: Uint8Array}[]): Blob {
  const parts: BlobPart[] = [], central: BlobPart[] = []; let offset=0, centralLength=0;
  for(const {name,data} of entries){
    if(!/^[a-zA-Z0-9_./-]+$/.test(name)||name.includes('..'))throw Error('Unsafe archive path');
    const filename=new TextEncoder().encode(name), crc=crc32(data);
    const local=new Uint8Array(30+filename.length),v=new DataView(local.buffer);
    v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint32(14,crc,true);v.setUint32(18,data.length,true);v.setUint32(22,data.length,true);v.setUint16(26,filename.length,true);local.set(filename,30);
    parts.push(local, new Uint8Array(data).buffer);
    const directory=new Uint8Array(46+filename.length),d=new DataView(directory.buffer);
    d.setUint32(0,0x02014b50,true);d.setUint16(4,20,true);d.setUint16(6,20,true);d.setUint32(16,crc,true);d.setUint32(20,data.length,true);d.setUint32(24,data.length,true);d.setUint16(28,filename.length,true);d.setUint32(42,offset,true);directory.set(filename,46);
    central.push(directory);centralLength+=directory.length;offset+=local.length+data.length;
  }
  const end=new Uint8Array(22),v=new DataView(end.buffer);v.setUint32(0,0x06054b50,true);v.setUint16(8,entries.length,true);v.setUint16(10,entries.length,true);v.setUint32(12,centralLength,true);v.setUint32(16,offset,true);
  return new Blob([...parts,...central,end],{type:'application/zip'});
}
export async function imageBytes(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  const blob=await new Promise<Blob>((resolve,reject)=>canvas.toBlob(b=>b?resolve(b):reject(Error('Image encoding failed')),'image/jpeg',0.95));
  return new Uint8Array(await blob.arrayBuffer());
}
export async function sha256(data: Uint8Array): Promise<string> {
  const hash=await crypto.subtle.digest('SHA-256',new Uint8Array(data).buffer);
  return Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,'0')).join('');
}
