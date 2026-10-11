import {it} from 'node:test';
import assert from 'node:assert/strict';
import {crc32,sessionZip,MAX_BYTES,MAX_FRAMES} from '../src/recording.ts';
it('uses the standard ZIP checksum',()=>assert.equal(crc32(new TextEncoder().encode('123456789')),0xcbf43926));
it('writes ZIP headers with bounded capture policy',async()=>{const data=new Uint8Array(await sessionZip([{name:'manifest.json',data:new TextEncoder().encode('{}')}]).arrayBuffer());assert.equal(new DataView(data.buffer).getUint32(0,true),0x04034b50);assert.equal(new DataView(data.buffer).getUint32(data.length-22,true),0x06054b50);assert.equal(MAX_FRAMES,60);assert.equal(MAX_BYTES,80*1024*1024);});
it('rejects paths outside the archive',()=>assert.throws(()=>sessionZip([{name:'../key.pem',data:new Uint8Array()}])));
