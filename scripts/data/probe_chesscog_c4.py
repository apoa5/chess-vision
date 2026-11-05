"""Bounded official Chesscog multipart-ZIP inventory and sample extraction.

Reads central directories, not whole archives. CRC-32 validates sample members;
full archive hashes are not claimed verified. Never extracts arbitrary paths.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zlib


def bounded_range(url,start,end):
    request=urllib.request.Request(url+('?c4_range='+str(start)+'_'+str(end)),headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=45) as response:
        expected=f'bytes {start}-{end}/'
        if response.status!=206 or not response.headers.get('Content-Range','').startswith(expected):
            raise ValueError('Exact range unavailable; refuse full archive download')
        data=response.read(end-start+2)
        if len(data)!=end-start+1:raise ValueError('Range length mismatch')
        return data


def inventory(file):
    size=file['attributes']['size'];url=file['links']['download'];tail=bounded_range(url,size-65536,size-1)
    index=tail.rfind(b'PK\x05\x06')
    if index<0:raise ValueError('No bounded ZIP end record')
    _,disk,cd_disk,on_disk,total,cd_size,cd_offset,comment=struct.unpack('<4s4H2IH',tail[index:index+22])
    if cd_size>5_000_000:raise ValueError('Central directory exceeds probe budget')
    data=bounded_range(url,cd_offset,cd_offset+cd_size-1)
    rows=[];offset=0
    while offset<len(data):
        values=struct.unpack('<4s6H3I5H2I',data[offset:offset+46])
        if values[0]!=b'PK\x01\x02':raise ValueError('Invalid central directory')
        name_len,extra_len,comment_len=values[10:13]
        name=data[offset+46:offset+46+name_len].decode('utf-8')
        rows.append(dict(name=name,method=values[4],crc32=values[7],compressed_bytes=values[8],bytes=values[9],disk=values[13],local_offset=values[16]))
        offset+=46+name_len+extra_len+comment_len
    if len(rows)!=total:raise ValueError('ZIP member count mismatch')
    return dict(name=file['attributes']['name'],archive_bytes=size,zip_disk=disk,member_count=total,
                extracted_bytes=sum(r['bytes'] for r in rows),rows=rows)


def member(record,parts,read_range=bounded_range):
    disk=record['disk'];url,size=parts[disk];start=record['local_offset']
    header=read_range(url,start,start+29)
    values=struct.unpack('<4s5H3I2H',header)
    if values[0]!=b'PK\x03\x04':raise ValueError('Local ZIP header mismatch')
    offset=start+30+values[-2]+values[-1];remaining=record['compressed_bytes'];chunks=[]
    if remaining>5_000_000 or record['bytes']>15_000_000:raise ValueError('Sample exceeds bounded member size')
    while remaining:
        amount=min(remaining,size-offset)
        chunks.append(read_range(url,offset,offset+amount-1));remaining-=amount
        if remaining:disk+=1;url,size=parts[disk];offset=0
    compressed=b''.join(chunks)
    if record['method']==8:data=zlib.decompress(compressed,-15)
    elif record['method']==0:data=compressed
    else:raise ValueError('Unsupported ZIP method')
    if len(data)!=record['bytes'] or zlib.crc32(data)!=record['crc32']:raise ValueError('Member CRC/length mismatch')
    return data


def run(root,samples):
    if (root/'inventory.json').exists():raise ValueError('Fresh probe output required')
    raw=json.loads(Path('outputs/evaluation/multidataset_c4/probes/osf_root.txt').read_text())
    files={r['attributes']['name']:r for r in raw['data']};root.mkdir(parents=True,exist_ok=True)
    inventories=[];downloaded=[]
    for split in ('train','val','test'):
        file=files[split+'.zip'];inv=inventory(file);inventories.append(inv)
        inv['png_count']=sum(r['name'].endswith('.png') for r in inv['rows']);inv['json_count']=sum(r['name'].endswith('.json') for r in inv['rows'])
        (root/(split+'_inventory.json')).write_text(json.dumps(inv,indent=2)+'\n')
        print(split,inv['png_count'],'images; extracted bytes',inv['extracted_bytes'],flush=True)
        if split=='test':continue
        parts=[(files['train.z01']['links']['download'],files['train.z01']['attributes']['size']),(file['links']['download'],file['attributes']['size'])] if split=='train' else [(file['links']['download'],file['attributes']['size'])]
        entries={r['name']:r for r in inv['rows']}
        annotations=sorted((r for r in inv['rows'] if r['name'].endswith('.json') and r['name'][:-5]+'.png' in entries),key=lambda r:r['name'])
        if not annotations:raise ValueError('No same-stem image/annotation pairs found')
        for record in annotations[:samples]:
            png_name=record['name'][:-5]+'.png'
            if png_name not in entries:raise ValueError('Missing paired image')
            for current in (record,entries[png_name]):
                data=member(current,parts);path=root/split/Path(current['name']).name;path.parent.mkdir(exist_ok=True);path.write_bytes(data)
                downloaded.append(dict(path=str(path),source_member=current['name'],split=split,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),crc32_verified=True))
                print('Sample',split,path.name,len(data),flush=True)
    result=dict(schema_version='chesscog_bounded_inventory_c4_v1',archives=inventories,sample_members=downloaded,
                full_archive_sha_verified=False,total_archive_bytes=sum(f['attributes']['size'] for f in files.values()),no_full_archive_download=True)
    (root/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--samples',type=int,default=2)
    a=parser.parse_args()
    if not 1<=a.samples<=4:parser.error('Bounded probe permits 1..4 pairs per training/validation split')
    run(a.output,a.samples)
