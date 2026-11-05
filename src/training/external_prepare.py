"""Explicit future full Chesscog preparation; never called by a smoke run.

Streams three official archive parts to a cache, validates official hashes when
provided, reads multipart members without shell unzip/path traversal, and keeps
only train/val images. Never downloads or extracts the external test archive.
"""
import hashlib,json,shutil,urllib.request
from pathlib import Path

def prepare_chesscog(root):
    from scripts.data.probe_chesscog_c4 import inventory,member
    from src.training.multidataset import chesscog_board,quarantine_protected,split_audit,position_key,PROFILE,digest
    from src.chess.labels import CLASS_NAMES
    root=Path(root)
    if (root/'manifest.json').exists():raise ValueError('Prepared dataset exists; use it rather than overwrite')
    root.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://api.osf.io/v2/nodes/xf3ka/files/osfstorage/?page[size]=100',timeout=45) as f:listing=json.load(f)
    files={f['attributes']['name']:f for f in listing['data']};required=('train.z01','train.zip','val.zip')
    expected=sum(files[n]['attributes']['size'] for n in required)
    if shutil.disk_usage(root).free<expected*2+500_000_000:raise ValueError('Insufficient disk for archive + extracted preparation')
    inv={n:inventory(files[n]) for n in ('train.zip','val.zip')};receipts=[]
    for name in required:
        meta=files[name];p=root/'archives'/name;p.parent.mkdir(exist_ok=True)
        if p.exists():raise ValueError('Existing partial/full archive requires explicit recovery review')
        md5=hashlib.md5();sha=hashlib.sha256();n=0
        with urllib.request.urlopen(meta['links']['download'],timeout=120) as source,p.open('xb') as dest:
            for chunk in iter(lambda:source.read(4*1024*1024),b''):
                n+=len(chunk)
                if n>meta['attributes']['size']:raise ValueError('Download exceeded published size')
                dest.write(chunk);md5.update(chunk);sha.update(chunk)
        if n!=meta['attributes']['size']:raise ValueError('Truncated archive')
        official=meta['attributes'].get('extra',{}).get('hashes',{}).get('md5')
        if official and official!=md5.hexdigest():raise ValueError('Official archive checksum mismatch')
        receipts.append(dict(name=name,bytes=n,sha256=sha.hexdigest(),md5=md5.hexdigest(),official_md5_verified=bool(official)))
    def local_range(path,start,end):
        with open(path,'rb') as f:f.seek(start);b=f.read(end-start+1)
        if len(b)!=end-start+1:raise ValueError('Truncated archive range')
        return b
    boards=[]
    for split in ('train','val'):
        rows={r['name']:r for r in inv[split+'.zip']['rows'] if r['name'].startswith(split+'/') and '__MACOSX' not in r['name']}
        parts=[(str(root/'archives'/n),files[n]['attributes']['size']) for n in (('train.z01','train.zip') if split=='train' else ('val.zip',))]
        for name in sorted(rows):
            if not name.endswith('.json') or name[:-5]+'.png' not in rows:continue
            dest=root/'images'/split;dest.mkdir(parents=True,exist_ok=True)
            for key in (name,name[:-5]+'.png'):
                (dest/Path(key).name).write_bytes(member(rows[key],parts,read_range=local_range))
            b=chesscog_board(dest/Path(name[:-5]+'.png').name,dest/Path(name).name,split)
            boards.append(b)
    protected=json.loads((root/'personal_protected_placements.json').read_text())
    protected.update({position_key(b['placement']):dict(source='official_chesscog_val') for b in boards if b['split']=='val'})
    quarantine_protected(boards,protected);audit=split_audit(boards)
    for b in boards:
        b['source']['path']=str(Path(b['source']['path']).relative_to(root))
        b['annotation']['path']=str(Path(b['annotation']['path']).relative_to(root))
    result=dict(schema_version='multidataset_c4_v1',representation=PROFILE,class_names=list(CLASS_NAMES),final_holdout=False,boards=boards,
                grouped_split_audit=audit,archive_receipts=receipts,external_geometry_approximation='b2-annotated-corners-regular-grid-v1')
    (root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    # Archives remain cached intentionally; no automatic destructive cleanup.
    return dict(boards=len(boards),included_train=sum(b['split']=='train' and b['inclusion']['status']=='included' for b in boards),archive_bytes=expected)
