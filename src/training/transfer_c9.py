"""Bounded C9 archive extraction and shared-context integrity, stdlib except digest."""
import hashlib,json,tarfile
from pathlib import Path
from src.training.multidataset import digest


def extract_archive(archive,root,receipt,shared_root='/data/personal-c7-bundle-v2'):
    archive=Path(archive);root=Path(root)
    if digest(archive)!=receipt['archive_sha256'] or archive.stat().st_size!=receipt['archive_bytes']:raise ValueError('Transfer archive changed')
    if root.exists():raise ValueError('Fresh C9 data root required')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or len(names)>2000 or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members):raise ValueError('Unsafe archive members')
        if sum(m.size for m in members)>2_000_000_000:raise ValueError('Archive size cap')
        inventory_bytes=tar.extractfile('file_inventory.json').read()
        if hashlib.sha256(inventory_bytes).hexdigest()!=receipt['inventory_sha256']:raise ValueError('Archive inventory changed')
        inventory=json.loads(inventory_bytes)
        if set(names)!=set(inventory)|{'file_inventory.json'}:raise ValueError('Unexpected/missing archive member')
        root.mkdir(parents=True)
        for member in members:
            path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
            with tar.extractfile(member) as source,path.open('xb') as dest:
                for block in iter(lambda:source.read(1024*1024),b''):dest.write(block);h.update(block)
            if member.name in inventory and (path.stat().st_size!=inventory[member.name]['bytes'] or h.hexdigest()!=inventory[member.name]['sha256']):raise ValueError('Extracted context checksum mismatch')
    if digest(root/'manifest.json')!=receipt['manifest_sha256']:raise ValueError('Manifest changed')
    m=json.loads((root/'manifest.json').read_text());shared=Path(shared_root);old=shared/'bundle_inventory.json'
    if digest(old)!='966310c762a663e5483510b122cd103513559eb9e21581303128342fc077289d':raise ValueError('Protected C7 inventory changed')
    existing=json.loads(old.read_text());checked=set()
    for b in m['boards']:
        for filename in [b['source']['path'],b['geometry']['valid_mask_path']]:
            path=Path(filename)
            if not path.is_relative_to(root) and not path.is_relative_to(shared):raise ValueError('Context escapes registered storage')
            if path in checked:continue
            if path.is_relative_to(shared):
                key=str(path.relative_to(shared))
                if key not in existing or digest(path)!=existing[key]['sha256']:raise ValueError('Shared C7 context/mask changed')
            checked.add(path)
        if digest(b['source']['path'])!=b['source']['sha256']:raise ValueError('Runtime source changed')
    return dict(manifest_sha256=digest(root/'manifest.json'),boards=len(m['boards']),files_verified=len(checked),shared_C7_inventory_verified=True,original_video_uploads=0)
