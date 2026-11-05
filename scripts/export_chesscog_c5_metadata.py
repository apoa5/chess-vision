"""Stream only the prepared metadata to gzip; never downloads source images."""
import gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal
from src.training.multidataset import board_ready,digest,split_audit
from src.chess.labels import CLASS_NAMES

if __name__=='__main__':
 root=Path('outputs/evaluation/chesscog_c5');out=root/'prepared_manifest_v1.json.gz'
 if out.exists():raise ValueError('Refuse metadata overwrite')
 expected=json.loads((root/'prepare-chesscog_result_v1.json').read_text())['dataset_fingerprint']
 volume=modal.Volume.from_name('chess-vision-c4-data-v1');sha=hashlib.sha256();n=0
 with out.open('xb') as dest:
  with gzip.GzipFile(filename='',fileobj=dest,mode='wb',mtime=0) as zipped:
   for chunk in volume.read_file('/chesscog-c5-v1/manifest.json'):
    n+=len(chunk)
    if n>100_000_000:raise ValueError('Metadata exceeds100MB limit')
    sha.update(chunk);zipped.write(chunk)
 if sha.hexdigest()!=expected:raise ValueError('Cloud manifest SHA mismatch')
 with gzip.open(out,'rt') as f:m=json.load(f)
 audit=split_audit(m['boards']);counts={}
 for split in ('train','val'):
  bs=[b for b in m['boards'] if b['split']==split and board_ready(b)];classes=[0]*13
  for b in bs:
   for o in b['observations']:classes[o['class_id']]+=1
  counts[split]=dict(boards=len(bs),squares=sum(classes),unique_placements=len({b['position_id'] for b in bs}),class_counts=classes)
 result=dict(schema_version='chesscog_manifest_export_c5_v1',raw_manifest_sha256=sha.hexdigest(),raw_bytes=n,gzip_sha256=digest(out),gzip_bytes=out.stat().st_size,
             cloud_volume='chess-vision-c4-data-v1',cloud_path='/chesscog-c5-v1/manifest.json',grouped_split_audit=audit,class_names=list(CLASS_NAMES),counts=counts)
 (root/'manifest_audit_v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
