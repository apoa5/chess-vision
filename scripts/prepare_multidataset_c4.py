"""C.4 local provenance manifests; never starts training."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.manifest_tools import personal_manifest,retire_d3c,smoke_bundle,write_new
from src.training.multidataset import chesscog_board,quarantine_protected,split_audit,PROFILE
from src.training.manifest_tools import protected_personal
from src.chess.labels import CLASS_NAMES

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['personal','retire-d3c','smoke-bundle','chesscog']);p.add_argument('--output',type=Path,required=True);p.add_argument('--source',type=Path)
 a=p.parse_args()
 if a.action=='personal':
  m=personal_manifest();write_new(a.output,m);print('Personal boards',len(m['boards']))
 elif a.action=='retire-d3c':
  m=retire_d3c();write_new(a.output,m);print('Retired ten historical independent sessions; no verification status upgraded.')
 elif a.action=='smoke-bundle':print(smoke_bundle(json.loads(a.source.read_text()),a.output))
 else:
  boards=[]
  for split in ('train','val'):
   for path in sorted((a.source/split).glob('*.png')):boards.append(chesscog_board(path,path.with_suffix('.json'),split))
  boards=quarantine_protected(boards,protected_personal());split_audit(boards)
  write_new(a.output,dict(schema_version='multidataset_c4_v1',representation=PROFILE,class_names=list(CLASS_NAMES),final_holdout=False,boards=boards,
    geometry_note='Exact B2 crop/mask/normalization operators; external GT regular grid is explicitly versioned, not claimed identical to historical fitted-grid pixels.'))
  print('Chesscog actual verified sample boards',len(boards))
