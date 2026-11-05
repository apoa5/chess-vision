"""Check recorded historical files and inventoried originals, never discover holdout files."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.multidataset import digest
from src.training.manifest_tools import write_new

ROOT=Path('outputs/evaluation/convnext_c9')

def main(output):
    historical=json.loads((ROOT/'protected_inputs_v1.json').read_text())['protected_sha256']
    inventory=json.loads((ROOT/'inventory_v1.json').read_text())
    media={p['source_path']:p['sha256'] for p in inventory['photos']}
    media.update({s['source']['path']:s['source']['sha256'] for s in inventory['sessions']})
    mismatch=[]
    for kind,expected in [('historical',historical),('original_media',media)]:
        for path,sha in expected.items():
            if 'holdout' in path.lower():raise ValueError('Holdout path prohibited')
            if not Path(path).is_file() or digest(path)!=sha:mismatch.append(dict(kind=kind,path=path))
    result=dict(historical_files_checked=len(historical),original_media_checked=len(media),mismatches=mismatch,final_holdout_access=False)
    write_new(output,result);print(json.dumps(result))
    if mismatch:raise RuntimeError('Protected artifact mismatch')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
