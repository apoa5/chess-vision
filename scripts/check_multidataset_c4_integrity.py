"""Read-only protected-history and original-video hash check; no holdout access."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.multidataset import digest

def check():
    expected=json.loads(Path('outputs/research/input_c1_audit_v1/preserved_inputs.json').read_text())['protected_sha256']
    expected=dict(expected)
    for index in ('input_representation_c1','chessqueries_c2','crop_representation_c3'):
        p=Path('data/metadata')/(index+'_index_v1.json')
        expected.update(json.loads(p.read_text())['artifacts'])
        expected[str(p)]=digest(p)
    for s in json.loads(Path('outputs/scanning/d3c_interim_v1/registration.json').read_text())['sessions']:
        expected[s['source']['path']]=s['source']['sha256']
    for path in expected:
        if 'holdout' in path.lower():raise ValueError('Refuse to open final-holdout path')
    failures=[p for p,h in expected.items() if not Path(p).is_file() or digest(p)!=h]
    if failures:raise ValueError('Protected artifact mismatch: '+str(failures[:10]))
    return dict(status='passed',protected_files_checked=len(expected),mismatches=[],final_holdout_accessed=False,
                scope='frozen C1 protected map plus C1/C2/C3 artifact maps, indexes and ten original D3c videos')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();result=check()
    if a.output:
        if a.output.exists():raise ValueError('Fresh check output required')
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
