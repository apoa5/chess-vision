"""Recheck C7's pre-execution protected hashes without opening final holdout."""
import hashlib
import json
from pathlib import Path
ROOT=Path('outputs/evaluation/personal_c7')

def check():
    expected=json.loads((ROOT/'preserved_inputs_v1.json').read_text())['protected_sha256']
    mismatches=[]
    for path,sha in expected.items():
        if 'holdout' in path.lower():raise ValueError('Holdout path prohibited')
        p=Path(path)
        if not p.is_file():mismatches.append({'path':path,'reason':'missing'});continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=sha:mismatches.append({'path':path,'reason':'changed'})
    result={'checked':len(expected),'mismatches':mismatches,'final_holdout_access':False}
    target=ROOT/'integrity_v1.json'
    if target.exists():raise ValueError('Fresh integrity output required')
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if mismatches:raise RuntimeError('Protected inputs changed')

if __name__=='__main__':check()
