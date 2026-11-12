"""Preserve historical inputs while allowing explicitly scoped MVP-1.6 edits."""
import argparse,json
from pathlib import Path
from check_assisted_mvp1 import check_snapshot
ALLOWED={'web/src/AssistedPhotoApp.tsx','web/src/MultiViewScan.tsx','web/src/scanCapture.ts','web/src/style.css','web/package.json','web/dist/index.html','src/scanning/assisted_multiview.py'}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/evaluation/assisted_mvp16_v1/integrity_v1.json'));a=p.parse_args()
 x=json.loads(Path('outputs/evaluation/assisted_mvp16_v1/preserved_inputs_v1.json').read_text())
 if set(x['authorized_application_edits'])!=ALLOWED:raise ValueError('Unexpected preservation scope')
 result=check_snapshot(x['sha256'],ALLOWED);result['schema_version']='assisted_mvp16_preservation_v1'
 with a.output.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='authorized_application_changes'}))
 if result['unexpected_changes'] or result['missing']:raise SystemExit(1)
if __name__=='__main__':main()
