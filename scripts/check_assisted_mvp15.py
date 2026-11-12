"""Check preserved historical inputs, permitting only listed MVP application edits."""
import argparse,json
from pathlib import Path
from check_assisted_mvp1 import check_snapshot
ALLOWED={
 'scripts/serve_scan_web.py','web/src/DigitalBoard.tsx',
 'web/src/style.css','web/package.json','web/dist/index.html',
 'web/src/AssistedPhotoApp.tsx','web/src/photoReview.ts',
}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--snapshot',type=Path,default=Path('outputs/evaluation/assisted_mvp15_v1/preserved_inputs_v1.json'))
 p.add_argument('--output',type=Path,default=Path('outputs/evaluation/assisted_mvp15_v1/integrity_v1.json'))
 a=p.parse_args();snapshot=json.loads(a.snapshot.read_text())
 if set(snapshot['authorized_application_edits'])!=ALLOWED:raise ValueError('Unexpected exception list')
 result=check_snapshot(snapshot['sha256'],ALLOWED);result['schema_version']='assisted_mvp15_preservation_v1'
 with a.output.open('x') as stream:json.dump(result,stream,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='authorized_application_changes'}))
 if result['unexpected_changes'] or result['missing']:raise SystemExit(1)
if __name__=='__main__':main()
