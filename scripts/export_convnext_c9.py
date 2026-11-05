"""Read completed C9 Volume artifacts, checking checkpoint hashes; never starts compute."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal
from src.training.multidataset import digest
ROOT=Path('outputs/evaluation/convnext_c9')

def main(training_only=False):
    volume=modal.Volume.from_name('chess-vision-c4-artifacts-v1',create_if_missing=False)
    remote='/convnext-c9-seed42-v1/';out=ROOT/'export_v1';out.mkdir(exist_ok=True)
    def fetch(name):
        target=out/name
        if target.exists():return target
        partial=out/(name+'.partial')
        if partial.exists():raise ValueError('Prior incomplete transfer requires inspection: '+str(partial))
        with partial.open('xb') as f:
            for chunk in volume.read_file(remote+name):f.write(chunk)
        partial.rename(target);return target
    # The original worker committed training + selection before its client
    # authentication failure. Training-only recovery performs no evaluation.
    result=json.loads(fetch('training_result.json' if training_only else 'c9_result.json').read_text())
    files=['training_result.json','authorized_config.json','history.jsonl','selection_lock.json','one_attempt_claim.json','authorized_continuation_v1.json','internal_test_one_attempt.json',
        'convnextv2_tiny_best.pt','last.pt']
    if training_only:
        files.remove('internal_test_one_attempt.json')
        files += ['validation_epoch_%02d.json'%h['epoch'] for h in result['history']]
    else:files += ['convnext_val.json','b2_val.json','convnext_internal_test.json','b2_internal_test.json']
    for name in files:fetch(name)
    expected=result['checkpoint_hashes']
    for name in ('convnextv2_tiny_best.pt','last.pt'):
        if digest(out/name)!=expected[name]:raise ValueError('Export checkpoint hash mismatch: '+name)
    receipt=dict(schema_version='convnext_c9_export_v1',training_only=training_only,remote_volume='chess-vision-c4-artifacts-v1',remote_root=remote,files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()})
    target=ROOT/'export_receipt_v1.json'
    if target.exists():raise ValueError('Already exported')
    target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt['files'][k] for k in expected},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-only',action='store_true');a=p.parse_args();main(a.training_only)
