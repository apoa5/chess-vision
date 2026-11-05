"""Read-only C5 checkpoint export and bounded CPU reload verification.

Optional wait observes the existing job; never launches/retries any training.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal

def main(wait_seconds):
    root=Path('outputs/evaluation/chesscog_c5');deadline=time.monotonic()+wait_seconds
    while True:
        try:result=json.loads((root/'train_result_v1.json').read_text());break
        except (FileNotFoundError,json.JSONDecodeError):
            if time.monotonic()>=deadline:raise ValueError('Completed training result unavailable; no retry launched')
            time.sleep(2)
    import torch
    from src.chess.labels import CLASS_NAMES
    from src.vision.classifier import create_classifier
    torch.set_num_threads(2)
    workspace=modal.Workspace.from_context();workspace.hydrate()
    if workspace.name!='akwasipoku-oforiagyei':raise ValueError('Wrong workspace')
    volume=modal.Volume.from_name('chess-vision-c4-artifacts-v1');folder=root/'checkpoints';folder.mkdir(exist_ok=True);exports={}
    names=['chesscog_best.pt','last.pt','authorized_config.json','metrics.jsonl','training_loss.json','validation.json','validation_logits.npz']
    for name in names:
        path=folder/name;sha=hashlib.sha256();size=0
        with path.open('xb') as dest:
            for chunk in volume.read_file('/chesscog-c5-seed42-v1/'+name):
                size+=len(chunk)
                if size>50_000_000:raise ValueError('Unexpectedly large export')
                dest.write(chunk);sha.update(chunk)
        exports[name]=dict(path=str(path),bytes=size,sha256=sha.hexdigest())
        if name.endswith('.pt'):
            if sha.hexdigest()!=result['checkpoint_hashes'][name]:raise ValueError('Checkpoint transfer hash mismatch')
            state=torch.load(path,map_location='cpu',weights_only=True)
            if (state.get('smoke_only') or state['stage']!='chesscog' or state['class_names']!=list(CLASS_NAMES)
                or state['input_channels']!=4 or state['dataset_fingerprint']!=result['dataset_fingerprint'] or state['plan_hash']!=result['plan_hash']):raise ValueError('Checkpoint provenance mismatch')
            model=create_classifier(pretrained=False,mode='fine_tune');model.load_state_dict(state['model_state_dict'],strict=True);model.eval()
            with torch.inference_mode():scores=model(torch.zeros(1,4,224,224))
            if scores.shape!=(1,13) or not torch.isfinite(scores).all():raise ValueError('CPU shape/reload check failed')
            model_hash=hashlib.sha256()
            for key,value in sorted(state['model_state_dict'].items()):model_hash.update(key.encode());model_hash.update(value.contiguous().numpy().tobytes())
            exports[name].update(cpu_reload_verified=True,shape=[1,13],model_tensor_sha256=model_hash.hexdigest(),epoch=state['epoch'],next_batch=state['next_batch'],updates=state['updates'])
            del model,state
        print('Exported and checked',name,size,flush=True)
    summary=dict(schema_version='chesscog_checkpoint_exports_c5_v1',exports=exports,
                 best_and_latest_model_tensors_equal=exports['chesscog_best.pt']['model_tensor_sha256']==exports['last.pt']['model_tensor_sha256'],
                 cpu_check='arbitrary zero input for compatibility only, not an accuracy evaluation',production_assets_changed=False)
    out=root/'checkpoint_exports_v1.json'
    if out.exists():raise ValueError('Refuse export-summary overwrite')
    out.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-seconds',type=int,default=0);a=p.parse_args();main(a.wait_seconds)
