"""C.7 one authorized L4 run per branch; no retries, chaining or resumption."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modal
from scripts.modal_multidataset_c4 import image, data, artifacts

app = modal.App('chess-vision-c7')
ROOT = Path('outputs/evaluation/personal_c7')

@app.function(image=image, gpu='L4', cpu=(2,2), memory=8192, timeout=2400,
              startup_timeout=300, max_containers=1, min_containers=0, retries=0,
              scaledown_window=2, volumes={'/data': data, '/artifacts': artifacts})
def train_remote(branch, config, bundle_id, expected_inventory_sha):
    import cv2
    import torch
    from src.training.personal_c7 import branch_plan, branch_manifest, validate_parent, exposure_report, canonical_hash
    from src.training.manifest_tools import rebase_manifest
    from src.training.multidataset import digest
    from src.training.staged import run_stage
    from src.training.chesscog_c5 import claim_once, training_loss_recorder, loss_summary, start_utilization_monitor
    torch.set_num_threads(2); cv2.setNumThreads(1)
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():
        raise ValueError('Single NVIDIA L4 required')
    root = Path('/data') / bundle_id
    if expected_inventory_sha != config['expected_inventory_sha256']:
        raise ValueError('Dataset fingerprint differs from authorized C.7 configuration')
    if digest(root/'bundle_inventory.json') != expected_inventory_sha:
        raise ValueError('Uploaded inventory changed')
    inventory = json.loads((root/'bundle_inventory.json').read_text())
    for rel, expected in inventory.items():
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()): raise ValueError('Escaping bundle path')
        if path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
            raise ValueError('Uploaded bundle member changed')
    plan = branch_plan(config, branch)
    initial = Path(config['parent_path'])
    state = torch.load(initial, map_location='cpu', weights_only=True)
    validate_parent(initial, state); del state
    manifest = branch_manifest(rebase_manifest(root), branch)
    out = Path('/artifacts') / ('personal-c7-' + branch.lower() + '-seed42-v1')
    out.mkdir(parents=True, exist_ok=True)
    claim_once(out/'one_attempt_claim.json', {'branch':branch, 'parent_sha256':digest(initial),
        'dataset_fingerprint':canonical_hash(manifest), 'plan_hash':canonical_hash(plan)}, artifacts.commit)
    (out/'authorized_config.json').write_text(json.dumps(plan, indent=2)+'\n')
    (out/'exposure.json').write_text(json.dumps(exposure_report(manifest, plan), indent=2)+'\n')
    import importlib
    modules=['src.training.personal_c7','src.training.staged','src.training.multidataset','src.vision.classifier','src.vision.crop_experiments','src.vision.grid_geometry','src.vision.perspective']
    executed_sources={name:digest(importlib.import_module(name).__file__) for name in modules}
    (out/'executed_sources.json').write_text(json.dumps(executed_sources,indent=2)+'\n')
    artifacts.commit()
    begin = time.perf_counter()
    samples, stopped, thread = start_utilization_monitor()
    try:
        with training_loss_recorder(112, out) as records:
            result = run_stage(manifest, plan, 'personal', out, device='cuda',
                initial=str(initial), commit=artifacts.commit)
    finally:
        stopped.set(); thread.join(timeout=3)
    result.update(c7_branch=branch, parent_sha256=digest(initial),
        training_loss_by_interval=loss_summary(records), remote_training_seconds=time.perf_counter()-begin,
        checkpoint_hashes={p.name:digest(p) for p in out.glob('*.pt')},
        gpu_utilization_samples=len(samples),
        mean_gpu_utilization_percent=sum(r[0] for r in samples)/len(samples) if samples else None,
        max_nvidia_smi_memory_mib=max((r[1] for r in samples), default=None))
    final = torch.load(out/'last.pt', map_location='cpu', weights_only=True)
    result.update(completed_intervals=final['epoch'], next_batch=final['next_batch'],
        partial_training=final['updates']<672 or final['epoch']<6,
        resumable_keys_present=all(k in final for k in ['optimizer_state_dict','scheduler_state_dict','scaler_state_dict','rng','next_batch','plan_hash','dataset_fingerprint']))
    (out/'c7_result.json').write_text(json.dumps(result, indent=2)+'\n'); artifacts.commit()
    return result

@app.local_entrypoint()
def main(action:str='upload', branch:str='B', bundle:str='outputs/evaluation/personal_c7/bundle_v1',
         bundle_id:str='personal-c7-bundle-v1', plan_path:str='configs/training/personal_c7_authorized_v1.json'):
    from src.training.multidataset import digest
    from src.training.manifest_tools import write_new
    from src.training.personal_c7 import BEST_NAMES, branch_plan
    workspace=modal.Workspace.from_context(); workspace.hydrate()
    config=json.loads(Path(plan_path).read_text()); branch_plan(config, branch)
    if workspace.name != config['modal']['workspace']: raise ValueError('Unexpected workspace')
    if action=='upload':
        if (ROOT/'upload_v1.json').exists(): raise ValueError('Already uploaded; refuse automatic repeat')
        root=Path(bundle)
        if any(p.suffix.lower() in ('.mov','.mp4') for p in root.rglob('*')): raise ValueError('No original media uploads')
        from src.training.personal_c7 import bundle_batches
        for index, paths in enumerate(bundle_batches(root)):
            with data.batch_upload() as upload:
                for path in paths: upload.put_file(path, '/'+bundle_id+'/'+str(path.relative_to(root)))
            (ROOT/'upload_progress_v2.json').write_text(json.dumps({'committed_batches':index+1,'bundle_id':bundle_id})+'\n')
            print('Committed upload batch', index+1, flush=True)
        write_new(ROOT/'upload_v1.json', {'inventory_sha256':digest(root/'bundle_inventory.json'), 'bundle_id':bundle_id,
            'bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file())})
    elif action=='train':
        target=ROOT/(branch+'_result_v1.json')
        if target.exists(): raise ValueError('Branch already ran; no automatic retraining')
        result=train_remote.remote(branch, config, bundle_id, digest(Path(bundle)/'bundle_inventory.json'))
        write_new(target, result); print(json.dumps(result, indent=2))
    elif action=='export':
        run='personal-c7-'+branch.lower()+'-seed42-v1'; target=ROOT/'checkpoints'/branch
        target.mkdir(parents=True, exist_ok=True)
        expected=json.loads((ROOT/(branch+'_result_v1.json')).read_text())['checkpoint_hashes']
        for name in [BEST_NAMES[branch], 'last.pt','authorized_config.json','exposure.json','metrics.jsonl','c7_result.json','one_attempt_claim.json','executed_sources.json']:
            path=target/name
            if path.exists(): raise ValueError('Refuse overwrite of export')
            with path.open('xb') as f:
                for chunk in artifacts.read_file('/'+run+'/'+name): f.write(chunk)
            if name in expected and digest(path)!=expected[name]: raise ValueError('Exported checkpoint checksum mismatch')
        write_new(ROOT/(branch+'_export_v1.json'), {'files':{p.name:digest(p) for p in target.iterdir() if p.is_file()}, 'bytes':sum(p.stat().st_size for p in target.iterdir())})
    else: raise ValueError('Only upload, train and export are supported')
