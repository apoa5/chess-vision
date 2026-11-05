"""C.4 isolated Modal runner. Only `--action smoke` is authorized in C.4.

No original photographs/videos, tokens, or production checkpoints enter the image.
Full training remains disabled in the checked-in plan.
"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal

def rebase_manifest(root):
    # Lazy: the separate persistence checker intentionally has no torch/OpenCV.
    from src.training.manifest_tools import rebase_manifest as load
    return load(root)

app=modal.App('chess-vision-c4')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.8.0','torchvision==0.23.0',index_url='https://download.pytorch.org/whl/cu128')
       .pip_install('numpy==2.2.6','opencv-python-headless==4.12.0.88','Pillow==11.3.0','python-chess==1.999')
       .add_local_python_source('src')
       .add_local_python_source('scripts'))
data=modal.Volume.from_name('chess-vision-c4-data-v1',create_if_missing=True)
artifacts=modal.Volume.from_name('chess-vision-c4-artifacts-v1',create_if_missing=True)

@app.function(image=image,gpu='L4',cpu=2,memory=8192,timeout=600,startup_timeout=600,
              max_containers=1,min_containers=0,retries=0,scaledown_window=2,
              volumes={'/data':data,'/artifacts':artifacts})
def smoke_remote(plan,run_id):
    entered=time.perf_counter()
    import torch
    from src.training.multidataset import digest
    from src.training.staged import run_stage
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():raise RuntimeError('Expected a single L4')
    root=Path('/data')/run_id
    inventory=json.loads((root/'bundle_inventory.json').read_text())
    for rel,expected in inventory.items():
        p=root/rel
        if p.stat().st_size!=expected['bytes'] or digest(p)!=expected['sha256']:raise ValueError('Changed uploaded bundle member')
    os.environ['TORCH_HOME']=str(root/'torch')
    output=Path('/artifacts')/run_id
    if output.exists():raise ValueError('Refuse to overwrite an earlier run')
    result=run_stage(rebase_manifest(root),plan,'smoke',output,device='cuda',smoke=True,commit=artifacts.commit)
    result['remote_function_seconds']=time.perf_counter()-entered
    result['torch_version']=torch.__version__;result['cuda_version']=torch.version.cuda
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');artifacts.commit()
    return result

@app.function(image=modal.Image.debian_slim(python_version='3.12'),cpu=.25,memory=256,
              timeout=60,max_containers=1,scaledown_window=2,volumes={'/artifacts':artifacts})
def verify_persistent(run_id,expected):
    import hashlib
    p=Path('/artifacts')/run_id/'last.pt'
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=expected:raise ValueError('Checkpoint not persistent across containers')
    return dict(verified_from_separate_cpu_container=True,sha256=h,bytes=p.stat().st_size)

@app.function(image=image,cpu=2,memory=8192,timeout=7200,max_containers=1,retries=0,
              scaledown_window=2,volumes={'/data':data})
def prepare_remote(dataset_id,protected):
    from src.training.external_prepare import prepare_chesscog
    root=Path('/data')/dataset_id;root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise ValueError('Use a fresh dataset directory; do not overwrite prepared data or partial receipts')
    (root/'personal_protected_placements.json').write_text(json.dumps(protected))
    result=prepare_chesscog(root);data.commit();return result

@app.function(image=image,gpu='L4',cpu=2,memory=8192,timeout=7200,startup_timeout=600,
              max_containers=1,min_containers=0,retries=0,scaledown_window=2,
              volumes={'/data':data,'/artifacts':artifacts})
def train_remote(plan,stage,manifest_directory,run_id,initial=None,resume=None):
    # Defend remotely as well as at the local entry point.
    if not plan.get('full_training_authorization'):raise ValueError('Separate full-training authorization required')
    import torch
    from src.training.staged import run_stage
    torch.set_num_threads(2)
    os.environ['TORCH_HOME']=str(Path('/data')/manifest_directory/'torch')
    out=Path('/artifacts')/run_id
    if out.exists() and not resume:raise ValueError('Use a fresh run ID or explicit resume')
    return run_stage(rebase_manifest(Path('/data')/manifest_directory),plan,stage,out,
                     device='cuda',initial=initial,resume=resume,commit=artifacts.commit)

@app.local_entrypoint()
def main(action:str='smoke',bundle:str='outputs/evaluation/multidataset_c4/smoke_bundle_v2',
         plan_path:str='configs/training/multidataset_c4_v1.json',run_id:str='c4-smoke-v1',
         stage:str='chesscog',initial:str='',resume:str='',authorize_full_download:bool=False,experiment:str='E'):
    from src.training.manifest_tools import write_new
    plan=json.loads(Path(plan_path).read_text())
    w=modal.Workspace.from_context();w.hydrate()
    if w.name!=plan['modal']['workspace']:raise ValueError('Unexpected Modal workspace')
    if action not in ('smoke','train','prepare-chesscog'):raise ValueError('Unknown action')
    if action=='prepare-chesscog':
        if not authorize_full_download:raise ValueError('Explicit full dataset download approval required')
        from src.training.manifest_tools import protected_personal
        result=prepare_remote.remote(bundle,protected_personal())
    elif action=='train':
        if not plan.get('full_training_authorization'):raise ValueError('Full training NOT authorized in C.4')
        from src.training.staged import experiment_plan
        plan=experiment_plan(plan,experiment)
        result=train_remote.remote(plan,stage,bundle,run_id,initial or None,resume or None)
    else:
        if plan['smoke']['max_updates']>4 or plan['smoke']['max_seconds']>180:raise ValueError('Smoke limits exceeded')
        root=Path(bundle)
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>25_000_000:raise ValueError('Smoke upload exceeds compact bundle limit')
        t=time.perf_counter()
        with data.batch_upload() as upload:upload.put_directory(root,'/'+run_id)
        result=smoke_remote.remote(plan,run_id)
        result['persistence']=verify_persistent.remote(run_id,result['last_checkpoint_sha256'])
        result['upload_and_remote_wall_seconds']=time.perf_counter()-t
    result['workspace']=w.name
    write_new(Path('outputs/evaluation/multidataset_c4')/(run_id+'_result.json'),result)
    print(json.dumps(result,indent=2))
