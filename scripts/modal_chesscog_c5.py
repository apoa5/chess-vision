"""C5 bounded one-run driver. No stage chaining, GPU retries or test downloads."""
import json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal
from scripts.modal_multidataset_c4 import image,data,artifacts

app=modal.App('chess-vision-c5')
ROOT=Path('outputs/evaluation/chesscog_c5')

@app.function(image=image,cpu=(2,2),memory=4096,timeout=1800,startup_timeout=300,
              nonpreemptible=True,max_containers=1,min_containers=0,retries=0,
              scaledown_window=2,volumes={'/data':data})
def prepare_remote(dataset_id):
    import cv2
    from src.training.chesscog_c5 import prepare
    cv2.setNumThreads(1)
    root=Path('/data')/dataset_id
    return prepare(root,json.loads((root/'source_plan_v1.json').read_text()),commit=data.commit)

def loaded_manifest(dataset_id,expected=None):
    from src.training.manifest_tools import rebase_manifest
    from src.training.multidataset import digest
    root=Path('/data')/dataset_id
    if expected and digest(root/'manifest.json')!=expected:raise ValueError('Prepared dataset changed')
    m=rebase_manifest(root)
    if any(b['dataset']!='chesscog' or b['split'] not in ('train','val') for b in m['boards']):raise ValueError('Non-Chesscog or test source prohibited')
    return root,m

def initialization(root):
    import shutil
    from src.training.multidataset import digest
    p=root/'torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth'
    source=Path('/data/c4-smoke-v1/torch/hub/checkpoints')/p.name
    if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,p);data.commit()
    if digest(p)!=digest(source):raise ValueError('ImageNet initialization changed')
    os.environ['TORCH_HOME']=str(root/'torch')

@app.function(image=image,gpu='L4',cpu=(2,2),memory=8192,timeout=240,startup_timeout=300,
              max_containers=1,min_containers=0,retries=0,scaledown_window=2,
              volumes={'/data':data,'/artifacts':artifacts})
def profile_remote(dataset_id,manifest_sha,config):
    import torch,cv2
    from src.training.chesscog_c5 import throughput,claim_once
    torch.set_num_threads(2);cv2.setNumThreads(1)
    root,m=loaded_manifest(dataset_id,manifest_sha);initialization(root)
    marker=Path('/artifacts/c5-throughput-started.json')
    claim_once(marker,dict(dataset=dataset_id,manifest_sha256=manifest_sha),artifacts.commit)
    return throughput(m,config)

@app.function(image=image,gpu='L4',cpu=(2,2),memory=8192,timeout=5100,startup_timeout=300,
              max_containers=1,min_containers=0,retries=0,scaledown_window=2,
              volumes={'/data':data,'/artifacts':artifacts})
def train_remote(dataset_id,config,run_id):
    import torch,cv2
    from src.training.staged import run_stage
    from src.training.chesscog_c5 import training_loss_recorder,loss_summary,evaluate_checkpoints,claim_once,initial_validation,start_utilization_monitor
    from src.training.multidataset import board_ready,digest
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if not config.get('full_training_authorization') or config['stages']['chesscog']['max_seconds']>4500:raise ValueError('C5 authorization/time bound required')
    root,m=loaded_manifest(dataset_id,config['expected_dataset_manifest_sha256']);initialization(root)
    out=Path('/artifacts')/run_id;out.mkdir(parents=True,exist_ok=True)
    marker=out/'one_attempt_claim.json'
    claim_once(marker,dict(run_id=run_id,dataset=dataset_id,source_manifest_sha256=digest(root/'manifest.json')),artifacts.commit)
    (out/'authorized_config.json').write_text(json.dumps(config,indent=2)+'\n');artifacts.commit()
    batches=sum(b['split']=='train' and board_ready(b) for b in m['boards'])
    begin=time.perf_counter()
    samples,monitor_stop,monitor_thread=start_utilization_monitor()
    initial=initial_validation(m,config,out);artifacts.commit()
    with training_loss_recorder(batches,out) as records:
        result=run_stage(m,config,'chesscog',out,device='cuda',commit=artifacts.commit)
    result['training_loss_by_epoch']=loss_summary(records)
    result['initial_validation']=initial
    (out/'training_loss.json').write_text(json.dumps(result['training_loss_by_epoch'],indent=2)+'\n')
    result['validation']=evaluate_checkpoints(m,out,config)
    result['remote_total_seconds']=time.perf_counter()-begin
    result['checkpoint_hashes']={p.name:digest(p) for p in out.glob('*.pt')}
    result['completed_epochs']=int(torch.load(out/'last.pt',map_location='cpu',weights_only=True)['epoch'])
    result['partial_training']=result['completed_epochs']<config['stages']['chesscog']['epochs']
    result['torch_version']=torch.__version__
    monitor_stop.set();monitor_thread.join(timeout=3)
    result['gpu_utilization_samples']=len(samples)
    result['mean_gpu_utilization_percent']=sum(r[0] for r in samples)/len(samples) if samples else None
    result['max_nvidia_smi_memory_mib']=max((r[1] for r in samples),default=None)
    (out/'c5_result.json').write_text(json.dumps(result,indent=2)+'\n');artifacts.commit()
    return result

@app.local_entrypoint()
def main(action:str='prepare-chesscog',dataset_id:str='chesscog-c5-v1',
         plan_path:str='configs/training/chesscog_c5_authorized_v1.json',run_id:str='chesscog-c5-seed42-v1'):
    from src.training.manifest_tools import write_new
    w=modal.Workspace.from_context();w.hydrate()
    if w.name!='akwasipoku-oforiagyei':raise ValueError('Unexpected workspace')
    if action=='prepare-chesscog':
        with data.batch_upload() as upload:upload.put_file(ROOT/'source_plan_v1.json','/'+dataset_id+'/source_plan_v1.json')
        result=prepare_remote.remote(dataset_id)
    elif action=='profile':
        p=json.loads((ROOT/'prepare-chesscog_result_v1.json').read_text());config=json.loads(Path('configs/training/multidataset_c4_v1.json').read_text())
        result=profile_remote.remote(dataset_id,p['dataset_fingerprint'],config)
    elif action=='train':
        config=json.loads(Path(plan_path).read_text());result=train_remote.remote(dataset_id,config,run_id)
    else:raise ValueError('Unknown action')
    write_new(ROOT/(action+'_result_v1.json'),result);print(json.dumps(result,indent=2))
