"""C9 isolated Modal infrastructure, single-attempt L4 smoke and bounded training."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import modal
from scripts.modal_multidataset_c4 import data,artifacts
app=modal.App('chess-vision-c9')
image=(modal.Image.debian_slim(python_version='3.12')
       .pip_install('torch==2.8.0','torchvision==0.23.0',index_url='https://download.pytorch.org/whl/cu128')
       .pip_install('numpy==2.2.6','opencv-python-headless==4.12.0.88','Pillow==11.3.0','python-chess==1.999')
       .pip_install('timm==1.0.22','safetensors==0.8.0','huggingface-hub==0.36.0')
       .env({'HF_HUB_DOWNLOAD_TIMEOUT':'30','HF_HUB_ETAG_TIMEOUT':'30','HF_HUB_DISABLE_PROGRESS_BARS':'1'})
       .add_local_python_source('src').add_local_python_source('scripts'))
ROOT=Path('outputs/evaluation/convnext_c9')

@app.function(image=image,cpu=(2,2),memory=4096,nonpreemptible=True,timeout=600,startup_timeout=300,
              retries=0,max_containers=1,min_containers=0,scaledown_window=2,volumes={'/data':data})
def prepare_weights():
    from huggingface_hub import hf_hub_download
    from src.training.convnext_c9 import REVISION,WEIGHT_SHA,WEIGHT_BYTES
    from src.training.multidataset import digest
    root=Path('/data/convnext-c9-pretrained-v1');root.mkdir(exist_ok=True)
    start=time.perf_counter()
    path=Path(hf_hub_download(repo_id='timm/convnextv2_tiny.fcmae_ft_in22k_in1k',filename='model.safetensors',revision=REVISION,cache_dir=str(root)))
    if digest(path)!=WEIGHT_SHA or path.stat().st_size!=WEIGHT_BYTES:raise ValueError('Upstream weight hash/size mismatch')
    receipt=dict(path=str(path),sha256=digest(path),bytes=path.stat().st_size,revision=REVISION,license='cc-by-nc-4.0',elapsed_seconds=time.perf_counter()-start)
    (root/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');data.commit();return receipt

@app.function(image=image,cpu=(2,2),memory=4096,nonpreemptible=True,timeout=600,startup_timeout=300,
              retries=0,max_containers=1,min_containers=0,scaledown_window=2,volumes={'/data':data})
def prepare_dataset(receipt):
    from src.training.transfer_c9 import extract_archive
    result=extract_archive('/data/convnext-c9-transfer-v1/data.tar.gz','/data/convnext-c9-data-v1',receipt)
    (Path('/data/convnext-c9-data-v1')/'preparation.json').write_text(json.dumps(result,indent=2)+'\n');data.commit();return result

def runtime_inputs(config):
    from src.training.multidataset import digest
    root=Path(config['dataset_root'])
    if digest(root/'manifest.json')!=config['manifest_sha256']:raise ValueError('Frozen C9 dataset changed')
    manifest=json.loads((root/'manifest.json').read_text())
    pretrained=json.loads(Path('/data/convnext-c9-pretrained-v1/receipt.json').read_text())
    return manifest,Path(pretrained['path'])

@app.function(image=image,gpu='L4',cpu=(2,2),memory=8192,timeout=480,startup_timeout=300,
              retries=0,max_containers=1,min_containers=0,scaledown_window=2,volumes={'/data':data,'/artifacts':artifacts})
def smoke_remote(config):
    import torch,cv2,numpy as np
    from torch.utils.data import DataLoader
    from src.training.dataset_c9 import C9Squares,coverage_batches
    from src.training.convnext_c9 import make_model,phase_parameters,set_phase,WEIGHT_SHA
    from src.training.staged import seed_all,checkpoint_save
    from src.training.multidataset import digest
    from src.training.chesscog_c5 import claim_once
    from src.chess.labels import CLASS_NAMES
    torch.set_num_threads(2);cv2.setNumThreads(1);seed_all(config['seed']);begin=time.perf_counter()
    if not torch.cuda.is_available() or 'L4' not in torch.cuda.get_device_name():raise ValueError('Expected L4')
    manifest,pretrained=runtime_inputs(config);out=Path('/artifacts/convnext-c9-smoke-v1');out.mkdir(exist_ok=True)
    claim_once(out/'one_attempt_claim.json',{'manifest_sha256':config['manifest_sha256']},artifacts.commit)
    ds=C9Squares(manifest,'train',seed=config['seed'],augmentation=config['augmentation'])
    model=make_model(pretrained).cuda().eval();first=ds[0][0][None].cuda();other=first.clone();other[:,3]=1-first[:,3]
    with torch.inference_mode():a=model(first);b=model(other)
    if a.shape!=(1,13) or not torch.equal(a,b) or torch.count_nonzero(model.stem[0].weight[:,3:]):raise ValueError('Initial stem/output contract failed')
    parameters=sum(p.numel() for p in model.parameters());optimizer=torch.optim.AdamW(phase_parameters(model,'full',config));batches=coverage_batches(ds,config['batch_size'],42,0)
    chosen=[batches[i] for i in np.linspace(0,len(batches)-1,64,dtype=int)]
    phases=[];nonzero=False
    for phase in ['head_stem','full']:
        phase_parameters(model,phase,config);set_phase(model,phase);rows=[]
        loader=iter(DataLoader(ds,batch_sampler=chosen,num_workers=config['workers'],generator=torch.Generator().manual_seed(42)))
        for index in range(len(chosen)):
            t=time.perf_counter();x,y=next(loader);loaded=time.perf_counter();x=x.cuda();y=y.cuda();optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:group['lr']=config[phase+'_lr'].get(group['group_name'].split('_')[0],0.)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
            loss.backward();nonzero |= bool(torch.count_nonzero(model.stem[0].weight.grad));optimizer.step();torch.cuda.synchronize()
            rows.append(dict(examples=len(y),seconds=time.perf_counter()-t,loader_wait=loaded-t))
            if time.perf_counter()-begin>390:break
        steady=rows[8:]
        if len(steady)<16:raise ValueError('Insufficient bounded throughput sample; no training authorization')
        phases.append(dict(phase=phase,batches=len(rows),warmup_excluded=8,measured_examples=sum(x['examples'] for x in steady),
            examples_per_second=sum(x['examples'] for x in steady)/sum(x['seconds'] for x in steady),loader_wait_seconds=sum(x['loader_wait'] for x in steady)))
    model.eval()
    with torch.inference_mode():expected=model(first).cpu()
    checkpoint_save(out/'smoke_only.pt',dict(model_state_dict=model.state_dict(),class_names=list(CLASS_NAMES),input_channels=4,architecture=config['architecture'],smoke_only=True),artifacts.commit)
    reloaded=make_model().cuda().eval();reloaded.load_state_dict(torch.load(out/'smoke_only.pt',map_location='cuda',weights_only=True)['model_state_dict'])
    with torch.inference_mode():actual=reloaded(first).cpu()
    if not torch.allclose(expected,actual,atol=1e-6,rtol=1e-5):raise ValueError('Smoke checkpoint reload mismatch')
    val=C9Squares(manifest,'val');t=time.perf_counter();n=0
    with torch.inference_mode():
        for x,y in DataLoader(val,batch_size=config['batch_size'],num_workers=0):
            model(x.cuda());n+=len(y)
            if n>=512:break
    torch.cuda.synchronize();val_sps=n/(time.perf_counter()-t)
    result=dict(schema_version='convnext_c9_smoke_v1',phases=phases,gradient_nonzero=nonzero,checkpoint_reload_valid=True,initial_mask_zero_rgb_behavior_preserved=True,
        output_shape=[1,13],parameters=parameters,gpu_name=torch.cuda.get_device_name(),gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),train_observations=len(ds),coverage_epoch_observations=sum(len(b) for b in batches),validation_observations=len(val),validation_examples_per_second=val_sps,
        manifest_sha256=config['manifest_sha256'],pretrained_sha256=digest(pretrained),checkpoint_sha256=digest(out/'smoke_only.pt'),elapsed_seconds=time.perf_counter()-begin,
        weights_discarded_for_training=True,smoke_only=True)
    (out/'smoke_result.json').write_text(json.dumps(result,indent=2)+'\n');artifacts.commit();return result

@app.function(image=image,cpu=.25,memory=512,timeout=60,startup_timeout=120,
              retries=0,max_containers=1,min_containers=0,scaledown_window=2,volumes={'/artifacts':artifacts})
def persistence(expected):
    import hashlib
    path=Path('/artifacts/convnext-c9-smoke-v1/smoke_only.pt');h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Independent persistence check failed')
    return dict(separate_cpu_container=True,sha256=h.hexdigest())

@app.function(image=image,gpu='L4',cpu=(2,2),memory=8192,timeout=10800,startup_timeout=300,
              retries=0,max_containers=1,min_containers=0,scaledown_window=2,volumes={'/data':data,'/artifacts':artifacts})
def train_remote(config, continuation=None):
    import torch,cv2
    from src.training.trainer_c9 import run_training,settings_guard,fingerprint
    from src.training.multidataset import digest
    from src.training.chesscog_c5 import claim_once,start_utilization_monitor
    from src.training.convnext_c9 import make_model
    from src.training.dataset_c9 import C9Squares
    from src.training.evaluation_c9 import evaluate
    from src.vision.classifier import create_classifier
    from src.chess.labels import CLASS_NAMES
    torch.set_num_threads(2);cv2.setNumThreads(1);settings_guard(config)
    if 'L4' not in torch.cuda.get_device_name():raise ValueError('Expected L4')
    manifest,pretrained=runtime_inputs(config);out=Path('/artifacts/convnext-c9-seed42-v1');out.mkdir(exist_ok=True)
    resume=None
    if continuation is None:
        claim_once(out/'one_attempt_claim.json',dict(plan_hash=fingerprint(config),manifest_sha256=config['manifest_sha256']),artifacts.commit)
        (out/'authorized_config.json').write_text(json.dumps(config,indent=2)+'\n');artifacts.commit()
    else:
        from src.training.resume_c9 import verify_continuation
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        resume=verify_continuation(out,continuation,config,state)
        claim_once(out/'authorized_continuation_v1.json',continuation,artifacts.commit)
        del state
    samples,stopped,thread=start_utilization_monitor();begin=time.perf_counter()
    try:result=run_training(manifest,config,pretrained,out,device='cuda',commit=artifacts.commit,resume=resume)
    finally:stopped.set();thread.join(timeout=3)
    (out/'training_result.json').write_text(json.dumps(result,indent=2)+'\n');artifacts.commit()
    best=out/'convnextv2_tiny_best.pt';state=torch.load(best,map_location='cpu',weights_only=True)
    lock=dict(checkpoint_sha256=digest(best),plan_hash=fingerprint(config),dataset_fingerprint=fingerprint(manifest),manifest_sha256=config['manifest_sha256'],
        selected_updates=state['updates'],selection='minimum validation loss; frozen before internal test',internal_test_tuning=False)
    with (out/'selection_lock.json').open('x') as f:f.write(json.dumps(lock,indent=2)+'\n')
    artifacts.commit();torch.cuda.empty_cache()
    model=make_model().cuda().eval();model.load_state_dict(state['model_state_dict'])
    ref_path=Path(config['dataset_root'])/'b2_reference.pt'
    if digest(ref_path)!=config['baseline_checkpoint_sha256']:raise ValueError('Frozen B2 reference changed')
    baseline_state=torch.load(ref_path,map_location='cpu',weights_only=True)
    if baseline_state['class_names']!=list(CLASS_NAMES):raise ValueError('B2 labels changed')
    baseline=create_classifier(pretrained=False,mode='fine_tune').cuda().eval();baseline.load_state_dict(baseline_state['model_state_dict'])
    evaluations={};val=C9Squares(manifest,'val')
    for name,m in [('convnext',model),('b2',baseline)]:
        evaluations[name+'_val']=evaluate(m,val,device='cuda',batch_size=config['batch_size'],class_weights=result['class_weights'])
        (out/(name+'_val.json')).write_text(json.dumps(evaluations[name+'_val'])+'\n');artifacts.commit()
    claim_once(out/'internal_test_one_attempt.json',lock,artifacts.commit)
    test=C9Squares(manifest,'internal_test',allow_internal_test=True)
    for name,m in [('convnext',model),('b2',baseline)]:
        evaluations[name+'_internal_test']=evaluate(m,test,device='cuda',batch_size=config['batch_size'],class_weights=result['class_weights'])
        (out/(name+'_internal_test.json')).write_text(json.dumps(evaluations[name+'_internal_test'])+'\n');artifacts.commit()
    result['evaluations']={k:v['metrics'] for k,v in evaluations.items()};result['selection_lock']=lock;result['gpu_utilization_samples']=len(samples)
    result['mean_gpu_utilization_percent']=sum(s[0] for s in samples)/len(samples) if samples else None;result['total_function_region_seconds']=time.perf_counter()-begin
    (out/'c9_result.json').write_text(json.dumps(result,indent=2)+'\n');artifacts.commit();return result

@app.local_entrypoint()
def main(action:str='prepare-weights'):
    workspace=modal.Workspace.from_context();workspace.hydrate()
    if workspace.name!='akwasipoku-oforiagyei':raise ValueError('Unexpected Modal workspace')
    from src.training.manifest_tools import write_new
    if action=='prepare-weights':
        if (ROOT/'pretrained_weights_v1.json').exists():raise ValueError('Already prepared; no automatic download repeat')
        result=prepare_weights.remote();write_new(ROOT/'pretrained_weights_v1.json',result);print(json.dumps(result,indent=2))
    elif action=='upload':
        if (ROOT/'upload_v1.json').exists():raise ValueError('Upload already completed')
        receipt=json.loads((ROOT/'package_v1.json').read_text())
        with data.batch_upload() as upload:upload.put_file(ROOT/'transfer_v1.tar.gz','/convnext-c9-transfer-v1/data.tar.gz')
        write_new(ROOT/'upload_v1.json',receipt)
    elif action=='prepare-data':
        if (ROOT/'prepared_dataset_v1.json').exists():raise ValueError('Dataset already prepared')
        result=prepare_dataset.remote(json.loads((ROOT/'package_v1.json').read_text()))
        write_new(ROOT/'prepared_dataset_v1.json',result);print(json.dumps(result,indent=2))
    elif action=='smoke':
        if not (ROOT/'prepared_dataset_v1.json').exists() or (ROOT/'smoke_result_v1.json').exists():raise ValueError('Prepared data/fresh smoke required')
        config=json.loads(Path('configs/training/convnext_c9_authorized_v1.json').read_text());result=smoke_remote.remote(config)
        result['persistence']=persistence.remote(result['checkpoint_sha256']);write_new(ROOT/'smoke_result_v1.json',result);print(json.dumps(result,indent=2))
    elif action=='recover-smoke':
        if (ROOT/'smoke_result_v1.json').exists():raise ValueError('Smoke receipt already recovered')
        raw=b''.join(artifacts.read_file('/convnext-c9-smoke-v1/smoke_result.json'))
        result=json.loads(raw)
        config=json.loads(Path('configs/training/convnext_c9_authorized_v1.json').read_text())
        if result['manifest_sha256']!=config['manifest_sha256'] or result['pretrained_sha256']!=config['pretrained_sha256']:raise ValueError('Recovered smoke provenance mismatch')
        (ROOT/'smoke_gpu_receipt_v1.json').write_bytes(raw)
        result['persistence']=persistence.remote(result['checkpoint_sha256'])
        result['recovered_without_gpu_rerun']=True
        write_new(ROOT/'smoke_result_v1.json',result);print(json.dumps(result,indent=2))
    elif action=='continue-authorized':
        if (ROOT/'training_result_v1.json').exists():raise ValueError('Campaign already complete')
        config=json.loads(Path('configs/training/convnext_c9_authorized_v1.json').read_text())
        authorization=json.loads((ROOT/'continuation_authorization_v1.json').read_text())
        if not authorization.get('continuation_authorized') or authorization['remaining_resource_bound_usd']+authorization['previous_metered_usd']>config['budget_usd']:
            raise ValueError('Continuation authorization or cumulative budget failed')
        result=train_remote.remote(config,authorization)
        write_new(ROOT/'training_result_v1.json',result);print(json.dumps(result,indent=2))
    elif action=='train':
        if (ROOT/'training_result_v1.json').exists():raise ValueError('No automatic repeat training')
        config=json.loads(Path('configs/training/convnext_c9_authorized_v1.json').read_text());smoke=json.loads((ROOT/'smoke_result_v1.json').read_text())
        if smoke['manifest_sha256']!=config['manifest_sha256'] or not smoke['checkpoint_reload_valid'] or not smoke['gradient_nonzero']:raise ValueError('Feasibility/data gate failed')
        head,full=[p['examples_per_second'] for p in smoke['phases']];n=smoke['coverage_epoch_observations']
        seconds=n/head*config['head_epochs']+n/full*(config['epochs']-config['head_epochs'])+config['epochs']*smoke['validation_observations']/smoke['validation_examples_per_second']
        estimate=seconds*1.35/3600*.9586+.65
        if seconds*1.35>config['max_seconds'] or estimate>5:raise ValueError('Measured schedule cannot fit registered time/cost; stop without reducing objective')
        write_new(ROOT/'training_cost_gate_v1.json',dict(projected_kernel_seconds=seconds,safety_factor=1.35,estimated_total_usd=estimate,upper_function_resource_usd=(10800+300+480+300)/3600*.9586+.85,budget_usd=5))
        result=train_remote.remote(config);write_new(ROOT/'training_result_v1.json',result);print(json.dumps(result,indent=2))
    else:raise ValueError('Unknown/unimplemented action; no training fallback')
