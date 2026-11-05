"""C5 recovery, measurements and validation; reuses the unchanged C4 model/kernel."""
import contextlib,hashlib,json,os,shutil,time,urllib.request
from pathlib import Path
import cv2,numpy as np
from src.training.multidataset import (PROFILE,MultiDatasetSquares,board_ready,chesscog_board,digest,
                                      position_key,quarantine_protected,split_audit)
from src.chess.labels import CLASS_NAMES

def claim_once(path,payload,commit=lambda:None):
    """A persisted exclusive claim prevents platform replay from restarting work."""
    try:
        with Path(path).open('x') as f:f.write(json.dumps(payload))
    except FileExistsError as e:raise ValueError('Already attempted; no automatic GPU retry') from e
    commit()

def incremental_cost(before,after):
    """Conservative snapshot delta: aggregate and component fields can lag."""
    aggregate=float(after['metered_cost'])-float(before['metered_cost'])
    components=sum(float(v) for v in after.get('metered_cost_breakdown',{}).values())-sum(float(v) for v in before.get('metered_cost_breakdown',{}).values())
    return max(0.,aggregate,components)

def start_utilization_monitor(interval=15):
    """Best-effort nvidia-smi sampling; no fabricated utilization if unavailable."""
    import subprocess,threading
    rows=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
                if p.returncode==0:rows.append([float(x.strip()) for x in p.stdout.strip().splitlines()[0].split(',')])
            except (OSError,ValueError,subprocess.TimeoutExpired):pass
            stop.wait(interval)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();return rows,stop,thread

def prepare(root,plan,commit=lambda:None):
    """Recover a preempted preparation without overwrites or full local copies."""
    from scripts.data.probe_chesscog_c4 import member
    root=Path(root);started=time.perf_counter()
    if (root/'manifest.json').exists():raise ValueError('Completed preparation exists; do not overwrite')
    files=plan['files']
    if set(files)!={'train.z01','train.zip','val.zip'}:raise ValueError('Only official train/validation allowed')
    receipts=[]
    for name in ('train.z01','train.zip','val.zip'):
        meta=files[name];size=meta['attributes']['size'];p=root/'archives'/name;p.parent.mkdir(exist_ok=True)
        if shutil.disk_usage(root).free<size+7_500_000_000:raise ValueError('Insufficient preparation disk')
        old=p.stat().st_size if p.exists() else 0
        if old>size:raise ValueError('Existing archive exceeds published size')
        if old<size:
            headers={'Range':f'bytes={old}-'} if old else {}
            url=meta['links']['download']+'?c5_full_archive=1'
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
                if old and (response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {old}-')):raise ValueError('Unsafe partial-download response')
                if not old and response.status!=200:raise ValueError('Full archive response required')
                with p.open('ab' if old else 'xb') as dest:
                    n=old
                    for data in iter(lambda:response.read(4*1024*1024),b''):
                        n+=len(data)
                        if n>size:raise ValueError('Oversized download')
                        dest.write(data)
            commit()
        if p.stat().st_size!=size:raise ValueError('Incomplete archive')
        sha=digest(p);expected=meta['attributes']['extra']['hashes']['sha256']
        if sha!=expected:raise ValueError('Official SHA256 mismatch: '+name)
        receipts.append(dict(name=name,bytes=size,sha256=sha,official_sha256_verified=True,reused_bytes=old))
        (root/'archive_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n');commit()
        print('Verified archive',name,size,flush=True)
    def local_range(path,a,b):
        with open(path,'rb') as f:f.seek(a);data=f.read(b-a+1)
        if len(data)!=b-a+1:raise ValueError('Truncated member range')
        return data
    boards=[];excluded=[];pairs={}
    for split in ('train','val'):
        inventory=plan['inventories'][split]
        rows={r['name']:r for r in inventory['rows'] if r['name'].startswith(split+'/') and '__MACOSX' not in r['name']}
        parts=[(str(root/'archives'/n),files[n]['attributes']['size']) for n in (('train.z01','train.zip') if split=='train' else ('val.zip',))]
        annotations=sorted(n for n in rows if n.endswith('.json') and n[:-5]+'.png' in rows);pairs[split]=len(annotations)
        for i,name in enumerate(annotations):
            dest=root/'images'/split;dest.mkdir(parents=True,exist_ok=True)
            for key in (name,name[:-5]+'.png'):
                data=member(rows[key],parts,read_range=local_range);p=dest/Path(key).name
                if p.exists():
                    if p.read_bytes()!=data:raise ValueError('Changed previously extracted member')
                else:p.write_bytes(data)
            try:
                board=chesscog_board(dest/Path(name[:-5]+'.png').name,dest/Path(name).name,split)
                # Validate representative target masks at both board edges and centre.
                ds=MultiDatasetSquares(dict(representation=PROFILE,boards=[board]),split)
                for square in ('a8','h8','d4','a1','h1'):
                    idx=next(j for j,(b,o) in enumerate(ds.rows) if o['square']==square);packed=ds.packed(idx)
                    if packed.shape!=(4,224,224) or set(np.unique(packed[3]))-{0,1}:raise ValueError('Invalid B2 crop/mask')
                boards.append(board)
            except (ValueError,KeyError,StopIteration) as e:
                excluded.append(dict(source_member=name,source_role=split,reason=type(e).__name__+': '+str(e)))
            if (i+1)%200==0:print('Validated',split,i+1,'/',len(annotations),'excluded',len(excluded),flush=True)
        commit()
    protected=json.loads((root/'personal_protected_placements.json').read_text())
    protected.update({position_key(b['placement']):dict(source='official_chesscog_val') for b in boards if b['split']=='val'})
    quarantine_protected(boards,protected);audit=split_audit(boards)
    for b in boards:
        b['source']['path']=str(Path(b['source']['path']).relative_to(root));b['annotation']['path']=str(Path(b['annotation']['path']).relative_to(root))
    manifest=dict(schema_version='multidataset_c5_prepared_v1',representation=PROFILE,class_names=list(CLASS_NAMES),final_holdout=False,boards=boards,
        grouped_split_audit=audit,archive_receipts=receipts,excluded_annotations=excluded,source_plan_sha256=digest(root/'source_plan_v1.json'))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=dict(official_pairs=pairs,valid_boards=len(boards),excluded_annotations=excluded,
        train_boards=sum(b['split']=='train' and board_ready(b) for b in boards),val_boards=sum(b['split']=='val' and board_ready(b) for b in boards),
        quarantined_boards=[dict(id=b['id'],split=b['split'],reason=b['inclusion']['reason']) for b in boards if not board_ready(b)],
        dataset_fingerprint=digest(root/'manifest.json'),archive_receipts=receipts,elapsed_seconds=time.perf_counter()-started,
        storage_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),external_test_downloaded=False)
    (root/'preparation_result.json').write_text(json.dumps(summary,indent=2)+'\n');commit();return summary

@contextlib.contextmanager
def training_loss_recorder(batches_per_epoch,output):
    """Observe the existing CE return value without changing its loss or gradients.

    Scoped to this isolated offline process; restores the original function.
    The C4 kernel logs validation loss but not training loss.
    """
    import torch
    original=torch.nn.functional.cross_entropy;records=[]
    def measured(logits,targets,*args,**kwargs):
        result=original(logits,targets,*args,**kwargs)
        if torch.is_grad_enabled() and logits.requires_grad:
            weight=kwargs.get('weight');denom=float(weight[targets].sum()) if weight is not None else targets.numel()
            records.append(dict(batch=len(records)+1,epoch=len(records)//batches_per_epoch,
                                loss=float(result.detach()),weight_sum=denom,examples=targets.numel()))
            if len(records)%200==0:
                (Path(output)/'train_loss_progress.json').write_text(json.dumps(dict(batches=len(records),recent_loss=float(np.mean([r['loss'] for r in records[-200:]])))))
                print('Training batch',len(records),'epoch',records[-1]['epoch'],'recent loss',round(np.mean([r['loss'] for r in records[-200:]]),5),flush=True)
        return result
    torch.nn.functional.cross_entropy=measured
    try:yield records
    finally:torch.nn.functional.cross_entropy=original

def loss_summary(records):
    result=[]
    for epoch in sorted({r['epoch'] for r in records}):
        rows=[r for r in records if r['epoch']==epoch]
        result.append(dict(epoch=epoch,batches=len(rows),examples=sum(r['examples'] for r in rows),
                           training_loss=sum(r['loss']*r['weight_sum'] for r in rows)/sum(r['weight_sum'] for r in rows)))
    return result

def throughput(manifest,config):
    """128 deterministic full-source boards, batch64, both training modes.

    Discard probe weights. Explicitly measure consumer-visible loader wait;
    CPU prefetch overlaps GPU work, so this is not total worker CPU time.
    """
    import subprocess,threading,torch
    from torch.utils.data import DataLoader
    from src.training.staged import BoardBatches,seed_all
    from src.vision.classifier import create_classifier,set_training_mode
    seed_all(config['seed']);settings=config['stages']['chesscog'];begin=time.perf_counter()
    ds=MultiDatasetSquares(manifest,'train',seed=config['seed'],augmentation=config['augmentation'])
    if len(ds.boards)<settings['minimum_train_boards']:raise ValueError('Full dataset guard not met')
    batches=BoardBatches(ds,settings['batch_size'],config['seed'],0).batches[:128]
    model=create_classifier(pretrained=True,mode='fine_tune').cuda()
    counts=np.bincount([o['class_id'] for b,o in ds.rows],minlength=13)
    weight=torch.tensor(np.sqrt(counts.sum()/np.maximum(counts,1)),dtype=torch.float32,device='cuda')
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate']);samples=[];stopped=threading.Event()
    def monitor():
        while not stopped.is_set():
            try:
                p=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
                if p.returncode==0:samples.append([float(x.strip()) for x in p.stdout.strip().splitlines()[0].split(',')])
            except (OSError,ValueError,subprocess.TimeoutExpired):pass
            stopped.wait(1.)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();phases=[]
    try:
        for phase,chosen in zip(('head_and_stem','fine_tune'),(batches[:64],batches[64:])):
            for p in model.parameters():p.requires_grad=True
            if phase=='head_and_stem':
                for p in model.features.parameters():p.requires_grad=False
                for p in model.features[0].parameters():p.requires_grad=True
            set_training_mode(model,phase)
            loader=iter(DataLoader(ds,batch_sampler=chosen,num_workers=settings['workers'],generator=torch.Generator().manual_seed(0)))
            rows=[]
            for batch in range(len(chosen)):
                start=time.perf_counter();x,y=next(loader);loaded=time.perf_counter();x=x.cuda();y=y.cuda();optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model(x),y,weight=weight)
                if not torch.isfinite(loss):raise ValueError('Nonfinite throughput loss')
                loss.backward();optimizer.step();torch.cuda.synchronize();end=time.perf_counter()
                rows.append(dict(batch=batch,examples=len(y),loader_wait=loaded-start,step_seconds=end-loaded,total=end-start))
                if time.perf_counter()-begin>170:break
            steady=rows[8:]
            if len(steady)<16:raise ValueError('Insufficient representative throughput sample')
            phases.append(dict(mode=phase,batches=len(rows),warmup_batches_excluded=8,examples=sum(r['examples'] for r in rows),
                measured_examples=sum(r['examples'] for r in steady),examples_per_second=sum(r['examples'] for r in steady)/sum(r['total'] for r in steady),
                loader_wait_seconds=sum(r['loader_wait'] for r in steady),gpu_step_seconds=sum(r['step_seconds'] for r in steady),
                measured_seconds=sum(r['total'] for r in steady)))
        val=MultiDatasetSquares(manifest,'val');vstart=time.perf_counter();n=0;model.eval()
        with torch.inference_mode():
            for x,y in DataLoader(val,batch_size=64,shuffle=False,num_workers=0):
                model(x.cuda());n+=len(y)
                if n>=1024:break
        torch.cuda.synchronize();vseconds=time.perf_counter()-vstart
    finally:stopped.set();thread.join(timeout=3)
    head,full=[p['examples_per_second'] for p in phases];epoch_head=len(ds)/head;epoch_full=len(ds)/full;validation_estimate=len(val)/(n/vseconds)
    epoch_budget=epoch_head+7*epoch_full+8*validation_estimate
    return dict(schema_version='chesscog_throughput_c5_v1',train_boards=len(ds.boards),train_examples=len(ds),validation_examples=len(val),
        sampled_board_ids=[ds.rows[batch[0]][0]['id'] for batch in batches],phases=phases,
        validation_sample_examples=n,validation_sample_seconds=vseconds,estimated_validation_seconds=validation_estimate,
        estimated_head_epoch_seconds=epoch_head,estimated_full_epoch_seconds=epoch_full,estimated_8_epoch_seconds=epoch_budget,
        estimated_8_epoch_resource_usd=epoch_budget/3600*.9586,elapsed_seconds=time.perf_counter()-begin,
        gpu_name=torch.cuda.get_device_name(),peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu_utilization_samples=len(samples),mean_gpu_utilization_percent=float(np.mean([r[0] for r in samples])) if samples else None,
        max_gpu_smi_memory_mib=max((r[1] for r in samples),default=None),probe_weights_discarded=True,
        cost_note='reference $0.9586/hour requested L4+CPU2+8GiB; excludes preparation, scheduling and storage; no accuracy conclusions')

def evaluate_checkpoints(manifest,output,config):
    import torch
    from torch.utils.data import DataLoader
    from src.vision.classifier import create_classifier
    from src.training.staged import contract_checkpoint
    from scripts.evaluate_multidataset_c4 import summarize
    ds=MultiDatasetSquares(manifest,'val');results={};matrices={}
    expected=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    expected_plan=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    for name in ('chesscog_best.pt','last.pt'):
        state=torch.load(Path(output)/name,map_location='cpu',weights_only=True);contract_checkpoint(state,'chesscog',expected)
        if state['plan_hash']!=expected_plan:raise ValueError('Checkpoint config provenance mismatch')
        model= create_classifier(pretrained=False,mode='fine_tune').cuda().eval();model.load_state_dict(state['model_state_dict'],strict=True)
        logits=[];started=time.perf_counter();num=den=0.
        counts=np.bincount([o['class_id'] for b in manifest['boards'] if b['split']=='train' and board_ready(b) for o in b['observations']],minlength=13)
        weights=torch.tensor(np.sqrt(counts.sum()/np.maximum(counts,1)),dtype=torch.float32,device='cuda')
        with torch.inference_mode():
            for x,y in DataLoader(ds,batch_size=64,shuffle=False,num_workers=0):
                y=y.cuda();z=model(x.cuda());logits.append(z.cpu().numpy());num+=float(torch.nn.functional.cross_entropy(z,y,weight=weights,reduction='sum'));den+=float(weights[y].sum())
        z=np.concatenate(logits);metrics=summarize(ds.rows,z)
        if metrics['confusion_matrix']!=state['history'][-1]['confusion_matrix']:raise ValueError('Saved validation did not reproduce')
        results[name]=dict(metrics=metrics,validation_loss=num/den,epoch_index=state['history'][-1]['epoch'],updates=state['updates'],
                           checkpoint_sha256=digest(Path(output)/name),elapsed_seconds=time.perf_counter()-started,checkpoint_reload_validated=True)
        matrices[name]=z
        del model
    np.savez_compressed(Path(output)/'validation_logits.npz',**matrices)
    (Path(output)/'validation.json').write_text(json.dumps(results,indent=2)+'\n');return results

def initial_validation(manifest,config,output):
    """Seed-matched ImageNet initialization reference, no checkpoint selection."""
    import torch
    from torch.utils.data import DataLoader
    from src.training.staged import seed_all
    from src.vision.classifier import create_classifier
    from scripts.evaluate_multidataset_c4 import summarize
    seed_all(config['seed']);ds=MultiDatasetSquares(manifest,'val')
    model=create_classifier(pretrained=True,mode='fine_tune').cuda().eval()
    counts=np.bincount([o['class_id'] for b in manifest['boards'] if b['split']=='train' and board_ready(b) for o in b['observations']],minlength=13)
    weights=torch.tensor(np.sqrt(counts.sum()/np.maximum(counts,1)),dtype=torch.float32,device='cuda')
    logits=[];num=den=0.;begin=time.perf_counter()
    with torch.inference_mode():
        for x,y in DataLoader(ds,batch_size=64,shuffle=False,num_workers=0):
            y=y.cuda();z=model(x.cuda());logits.append(z.cpu().numpy())
            num+=float(torch.nn.functional.cross_entropy(z,y,weight=weights,reduction='sum'));den+=float(weights[y].sum())
    result=dict(metrics=summarize(ds.rows,np.concatenate(logits)),validation_loss=num/den,elapsed_seconds=time.perf_counter()-begin,
                initialization='seed42 ImageNet MobileNetV3 Small RGB, zero fourth channel, random13-class head; no optimization')
    (Path(output)/'initial_validation.json').write_text(json.dumps(result,indent=2)+'\n');return result
