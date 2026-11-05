"""C.4 bounded staged training kernel, resumable checkpoints and grouped batches."""
import contextlib,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
from src.training.multidataset import MultiDatasetSquares,digest
from src.vision.classifier import create_classifier,set_training_mode
from src.chess.labels import CLASS_NAMES


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]))


def restore_rng(state):
    random.setstate(state['python']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])
    n=state['numpy'];np.random.set_state((n[0],np.array(n[1],np.uint32),n[2],n[3],n[4]))


class BoardBatches(Sampler):
    """Shuffle boards and their squares; reuse one warp/decode for up to64 targets.

    Optional E mixture is position-balanced, label-blind and has the SAME fixed
    optimizer-update budget as D. No random frame-level validation splitting.
    """
    def __init__(self,dataset,batch_size,seed,epoch,*,video_fraction=0.,position_balanced=False,max_batches=None):
        groups={}
        for i,(b,o) in enumerate(dataset.rows):groups.setdefault(b['id'],[]).append(i)
        rng=np.random.default_rng(seed+epoch);keys=list(groups);rng.shuffle(keys)
        if video_fraction or position_balanced:
            photos=[k for k in keys if not k.startswith('d3c:')];video=[k for k in keys if k.startswith('d3c:')]
            if not photos or (video_fraction and not video):raise ValueError('Photo+video comparison requires VERIFIED data in both sources')
            board_by_id={b['id']:b for b in dataset.boards};pools=[]
            for pool in (photos,video):
                positions={}
                for k in pool:positions.setdefault(board_by_id[k]['position_id'],[]).append(k)
                pools.append(positions)
            mixed=[]
            for _ in range(max(len(keys),2)):
                pool=pools[int(rng.random()<video_fraction)];pos=sorted(pool)[int(rng.integers(len(pool)))];mixed.append(pool[pos][int(rng.integers(len(pool[pos])))])
            keys=mixed
        self.batches=[]
        for key in keys:
            indices=np.asarray(groups[key]);rng.shuffle(indices)
            self.batches.extend(indices[i:i+batch_size].tolist() for i in range(0,len(indices),batch_size))
        if max_batches is not None:
            if not self.batches:raise ValueError('No grouped batches')
            self.batches=(self.batches*((max_batches+len(self.batches)-1)//len(self.batches)))[:max_batches]
    def __iter__(self):return iter(self.batches)
    def __len__(self):return len(self.batches)


def checkpoint_save(path,payload,commit=lambda:None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp);tmp.replace(path);commit()


def contract_checkpoint(state,stage,fingerprint):
    if state.get('class_names')!=list(CLASS_NAMES) or state.get('input_channels')!=4 or state.get('architecture')!='mobilenet_v3_small':raise ValueError('Checkpoint model contract mismatch')
    if state['stage']!=stage or state['dataset_fingerprint']!=fingerprint:raise ValueError('Resume stage/data provenance mismatch')


def experiment_plan(config,experiment):
    """Apply only the preregistered D/E video mixture; no automatic matrix launch."""
    plan=json.loads(json.dumps(config))
    if experiment not in ('B','C','D','E'):raise ValueError('A is a protected historical reference, not a training job')
    plan['experiment']=experiment
    plan['stages']['personal']['video_fraction']=plan['experiment_matrix'][experiment]['video_fraction']
    return plan


def check_transfer(state,stage,experiment):
    if state.get('smoke_only') or state.get('class_names')!=list(CLASS_NAMES) or state.get('input_channels')!=4:
        raise ValueError('Cannot transfer smoke or incompatible checkpoint')
    expected='chesscog' if stage=='chessred' or experiment=='B' else 'chessred'
    if state.get('stage')!=expected:raise ValueError('Wrong staged initialization checkpoint')


def run_stage(manifest,config,stage,output,*,device='cpu',initial=None,resume=None,commit=lambda:None,smoke=False):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=config['smoke'] if smoke else config['stages'][stage]
    if smoke and (settings['max_updates']>4 or settings['max_seconds']>180):raise ValueError('Smoke budget exceeded')
    if not smoke and not config.get('full_training_authorization'):raise ValueError('Full training needs separate explicit authorization')
    plan_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    data_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest();seed=config['seed'];seed_all(seed)
    train=MultiDatasetSquares(manifest,'train',seed=seed,augmentation=config['augmentation']);val=MultiDatasetSquares(manifest,'val',seed=seed)
    if settings.get('required_dataset') and not all(b['dataset'] in settings['required_dataset'] for b in train.boards):raise ValueError('Wrong source for training stage')
    if not smoke and len(train.boards)<settings['minimum_train_boards']:raise ValueError('Only a probe/subset is installed; full stage not ready')
    model=create_classifier(pretrained=initial is None and resume is None,mode='fine_tune').to(device)
    if sum(p.numel() for p in model.parameters())!=1531325:raise ValueError('Backbone architecture changed')
    if initial:
        state=torch.load(initial,map_location=device,weights_only=True)
        check_transfer(state,stage,config.get('experiment','E'))
        model.load_state_dict(state['model_state_dict'],strict=True)
    elif not smoke and not resume and (stage=='personal' or (stage=='chessred' and config.get('experiment','E')!='C')):
        raise ValueError('This stage requires its designated parent checkpoint')
    initial_hash=digest(initial) if initial else None
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=settings.get('lr_gamma',.9))
    use_amp=device.startswith('cuda') and settings.get('precision') in ('fp16','bf16')
    dtype=torch.float16 if settings.get('precision')=='fp16' else torch.bfloat16
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp and dtype==torch.float16)
    counts=np.bincount([o['class_id'] for b,o in train.rows if stage!='personal' or b['dataset']=='chessvision_photos'],minlength=13)
    if not counts.sum():raise ValueError('Personal stage requires fixed photo-derived class weights')
    fixed=settings.get('class_weights')
    weights=torch.tensor(fixed if fixed else (np.sqrt(max(1,counts.sum())/np.maximum(counts,1))).tolist(),dtype=torch.float32,device=device)
    epoch=cursor=updates=0;best=float('inf');bad_epochs=0
    if resume:
        state=torch.load(resume,map_location=device,weights_only=True);contract_checkpoint(state,stage,data_hash)
        if state['plan_hash']!=plan_hash:raise ValueError('Resume config changed')
        model.load_state_dict(state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict']);scheduler.load_state_dict(state['scheduler_state_dict']);scaler.load_state_dict(state['scaler_state_dict'])
        epoch,cursor,updates=state['epoch'],state['next_batch'],state['updates'];best=state['best_loss'];bad_epochs=state['bad_epochs'];restore_rng(state['rng'])
        initial_hash=state.get('initial_checkpoint_sha256')
    max_updates=settings['max_updates'];accum=settings.get('gradient_accumulation',1)
    if accum<1:raise ValueError('Positive accumulation required')
    # Periodic saves occur only at optimizer boundaries (no unsaved partial grads).
    history=state.get('history',[]) if resume else []
    grad_nonzero=False;loading_seconds=0;checkpoint_seconds=0;train_seconds=0;load_start=time.perf_counter()
    first_val=val[0][0]
    first_val_hash=hashlib.sha256(first_val.numpy().tobytes()).hexdigest();loading_seconds+=time.perf_counter()-load_start
    checks=manifest.get('smoke_checks',{})
    if checks and first_val_hash!=checks['normalized_first_val_sha256']:raise ValueError('Cloud B2 normalization/crop parity failed')
    def payload(next_epoch,next_batch):
        return dict(model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),scaler_state_dict=scaler.state_dict(),
          rng=rng_state(),epoch=next_epoch,next_batch=next_batch,updates=updates,best_loss=best,bad_epochs=bad_epochs,history=list(history),
          stage=stage,smoke_only=smoke,plan_hash=plan_hash,dataset_fingerprint=data_hash,initial_checkpoint_sha256=initial_hash,architecture='mobilenet_v3_small',input_channels=4,input_size=224,class_names=list(CLASS_NAMES))
    def save(name,e,b):
        nonlocal checkpoint_seconds
        t=time.perf_counter();checkpoint_save(output/name,payload(e,b),commit);checkpoint_seconds+=time.perf_counter()-t
    stop=False;optimizer.zero_grad(set_to_none=True)
    while epoch<settings['epochs'] and updates<max_updates and not stop:
        train.set_epoch(epoch);batches=BoardBatches(train,settings['batch_size'],seed,epoch,video_fraction=settings.get('video_fraction',0),position_balanced=stage=='personal',max_batches=settings.get('validation_every_updates',0)*accum or None)
        loader=DataLoader(train,batch_sampler=batches.batches[cursor:],num_workers=settings.get('workers',0),generator=torch.Generator().manual_seed(0))
        mode='head_and_stem' if epoch<settings.get('head_epochs',0) else 'fine_tune'
        for p in model.parameters():p.requires_grad=True
        if mode=='head_and_stem':
            for p in model.features.parameters():p.requires_grad=False
            for p in model.features[0].parameters():p.requires_grad=True
        set_training_mode(model,mode);batch_index=cursor
        for images,labels in loader:
            step_start=time.perf_counter();images=images.to(device);labels=labels.to(device)
            with torch.amp.autocast(device_type='cuda',dtype=dtype,enabled=use_amp):
                loss=torch.nn.functional.cross_entropy(model(images),labels,weight=weights)/accum
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward();batch_index+=1
            boundary=batch_index%accum==0 or batch_index==len(batches)
            if boundary:
                scaler.unscale_(optimizer)
                grad_nonzero|=any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.parameters())
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);updates+=1
            if device.startswith('cuda'):torch.cuda.synchronize()
            train_seconds+=time.perf_counter()-step_start
            if boundary and (updates%settings.get('checkpoint_every_updates',100)==0 or updates>=max_updates):save('last.pt',epoch,batch_index)
            if boundary and (updates>=max_updates or time.perf_counter()-started>=settings['max_seconds']):stop=True;break
        cursor=0;model.eval();total=correct=n=0.;confusion=np.zeros((13,13),int)
        with torch.inference_mode():
            for images,labels in DataLoader(val,batch_size=settings['batch_size'],shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(0)):
                images=images.to(device);labels=labels.to(device);logits=model(images)
                total+=float(torch.nn.functional.cross_entropy(logits,labels,weight=weights,reduction='sum'));n+=float(weights[labels].sum())
                pred=logits.argmax(1);np.add.at(confusion,(labels.cpu().numpy(),pred.cpu().numpy()),1)
        val_loss=total/n;history.append(dict(epoch=epoch,updates=updates,validation_loss=val_loss,confusion_matrix=confusion.tolist()))
        if not np.isfinite(val_loss):raise ValueError('Nonfinite validation loss')
        with (output/'metrics.jsonl').open('a') as log:log.write(json.dumps(history[-1])+'\n')
        if val_loss<best:best=val_loss;bad_epochs=0;save('smoke_resume.pt' if smoke else settings['best_name'],epoch,batch_index)
        else:bad_epochs+=1
        completed=batch_index==len(batches)
        if completed:scheduler.step();epoch+=1;cursor=0
        else:cursor=batch_index
        save('last.pt',epoch,cursor)
        if bad_epochs>=settings.get('patience',3):stop=True
    # Reload into a separate object and prove persistence-ready inference parity.
    model.eval()
    with torch.inference_mode():expected=model(first_val[None].to(device)).cpu()
    final=torch.load(output/'last.pt',map_location=device,weights_only=True);contract_checkpoint(final,stage,data_hash)
    reloaded=create_classifier(pretrained=False,mode='fine_tune').to(device).eval();reloaded.load_state_dict(final['model_state_dict'])
    with torch.inference_mode():actual=reloaded(first_val[None].to(device)).cpu()
    if not torch.equal(expected,actual):raise ValueError('Reloaded model logits differ')
    result=dict(stage=stage,smoke_only=smoke,updates=updates,history=history,gradient_nonzero=grad_nonzero,checkpoint_reload_exact=True,
       train_observations=len(train),validation_observations=len(val),class_counts_train=counts.tolist(),class_names=list(CLASS_NAMES),
       model_parameters=sum(p.numel() for p in model.parameters()),normalized_first_val_sha256=first_val_hash,
       elapsed_seconds=time.perf_counter()-started,train_compute_seconds=train_seconds,dataset_first_decode_seconds=loading_seconds,checkpoint_write_commit_seconds=checkpoint_seconds,
       gpu_name=torch.cuda.get_device_name() if device.startswith('cuda') else None,
       peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
       last_checkpoint_sha256=digest(output/'last.pt'),plan_hash=plan_hash,dataset_fingerprint=data_hash)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');commit();return result
