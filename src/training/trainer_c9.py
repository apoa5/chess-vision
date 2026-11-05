"""One bounded ConvNeXt C9 campaign with complete source coverage and test lock."""
import hashlib,json,math,time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training.dataset_c9 import C9Squares,coverage_batches,audit_splits
from src.training.convnext_c9 import make_model,phase_parameters,set_phase,WEIGHT_SHA,MODEL_NAME
from src.training.multidataset import digest,position_key,placement_labels
from src.training.staged import seed_all,rng_state,restore_rng,checkpoint_save
from src.training.evaluation_c9 import evaluate
from src.training.chesscog_c5 import claim_once,start_utilization_monitor
from src.chess.labels import CLASS_NAMES


def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def settings_guard(config):
    if config['architecture']!=MODEL_NAME or config['representation']!='b2-context1-224-v1' or config['seed']!=42:raise ValueError('Unregistered architecture/profile/seed')
    if config['budget_usd']>5 or config['epochs']>12 or config['max_seconds']>9000 or config['max_updates']>20000:raise ValueError('C9 resource/experiment budget exceeded')
    if config['batch_size'] not in (16,32) or config['gradient_accumulation'] not in (1,2):raise ValueError('Unregistered memory settings')

def class_weights(ds):
    places={position_key(b['placement']):b['placement'] for b in ds.boards};counts=np.bincount([k for f in places.values() for k in placement_labels(f)],minlength=13)
    if (counts==0).any():raise ValueError('Training class coverage missing')
    weight=np.sqrt(counts.sum()/counts);return (weight/weight.mean()).tolist()

def run_training(manifest,config,pretrained,output,*,device='cuda',commit=lambda:None,resume=None):
    settings_guard(config)
    if not config.get('training_authorized'):raise ValueError('Explicit C9 training authorization required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();seed_all(config['seed']);audit_splits(manifest['boards'])
    dh=fingerprint(manifest);ph=fingerprint(config)
    train=C9Squares(manifest,'train',seed=config['seed'],augmentation=config['augmentation']);val=C9Squares(manifest,'val')
    if len(train.boards)<100 or len({position_key(b['placement']) for b in train.boards})<15:raise ValueError('Only a probe dataset is installed')
    weights=class_weights(train);model=make_model(pretrained).to(device)
    optimizer=torch.optim.AdamW(phase_parameters(model,'full',config));scaler=torch.amp.GradScaler('cuda',enabled=False)
    resume_interval_num=resume_interval_den=0.
    updates=epoch=cursor=0;best=float('inf');bad=0;history=[];exposure=Counter();frame_exposure=Counter();class_exposure=Counter();initial_weights_sha=digest(pretrained)
    if resume:
        state=torch.load(resume,map_location=device,weights_only=True)
        if state['plan_hash']!=ph or state['dataset_fingerprint']!=dh or state['class_names']!=list(CLASS_NAMES) or state['pretrained_sha256']!=WEIGHT_SHA:raise ValueError('Resume provenance changed')
        model.load_state_dict(state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict']);scaler.load_state_dict(state['scaler_state_dict']);restore_rng(state['rng'])
        updates,epoch,cursor,best,bad=state['updates'],state['epoch'],state['next_batch'],state['best_loss'],state['bad_epochs'];history=state['history'];exposure=Counter(state['source_exposure']);frame_exposure=Counter(state['frame_exposure']);class_exposure=Counter({int(k):v for k,v in state['class_exposure'].items()});resume_interval_num=state.get('current_interval_loss_numerator',0.);resume_interval_den=state.get('current_interval_loss_denominator',0.)
    def state(e,c,full=True):
        value=dict(model_state_dict=model.state_dict(),architecture=MODEL_NAME,input_channels=4,input_size=224,class_names=list(CLASS_NAMES),representation=config['representation'],pretrained_sha256=initial_weights_sha,
            plan_hash=ph,dataset_fingerprint=dh,updates=updates,epoch=e,next_batch=c,best_loss=best,bad_epochs=bad,history=history,class_weights=weights,source_exposure=dict(exposure),frame_exposure=dict(frame_exposure),class_exposure=dict(class_exposure),current_interval_loss_numerator=interval_num if c else 0.,current_interval_loss_denominator=interval_den if c else 0.)
        if full:value.update(optimizer_state_dict=optimizer.state_dict(),scaler_state_dict=scaler.state_dict(),rng=rng_state(),scheduler_state_dict={'type':'deterministic_two_phase_cosine','updates':updates,'epoch':e,'next_batch':c})
        return value
    stop=False;gradient_nonzero=False;compute=checkpoint_seconds=0.;optimizer.zero_grad(set_to_none=True)
    while epoch<config['epochs'] and updates<config['max_updates'] and not stop:
        train.set_epoch(epoch);batches=coverage_batches(train,config['batch_size'],config['seed'],epoch);phase='head_stem' if epoch<config['head_epochs'] else 'full'
        phase_parameters(model,phase,config);set_phase(model,phase)
        # Sampling weights equalize physical-position contribution while keeping
        # all photo/video passes. Do not divide these constant-within-board
        # factors back out via a per-batch weighted denominator.
        counts=Counter(position_key(train.rows[i%len(train)][0]['placement']) for batch in batches for i in batch);total=sum(counts.values());factors={k:total/(len(counts)*v) for k,v in counts.items()}
        loader=DataLoader(train,batch_sampler=batches[cursor:],num_workers=config['workers'],generator=torch.Generator().manual_seed(config['seed']+epoch))
        interval_num=resume_interval_num if cursor else 0.;interval_den=resume_interval_den if cursor else 0.;resume_interval_num=resume_interval_den=0.;batch_index=cursor
        for x,y in loader:
            ids=batches[batch_index];begin=time.perf_counter();x=x.to(device);y=y.to(device)
            fraction=(epoch-config['head_epochs']+batch_index/len(batches))/max(1,config['epochs']-config['head_epochs'])
            cosine=.1+.9*.5*(1+math.cos(math.pi*min(1,max(0,fraction)))) if phase=='full' else 1.
            for group in optimizer.param_groups:
                kind=group['group_name'].split('_')[0];group['lr']=config[phase+'_lr'].get(kind,0.)*cosine
            factor=torch.tensor([factors[position_key(train.rows[i%len(train)][0]['placement'])] for i in ids],device=device,dtype=torch.float32)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):
                losses=torch.nn.functional.cross_entropy(model(x),y,weight=torch.tensor(weights,device=device),reduction='none');loss=(losses*factor).mean()/config['gradient_accumulation']
            if not torch.isfinite(loss):raise ValueError('Nonfinite C9 training loss')
            loss.backward();batch_index+=1
            for i in ids:
                b,o=train.rows[i%len(train)];exposure[b['dataset']]+=1;frame_exposure[b['id']]+=1;class_exposure[o['class_id']]+=1
            interval_num+=float((losses.detach()*factor).sum());interval_den+=len(y)
            boundary=batch_index%config['gradient_accumulation']==0 or batch_index==len(batches)
            if boundary:
                gradient_nonzero|=any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True);updates+=1
            if device.startswith('cuda'):torch.cuda.synchronize()
            compute+=time.perf_counter()-begin
            if boundary and updates%config['checkpoint_every_updates']==0:
                t=time.perf_counter();checkpoint_save(out/'last.pt',state(epoch,batch_index),commit);checkpoint_seconds+=time.perf_counter()-t
                print('C9 optimizer update',updates,'epoch',epoch,'phase',phase,flush=True)
            if boundary and (updates>=config['max_updates'] or time.perf_counter()-start>=config['max_seconds']):stop=True;break
        validation=evaluate(model,val,device=device,batch_size=config['batch_size'],class_weights=weights);lossvalue=validation['metrics']['macro_position_validation_loss']
        (out/f'validation_epoch_{epoch:02d}.json').write_text(json.dumps(validation['metrics'],indent=2)+'\n')
        history.append(dict(epoch=epoch,phase=phase,updates=updates,training_loss=interval_num/max(1,interval_den),validation_loss=lossvalue,
            overall_accuracy=validation['metrics']['overall_accuracy'],occupied_accuracy=validation['metrics']['occupied_accuracy'],empty_false_positives=validation['metrics']['empty_false_positives'],exact_boards=validation['metrics']['exact_boards']))
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(history[-1])+'\n')
        improved=lossvalue<best
        if improved:
            best=lossvalue;bad=0;checkpoint_save(out/'convnextv2_tiny_best.pt',state(epoch,batch_index,False),commit)
        else:bad+=1
        complete=batch_index==len(batches)
        if complete:epoch+=1;cursor=0
        else:cursor=batch_index
        checkpoint_save(out/'last.pt',state(epoch,cursor),commit)
        print('C9 validation',json.dumps(history[-1]),flush=True)
        if bad>=config['patience'] and phase=='full':stop=True
    # Prove last checkpoint reload before declaring it resumable.
    model.eval();sample=val[0][0][None].to(device)
    with torch.inference_mode():expected=model(sample).cpu()
    last=torch.load(out/'last.pt',map_location='cpu',weights_only=True);reloaded=make_model().to(device).eval();reloaded.load_state_dict(last['model_state_dict'])
    with torch.inference_mode():actual=reloaded(sample).cpu()
    reload_exact=bool(torch.equal(expected,actual));reload_difference=float((expected-actual).abs().max())
    if not torch.allclose(expected,actual,atol=1e-6,rtol=1e-5):raise ValueError('C9 last reload logits differ beyond registered tolerance')
    return dict(updates=updates,completed_coverage_epochs=epoch,partial_training=epoch<config['epochs'],stop_reason='wall_or_update_limit' if stop and bad<config['patience'] else 'early_stopping' if bad>=config['patience'] else 'configured_epochs',
        history=history,source_exposure=dict(exposure),distinct_board_exposure=len(frame_exposure),frame_exposure=dict(frame_exposure),class_exposure=dict(class_exposure),
        dataset_fingerprint=dh,plan_hash=ph,parent_sha256=initial_weights_sha,class_weights=weights,checkpoint_reload_exact=reload_exact,checkpoint_reload_valid=True,checkpoint_reload_max_abs_difference=reload_difference,gradient_nonzero=gradient_nonzero,
        elapsed_seconds=time.perf_counter()-start,train_compute_seconds=compute,checkpoint_seconds=checkpoint_seconds,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
        checkpoint_hashes={p.name:digest(p) for p in out.glob('*.pt')})
