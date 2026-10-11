"""Time real CPU training batches; discard trial weights and leave the test set untouched."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from src.utils.config import load_config
from src.vision.classifier import create_classifier
from src.vision.dataset import ContextSquareDataset
from src.chess.labels import CLASSES


def benchmark(config,batch_size,mode,warmup,batches,threads):
    settings=config['classifier'];seed=settings['seed']
    torch.manual_seed(seed);np.random.seed(seed)
    model=create_classifier(mode=mode)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'])
    manifest=json.loads(settings['dataset_paths']['metadata'].read_text())
    weights=torch.tensor([manifest['split_report']['class_weights'][c.name] for c in CLASSES])
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    dataset=ContextSquareDataset(settings['dataset_paths']['metadata'],'train',input_size=settings['input_size'],
                                 seed=seed,augmentation=settings['augmentation'])
    count=(warmup+batches)*batch_size
    if count>len(dataset):raise ValueError('Requested benchmark exceeds available training samples.')
    indices=np.random.default_rng(seed).permutation(len(dataset))[:count].tolist()
    iterator=iter(DataLoader(Subset(dataset,indices),batch_size=batch_size,num_workers=0))
    times=[];mask_gradient_observed=False
    for step in range(warmup+batches):
        start=time.perf_counter();images,labels=next(iterator);loaded=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits=model(images);loss=criterion(logits,labels)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss.')
        loss.backward()
        gradient=model.features[0][0].weight.grad[:,3]
        mask_gradient_observed |= bool(torch.isfinite(gradient).all() and gradient.abs().sum()>0)
        optimizer.step();end=time.perf_counter()
        if step>=warmup:times.append({'loading_seconds':loaded-start,'compute_seconds':end-loaded,'total_seconds':end-start})
        print(f'{mode} batch={batch_size} step={step+1}/{warmup+batches} {end-start:.3f}s'+(' warmup' if step<warmup else ''),flush=True)
    if not mask_gradient_observed:raise ValueError('Target-mask input received no gradient.')
    validation=ContextSquareDataset(settings['dataset_paths']['metadata'],'val',input_size=settings['input_size'])
    val_iterator=iter(DataLoader(validation,batch_size=batch_size,num_workers=0));model.eval();val_times=[]
    with torch.no_grad():
        for step in range(5):
            start=time.perf_counter();images,labels=next(val_iterator);criterion(model(images),labels);elapsed=time.perf_counter()-start
            if step:val_times.append(elapsed)
    mean=float(np.mean([t['total_seconds'] for t in times]));val_mean=float(np.mean(val_times))
    train_epoch=math.ceil(len(dataset)/batch_size)*mean
    val_epoch=math.ceil(len(validation)/batch_size)*val_mean
    return {'mode':mode,'batch_size':batch_size,'cpu_threads':threads,'warmup_batches':warmup,
            'measured_batches':batches,'training_samples':len(dataset),'validation_samples':len(validation),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'input_shape':[batch_size,4,settings['input_size'],settings['input_size']],
            'mean_batch_seconds':mean,'median_batch_seconds':float(np.median([t['total_seconds'] for t in times])),
            'mean_loading_seconds':float(np.mean([t['loading_seconds'] for t in times])),
            'mean_compute_seconds':float(np.mean([t['compute_seconds'] for t in times])),
            'samples_per_second':batch_size/mean,'validation_batch_seconds':val_mean,
            'estimated_training_epoch_seconds':train_epoch,'estimated_validation_epoch_seconds':val_epoch,
            'estimated_epoch_with_validation_seconds':train_epoch+val_epoch,
            'estimated_configured_epochs_seconds':settings['epochs']*(train_epoch+val_epoch),
            'target_mask_gradient_verified':mask_gradient_observed,'batches':times}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-sizes',type=int,nargs='+',default=[8,16])
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--batches',type=int,default=8)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--modes',nargs='+',choices=['head_and_stem','fine_tune'],default=['head_and_stem','fine_tune'])
    parser.add_argument('--output',type=Path,default=Path('outputs/week2/cpu_training_benchmark.json'))
    args=parser.parse_args()
    if min(args.batch_sizes+[args.threads,args.batches])<1 or args.warmup<0:parser.error('Invalid benchmark size.')
    os.environ.setdefault('TORCH_HOME',str(Path('models/cache').resolve()))
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    config=load_config()
    results=[benchmark(config,size,mode,args.warmup,args.batches,args.threads) for mode in args.modes for size in args.batch_sizes]
    report={'architecture':'mobilenet_v3_small','weights':'IMAGENET1K_V1','device':'cpu','torch_version':torch.__version__,
            'platform':platform.platform(),'seed':config['classifier']['seed'],
            'dataset_fingerprint':json.loads(config['classifier']['dataset_paths']['metadata'].read_text())['split_report']['dataset_fingerprint'],
            'peak_process_memory_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'results':results,'note':'Timing trial only; no checkpoint saved, no test data used, no accuracy claim. Extrapolation excludes startup/downloads and may change with sustained CPU heat or other workloads.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    compact={**report,'results':[{k:v for k,v in r.items() if k!='batches'} for r in results]}
    (config['data']['metadata']/'day11_cpu_benchmark.json').write_text(json.dumps(compact,indent=2)+'\n')
    for r in results:print(f"{r['mode']} batch {r['batch_size']}: {r['mean_batch_seconds']:.3f}s/batch; epoch incl. validation ~{r['estimated_epoch_with_validation_seconds']/60:.1f} minutes")

if __name__=='__main__':main()
