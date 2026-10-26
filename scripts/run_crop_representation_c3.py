"""C.3 preregistered historical train/val preparation and matched CPU arms."""
import argparse
from collections import Counter
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.chess.labels import CLASS_NAMES, LABEL_VERSION
from src.vision.crop_training_c3 import (sha,read,historical_inputs,source_geometry,native_input,
    b2_cache,make_native_cache,TrackedDataset,epoch_metrics,checkpoint_key,state_hash)
from src.vision.input_representations import RepresentationSettings
from src.vision.crop_experiments import CropSpec,packed_input

CONFIG=Path('configs/experiments/crop_representation_c3_v1.json')
SOURCE_FILES=['src/vision/crop_training_c3.py','scripts/run_crop_representation_c3.py',str(CONFIG),
              'docs/evaluation/crop_representation_c3_design_v1.md','src/vision/input_representations.py',
              'src/vision/classifier.py','src/vision/training.py','src/vision/crop_experiments.py']


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def verify(hashes):
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+str(path))


def setup_cpu(config):
    import torch
    torch.set_num_threads(config['cpu_threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);cv2.setNumThreads(1)
    os.environ['TORCH_HOME']=str(Path('models/cache').resolve())


def settings(config):
    b=config['arms']['taller']
    return RepresentationSettings(size=b['input_size'],height_squares=b['height_squares'],raw_margin_squares=b['raw_margin_squares'])


def peak_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def prepare(root):
    if root.exists():raise ValueError('Fresh C.3 run directory required')
    config=read(CONFIG);manifest,rows,prepared,images,leakage=historical_inputs(config)
    cache_provenance=read('outputs/evaluation/crop_b2_20261018_run1/provenance.json')
    if cache_provenance['manifest_sha256']!=sha(config['manifest']):raise ValueError('B2 cache row ordering manifest changed')
    checkpoint=Path('models/cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth')
    if not checkpoint.is_file():raise ValueError('Required existing ImageNet initialization unavailable; no silent download')
    root.mkdir(parents=True);(root/'pretraining_examples').mkdir()
    inputs={p:sha(p) for p in SOURCE_FILES+[config[k] for k in ('manifest','prepared_boards','images','positions','reviews')]+[str(checkpoint)]}
    sources=sorted({r['source_image'] for rs in rows.values() for r in rs})
    raw_hashes={images[n]['path']:prepared[n]['source_sha256'] for n in sources};verify(raw_hashes)
    caches={};hashes={};modes=Counter();stats=[];visuals=[]
    # Registered diagnostic selection uses training rows, labels/metadata only;
    # never new model outcomes. It does not alter sample eligibility.
    candidates=sorted(rows['train'],key=lambda r:(images[r['source_image']]['camera_angle_degrees'],r['crop_id']))
    selected={}
    for name,predicate in [
        ('white_king',lambda r:r['class_id']==6),('black_king',lambda r:r['class_id']==12),
        ('edge_rook',lambda r:r['class_id'] in (4,10) and (r['row'] in (0,7) or r['col'] in (0,7))),
        ('empty_near_piece',lambda r:r['class_id']==0 and r['row'] in (1,6)),
        ('knight',lambda r:r['class_id'] in (2,8)),
        ('overhead_king',lambda r:r['class_id'] in (6,12) and images[r['source_image']]['camera_angle_degrees']==90)]:
        chosen=next((r for r in candidates if predicate(r)),None)
        if chosen:selected[chosen['crop_id']]=name
    for orientation in ('white_near_camera','black_near_camera','white_left','white_right'):
        chosen=next((r for r in candidates if r['class_id'] in (6,12) and images[r['source_image']]['orientation']==orientation),None)
        if chosen:selected[chosen['crop_id']]='orientation_'+orientation
    start=time.perf_counter();parity=0
    for split,rs in rows.items():
        cache,meta=b2_cache(split);verify({str(cache):meta['sha256']});inputs[str(cache)]=meta['sha256']
        caches[split]=dict(path=str(cache),**meta);array=np.load(cache,mmap_mode='r')
        if array.shape!=(len(rs),4,224,224):raise ValueError('B2 cache shape changed')
        source=None;image=None;geometry=None
        for i,row in enumerate(rs):
            if source!=row['source_image']:
                source=row['source_image'];image=cv2.imread(images[source]['path']);geometry=source_geometry(prepared[source])
                if image is None:raise ValueError('Missing native source')
            native=native_input(image,row,prepared[source],settings(config),geometry);p=native['packed'];a=np.asarray(array[i])
            if i%64 in (0,63) or row['crop_id'] in selected:
                verify({row['crop_path']:row['crop_sha256']})
                expected=packed_input(cv2.imread(row['crop_path']),row['target_polygon'],CropSpec('context1_224',224,1.))
                if not np.array_equal(a,expected):raise ValueError('B2 cached representation mismatch')
                parity+=1
            hashes[row['crop_id']]=dict(b2_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                taller_sha256=hashlib.sha256(p.tobytes()).hexdigest(),height_mode=native['height_proxy']['mode'])
            modes[(split,native['height_proxy']['mode'])]+=1
            stats.append(dict(crop_id=row['crop_id'],split=split,target_width_b2=float(np.sqrt(a[3].sum())),
                              target_width_taller=float(np.sqrt(p[3].sum())),valid_fraction=native['valid_fraction']))
            if row['crop_id'] in selected:
                panels=[]
                for name,v in [('B2 tighter 224',a),('Taller native 224',p)]:
                    preview=cv2.cvtColor(v[:3].transpose(1,2,0),cv2.COLOR_RGB2BGR).copy()
                    contours,_=cv2.findContours(v[3].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(preview,contours,-1,(0,255,0),1)
                    panel=np.full((270,240,3),250,np.uint8);panel[30:254,8:232]=preview
                    cv2.putText(panel,name,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(10,10,10),1);panels.append(panel)
                path=root/'pretraining_examples'/(row['crop_id']+'.jpg');cv2.imwrite(str(path),cv2.hconcat(panels),[cv2.IMWRITE_JPEG_QUALITY,92])
                source_preview=image.copy();cv2.polylines(source_preview,[np.rint(native['source_target_polygon']).astype(np.int32)],True,(0,255,0),2)
                sc=min(1.,520/max(source_preview.shape[:2]));source_preview=cv2.resize(source_preview,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
                rawpath=root/'pretraining_examples'/(row['crop_id']+'_source.jpg');cv2.imwrite(str(rawpath),source_preview,[cv2.IMWRITE_JPEG_QUALITY,86])
                visuals.append(dict(crop_id=row['crop_id'],square=row['chess_square'],class_name=row['class_name'],source=source,
                    case=selected[row['crop_id']],height_proxy=native['height_proxy'],orientation=images[source]['orientation'],
                    annotated_angle=images[source]['camera_angle_degrees'],split=split,packed=path.name,raw=rawpath.name))
        del array
    write(root/'input_hashes.json',dict(schema_version='c3_input_hashes_v1',rows=hashes,statistics=stats))
    write(root/'visual_cases.json',visuals)
    sections=''.join(f'<section><h2>{html.escape(v["case"])}: {v["crop_id"]} {v["square"]} {v["class_name"]}</h2><p>{v["orientation"]}, historical angle {v["annotated_angle"]}°, {v["height_proxy"]["mode"]}. Training only; no model prediction.</p><img height="350" src="{v["raw"]}"><img src="{v["packed"]}"></section>' for v in visuals)
    (root/'pretraining_examples/index.html').write_text('<!doctype html><meta charset="utf-8"><title>C3 pretraining representation checks</title><style>body{font:16px system-ui;max-width:1100px;margin:auto}section{border-top:1px solid #aaa;padding:12px}img{max-width:100%;object-fit:contain}</style><h1>C.3 pretraining representation audit</h1><p>Fixed C.1 height2/margin.35. Ground-square masks, not silhouettes. No source image/geometry edited; no validation-performance-driven redesign. These approved boards do not have individually verified silhouette annotations. Image-up fallback is a heuristic, especially under roll. Context can contain neighbors and black source padding.</p>'+sections)
    small={k:v for k,v in inputs.items() if not k.endswith('.npy')}
    registration=dict(schema_version='crop_representation_c3_registration_v1',config=config,config_sha256=sha(CONFIG),
        immutable_inputs=inputs,epoch_checked_inputs=small,raw_source_sha256=raw_hashes,input_hashes_sha256=sha(root/'input_hashes.json'),
        group_assignment=manifest['split_report']['position_assignment'],leakage=leakage,
        train_sources=112,validation_sources=25,train_crops=7168,validation_crops=1600,
        cache_reuse=caches,initialization_path=str(checkpoint),initialization_sha256=sha(checkpoint),
        height_modes=[dict(split=s,mode=m,targets=n) for (s,m),n in sorted(modes.items())],
        b2_exact_pixel_parity_checks=parity,geometry_failures=0,excluded_samples=0,
        preparation_seconds=time.perf_counter()-start,selection_outcomes_inspected=False,
        native_cache_persisted=False,original_session_ids_missing=leakage['unknown_session_sources'])
    write(root/'registration.json',registration)
    print('Prepared',len(hashes),'paired inputs; modes',registration['height_modes'],'parity',parity,flush=True)


def load_cache(arm,split,rows,prepared,images,config,registration,hashes):
    if arm=='b2':return np.load(registration['cache_reuse'][split]['path'],mmap_mode='r')
    return make_native_cache(rows,prepared,images,settings(config),hashes)


def train(root,arm,seed):
    import torch
    from torch.utils.data import DataLoader
    from src.vision.classifier import create_classifier
    from src.vision.training import run_epoch
    registration=read(root/'registration.json');config=registration['config'];setup_cpu(config)
    if arm not in ('b2','taller') or seed not in config['seeds']:raise ValueError('Unregistered arm/seed')
    verify(registration['epoch_checked_inputs']);verify(registration['raw_source_sha256'])
    manifest,rows,prepared,images,_=historical_inputs(config)
    dest=root/'training'/f'{arm}_seed{seed}'
    if dest.exists():raise ValueError('Fresh isolated arm required')
    dest.mkdir(parents=True);start=time.perf_counter();hashes=read(root/'input_hashes.json')['rows']
    caches={s:load_cache(arm,s,rows[s],prepared,images,config,registration,hashes) for s in ('train','val')}
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=create_classifier(pretrained=True,mode='head_and_stem')
    initial_hash=state_hash(model)
    datasets={s:TrackedDataset(rows[s],caches[s],split=s,seed=seed,augmentation=config['augmentation'] if s=='train' else {}) for s in ('train','val')}
    loaders={s:DataLoader(datasets[s],batch_size=config['batch_size'],shuffle=s=='train',num_workers=0,
                         generator=torch.Generator().manual_seed(seed) if s=='train' else torch.Generator().manual_seed(0)) for s in ('train','val')}
    weights=torch.tensor([manifest['split_report']['class_weights'][name] for name in CLASS_NAMES],dtype=torch.float32)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
    history=[];best=None;mode='head_and_stem'
    status=dict(schema_version='c3_training_status_v1',status='running',arm=arm,seed=seed,initial_state_sha256=initial_hash,
                config_sha256=registration['config_sha256'],history=history,model_parameter_count=sum(p.numel() for p in model.parameters()),
                samples_per_epoch=len(rows['train']),validation_samples=len(rows['val']))
    def save_checkpoint(path,epoch,validation):
        torch.save(dict(model_state_dict=model.state_dict(),epoch=epoch,mode=mode,validation=validation,
            architecture='mobilenet_v3_small',input_channels=4,input_size=224,label_version=LABEL_VERSION,
            class_names=list(CLASS_NAMES),experiment_only=True,arm=arm,seed=seed,
            config_sha256=registration['config_sha256'],initial_state_sha256=initial_hash,
            manifest_sha256=sha(config['manifest'])),path)
    try:
        for epoch in range(config['epochs']+1):
            verify(registration['epoch_checked_inputs']);epoch_start=time.perf_counter();tm=None;order=None
            if epoch==config['head_epochs']+1:
                mode='fine_tune'
                for p in model.parameters():p.requires_grad=True
                optimizer=torch.optim.Adam(model.parameters(),lr=config['fine_tune_learning_rate'],betas=tuple(config['betas']),eps=config['eps'],weight_decay=config['weight_decay'])
            if epoch:
                datasets['train'].begin_epoch(epoch-1)
                def progress(step,total):
                    status.update(epoch=epoch,step=step,total_steps=total,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_bytes())
                    write(dest/'status.json',status)
                    print(arm,seed,'epoch',epoch,mode,step,'/',total,flush=True)
                    if peak_bytes()>config['resource_limits']['max_peak_rss_bytes']:raise RuntimeError('C3 memory budget exceeded')
                    if time.perf_counter()-start>config['resource_limits']['max_seconds_per_arm']:raise RuntimeError('C3 arm wall-clock budget exceeded')
                tm=epoch_metrics(run_epoch(model,loaders['train'],weights,optimizer=optimizer,mode=mode,progress=progress));order=datasets['train'].order_sha256()
            validation=epoch_metrics(run_epoch(model,loaders['val'],weights))
            key=checkpoint_key(validation,epoch)
            history.append(dict(epoch=epoch,mode=mode,training=tm,validation=validation,sample_order_sha256=order,
                                epoch_seconds=time.perf_counter()-epoch_start,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_bytes()))
            if best is None or key<best:
                best=key;save_checkpoint(dest/'best.pt',epoch,validation);status.update(best_epoch=epoch,best_validation=validation)
            status.update(epoch=epoch,elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_bytes())
            write(dest/'status.json',status)
            print('VALIDATION',arm,seed,epoch,'occupied',validation['occupied_accuracy'],'FP',validation['empty_false_positives'],'loss',validation['loss'],flush=True)
        save_checkpoint(dest/'last.pt',config['epochs'],validation)
        status.update(status='complete',elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_bytes(),
                      best_checkpoint_sha256=sha(dest/'best.pt'),last_checkpoint_sha256=sha(dest/'last.pt'))
        write(dest/'status.json',status)
    except BaseException as error:
        status.update(status='blocked',error=str(error),elapsed_seconds=time.perf_counter()-start,peak_rss_bytes=peak_bytes())
        write(dest/'status.json',status);raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','train']);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--arm',choices=['b2','taller']);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.run)
    else:
        if args.arm is None or args.seed is None:parser.error('Training requires --arm and --seed')
        train(args.run,args.arm,args.seed)
