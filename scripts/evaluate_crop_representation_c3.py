"""C.3 fixed-validation scoring and same-shape CPU profiling; no tuning."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.chess.labels import CLASS_NAMES
from src.scanning.square_evidence import classification_metrics
from src.vision.classifier import create_classifier
from src.vision.crop_experiments import CachedCropDataset,CropSpec,packed_input,tensor_from_packed
from src.vision.crop_training_c3 import read,sha,historical_inputs,native_input,source_geometry
from src.vision.grid_geometry import polygon_crop
from scripts.run_crop_representation_c3 import setup_cpu,settings,verify,load_cache,peak_bytes,write


def score(rows,predictions,images):
    truth=np.asarray([r['class_id'] for r in rows]);p=np.asarray(predictions,dtype=int)
    if truth.shape!=p.shape:raise ValueError('Fixed row order required')
    result=classification_metrics(truth,p)
    boards=[]
    for source in sorted({r['source_image'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['source_image']==source]
        if len(indices)!=64 or len({rows[i]['chess_square'] for i in indices})!=64:raise ValueError('Incomplete board')
        boards.append(dict(source_image=source,position_id=rows[indices[0]]['position_id'],
                           errors=int(np.sum(p[indices]!=truth[indices])),exact=bool(np.all(p[indices]==truth[indices]))))
    result.update(boards=len(boards),exact_boards=sum(b['exact'] for b in boards),per_board=boards)
    for name,key in [('position',lambda r:r['position_id']),('orientation',lambda r:images[r['source_image']]['orientation']),
                     ('annotated_angle',lambda r:str(images[r['source_image']]['camera_angle_degrees']))]:
        groups={}
        for group in sorted({key(r) for r in rows}):
            ix=[i for i,r in enumerate(rows) if key(r)==group];sources={rows[i]['source_image'] for i in ix}
            groups[group]=dict(classification_metrics(truth[ix],p[ix]),boards=len(sources),
                               exact_boards=sum(b['exact'] for b in boards if b['source_image'] in sources))
        result['by_'+name]=groups
    known={images[r['source_image']].get('capture_session_id') or images[r['source_image']].get('session_id') for r in rows}
    result['capture_session_status']=dict(known_ids=sorted(k for k in known if k),
        unknown_source_count=len({r['source_image'] for r in rows if not(images[r['source_image']].get('capture_session_id') or images[r['source_image']].get('session_id'))}),
        note='No historical session IDs fabricated; use per-position and nested per-board results.')
    result['king_rook_errors']=dict(white_king_to_rook=result['confusion_matrix'][6][4],black_king_to_rook=result['confusion_matrix'][12][10],
                                    white_rook_to_king=result['confusion_matrix'][4][6],black_rook_to_king=result['confusion_matrix'][10][12])
    result['position_macro_occupied_accuracy']=float(np.mean([g['occupied_accuracy'] for g in result['by_position'].values()]))
    return result


def load_model(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not state['experiment_only'] or state['class_names']!=list(CLASS_NAMES):raise ValueError('Only registered C.3 checkpoints allowed')
    model=create_classifier(pretrained=False,mode='fine_tune');model.load_state_dict(state['model_state_dict'],strict=True)
    return model.eval(),state


def evaluate(root,arm,seed):
    registration=read(root/'registration.json');config=registration['config'];setup_cpu(config)
    verify(registration['epoch_checked_inputs']);status=read(root/'training'/f'{arm}_seed{seed}'/'status.json')
    if status['status']!='complete' or len(status['history'])!=config['epochs']+1:raise ValueError('Matched full arm must complete before scoring')
    dest=root/'evaluation'/f'{arm}_seed{seed}'
    if dest.exists():raise ValueError('Fresh evaluation required')
    dest.mkdir(parents=True);_,allrows,prepared,images,_=historical_inputs(config);rows=allrows['val']
    hashes=read(root/'input_hashes.json')['rows'];cache=load_cache(arm,'val',rows,prepared,images,config,registration,hashes)
    ds=CachedCropDataset(rows,cache,split='val');loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
    results={}
    for label in ('best','last'):
        path=root/'training'/f'{arm}_seed{seed}'/(label+'.pt');model,state=load_model(path)
        expected=status[label+'_checkpoint_sha256'];verify({str(path):expected})
        if state['initial_state_sha256']!=status['initial_state_sha256']:raise ValueError('Initialization identity changed')
        logits=[];start=time.perf_counter()
        with torch.inference_mode():
            for x,_ in loader:logits.append(model(x).numpy())
        logits=np.concatenate(logits);pred=logits.argmax(axis=1)
        if not np.isfinite(logits).all():raise ValueError('Nonfinite evaluation output')
        cm=classification_metrics([r['class_id'] for r in rows],pred)['confusion_matrix']
        if cm!=state['validation']['confusion_matrix']:raise ValueError('Saved epoch predictions did not reproduce')
        np.save(dest/(label+'_logits.npy'),logits)
        results[label]=dict(score(rows,pred,images),epoch=state['epoch'],checkpoint_path=str(path),checkpoint_sha256=expected,
                            checkpoint_bytes=path.stat().st_size,evaluation_seconds=time.perf_counter()-start,
                            logits_path=str(dest/(label+'_logits.npy')),logits_sha256=sha(dest/(label+'_logits.npy')))
        del model,state;gc.collect()
    write(dest/'metrics.json',dict(schema_version='c3_arm_evaluation_v1',arm=arm,seed=seed,results=results,
        config_sha256=registration['config_sha256'],peak_rss_bytes=peak_bytes(),geometry='same approved frozen source polygons; no localization/orientation rerun',
        source_sha256=sha(Path(__file__))))
    print(arm,seed,'best',results['best']['occupied_accuracy'],'FP',results['best']['empty_false_positives'],flush=True)


def profile(root,arm,seed):
    registration=read(root/'registration.json');config=registration['config'];setup_cpu(config)
    _,allrows,prepared,images,_=historical_inputs(config)
    source=allrows['val'][0]['source_image'];rows=[r for r in allrows['val'] if r['source_image']==source]
    dest=root/'profiles'/f'{arm}_seed{seed}.json'
    if dest.exists():raise ValueError('Fresh profile required')
    dest.parent.mkdir(exist_ok=True);source_image=cv2.imread(images[source]['path']);p=prepared[source]
    geometry=source_geometry(p);context=cv2.imread(p['context_path']);valid=cv2.imread(p['mask_path'],0)
    cache_path=Path(registration['cache_reuse']['val']['path']);expected=np.load(cache_path,mmap_mode='r')[:64]
    extraction=[]
    for _ in range(3):
        start=time.perf_counter();packed=[]
        for i,row in enumerate(rows):
            if arm=='taller':v=native_input(source_image,row,p,settings(config),geometry)['packed']
            else:
                crop,_,_,_,local,_=polygon_crop(context,valid,p['grid_geometry']['square_polygons'][row['row']*8+row['col']],200)
                v=packed_input(crop,local,CropSpec('context1_224',224,1.))
                if not np.array_equal(v,expected[i]):raise ValueError('B2 runtime crop packing differs from frozen input')
            packed.append(v)
        extraction.append(time.perf_counter()-start)
    start=time.perf_counter();tensor=torch.from_numpy(np.stack([tensor_from_packed(v) for v in packed]));normalization=time.perf_counter()-start
    model,state=load_model(root/'training'/f'{arm}_seed{seed}'/'best.pt');times=[]
    with torch.inference_mode():
        for repeat in range(7):
            start=time.perf_counter()
            for i in range(0,64,16):model(tensor[i:i+16])
            elapsed=time.perf_counter()-start
            if repeat>=2:times.append(elapsed)
    write(dest,dict(schema_version='c3_cpu_profile_v1',arm=arm,seed=seed,source_image=source,
        frame_selection='first validation source; fixed without prediction knowledge',cpu_threads=4,batch_size=16,
        median_forward_board_seconds=float(np.median(times)),forward_board_seconds=times,
        median_crop_pack_board_seconds=float(np.median(extraction)),normalization_board_seconds=normalization,
        peak_process_rss_bytes=peak_bytes(),model_parameters=sum(p.numel() for p in model.parameters()),
        model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        scope='Standalone process, one 64-square board; excludes decoding, board detection/warp and context cache IO. B2 starts from frozen context; native from original RGB.',
        source_sha256=sha(Path(__file__))))
    print(arm,seed,'profile forward',float(np.median(times)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['evaluate','profile'])
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--arm',choices=['b2','taller'],required=True)
    parser.add_argument('--seed',type=int,required=True);args=parser.parse_args()
    (evaluate if args.action=='evaluate' else profile)(args.run,args.arm,args.seed)
