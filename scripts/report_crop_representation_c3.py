"""C.3 paired seed/position summaries and compact post-hoc crop examples."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from src.chess.labels import CLASS_NAMES
from src.vision.crop_training_c3 import read,sha,historical_inputs,native_input,b2_cache
from scripts.run_crop_representation_c3 import settings,write,verify
from scripts.evaluate_crop_representation_c3 import score


def softmax(z):
    z=np.asarray(z,float);p=np.exp(z-z.max(axis=-1,keepdims=True));return p/p.sum(axis=-1,keepdims=True)


def run(root,output,examples):
    if output.exists() or examples.exists():raise ValueError('Fresh public report outputs required')
    registration=read(root/'registration.json');config=registration['config'];manifest,rows,prepared,images,_=historical_inputs(config)
    val=rows['val'];truth=np.asarray([r['class_id'] for r in val]);models=[];predictions={};logits={};statuses={}
    for seed in config['seeds']:
        for arm in ('b2','taller'):
            name=f'{arm}_seed{seed}';status=read(root/'training'/name/'status.json');evaluation=read(root/'evaluation'/name/'metrics.json')
            if status['status']!='complete' or len(status['history'])!=config['epochs']+1:raise ValueError('Incomplete matched training')
            metrics=evaluation['results']['best'];verify({metrics['logits_path']:metrics['logits_sha256']})
            z=np.load(metrics['logits_path']);p=z.argmax(axis=1)
            if metrics['confusion_matrix']!=score(val,p,images)['confusion_matrix']:raise ValueError('Metrics did not replay')
            predictions[(seed,arm)]=p;logits[(seed,arm)]=z;statuses[(seed,arm)]=status
            models.append(dict(arm=arm,seed=seed,best=metrics,last=evaluation['results']['last'],
                               training_seconds=status['elapsed_seconds'],peak_training_rss_bytes=status['peak_rss_bytes'],
                               initial_state_sha256=status['initial_state_sha256'],history=status['history'],
                               profile=read(root/'profiles'/(name+'.json'))))
    paired=[]
    for seed in config['seeds']:
        a,b=(statuses[(seed,arm)] for arm in ('b2','taller'))
        if a['initial_state_sha256']!=b['initial_state_sha256'] or [h['sample_order_sha256'] for h in a['history']]!=[h['sample_order_sha256'] for h in b['history']]:
            raise ValueError('Unmatched initialization or epoch orders')
        ap,bp=(predictions[(seed,arm)] for arm in ('b2','taller'))
        changes=[]
        for i,r in enumerate(val):
            if ap[i]!=bp[i]:changes.append(dict(crop_id=r['crop_id'],source_image=r['source_image'],position_id=r['position_id'],square=r['chess_square'],
                true_class_id=int(truth[i]),b2_class_id=int(ap[i]),taller_class_id=int(bp[i]),
                corrected=bool(ap[i]!=truth[i] and bp[i]==truth[i]),regressed=bool(ap[i]==truth[i] and bp[i]!=truth[i])))
        paired.append(dict(seed=seed,corrected=int(((ap!=truth)&(bp==truth)).sum()),regressed=int(((ap==truth)&(bp!=truth)).sum()),
                           both_wrong=int(((ap!=truth)&(bp!=truth)).sum()),net_correct=int((bp==truth).sum()-(ap==truth).sum()),changes=changes))
    aggregate={}
    for arm in ('b2','taller'):
        subset=[m for m in models if m['arm']==arm]
        aggregate[arm]={key:dict(mean=float(np.mean([m['best'][key] for m in subset])),minimum=float(min(m['best'][key] for m in subset)),maximum=float(max(m['best'][key] for m in subset)))
                        for key in ('overall_accuracy','occupied_accuracy','empty_false_positives','empty_false_positive_rate','occupied_to_empty_misses','exact_boards','position_macro_occupied_accuracy')}
    historical_path=Path('outputs/evaluation/crop_b2_20261018_run1/evaluation/context1_224_adapted/val_probabilities.npy')
    historical=score(val,np.load(historical_path).argmax(axis=1),images)
    # Post-hoc diagnostics are not checkpoint/crop/seed selection.
    seed=config['seeds'][0];a,b=(predictions[(seed,arm)] for arm in ('b2','taller'));chosen=[]
    for label,predicate in [('king_corrected',lambda i:truth[i] in (6,12) and a[i]!=truth[i] and b[i]==truth[i]),
                            ('king_regressed',lambda i:truth[i] in (6,12) and a[i]==truth[i] and b[i]!=truth[i]),
                            ('empty_false_positive_introduced',lambda i:truth[i]==0 and a[i]==0 and b[i]!=0),
                            ('empty_false_positive_removed',lambda i:truth[i]==0 and a[i]!=0 and b[i]==0),
                            ('rook_corrected',lambda i:truth[i] in (4,10) and a[i]!=truth[i] and b[i]==truth[i]),
                            ('rook_regressed',lambda i:truth[i] in (4,10) and a[i]==truth[i] and b[i]!=truth[i]),
                            ('corrected_other',lambda i:a[i]!=truth[i] and b[i]==truth[i]),
                            ('persistent_error',lambda i:a[i]!=truth[i] and b[i]!=truth[i])]:
        eligible=[i for i in range(len(val)) if predicate(i) and i not in [j for _,j in chosen]]
        if eligible:
            i=min(eligible,key=lambda i:(images[val[i]['source_image']]['camera_angle_degrees'],val[i]['crop_id']))
            chosen.append((label,i))
    examples.mkdir(parents=True);shutil.copytree(root/'pretraining_examples',examples/'pretraining')
    cache=np.load(b2_cache('val')[0],mmap_mode='r');input_hashes=read(root/'input_hashes.json')['rows'];sections=[];case_records=[]
    for label,i in chosen:
        row=val[i];source=row['source_image'];image=cv2.imread(images[source]['path']);native=native_input(image,row,prepared[source],settings(config));packets=[cache[i],native['packed']]
        import hashlib
        for arm,packed in zip(('b2','taller'),packets):
            if hashlib.sha256(packed.tobytes()).hexdigest()!=input_hashes[row['crop_id']][arm+'_sha256']:raise ValueError('Diagnostic pixels changed')
        panels=[]
        for arm,p in zip(('B2','Taller'),packets):
            preview=cv2.cvtColor(p[:3].transpose(1,2,0),cv2.COLOR_RGB2BGR).copy()
            contours,_=cv2.findContours(p[3].copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(preview,contours,-1,(0,255,0),1)
            panel=np.full((270,240,3),250,np.uint8);panel[30:254,8:232]=preview
            cv2.putText(panel,arm,(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);panels.append(panel)
        photo=image.copy();cv2.polylines(photo,[np.rint(native['source_target_polygon']).astype(np.int32)],True,(0,255,0),2)
        factor=min(1.,400/max(photo.shape[:2]));photo=cv2.resize(photo,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA)
        name=row['crop_id'];cv2.imwrite(str(examples/(name+'.jpg')),cv2.hconcat(panels),[cv2.IMWRITE_JPEG_QUALITY,93]);cv2.imwrite(str(examples/(name+'_source.jpg')),photo,[cv2.IMWRITE_JPEG_QUALITY,86])
        seed_lines=[];seed_records={}
        for s in config['seeds']:
            seed_records[s]={arm:dict(predicted_class_id=int(predictions[(s,arm)][i]),probabilities=softmax(logits[(s,arm)][i]).tolist()) for arm in ('b2','taller')}
            seed_lines.append(f'Seed {s}: B2 {CLASS_NAMES[predictions[(s,"b2")][i]]}; taller {CLASS_NAMES[predictions[(s,"taller")][i]]}')
        cases=html.escape(' · '.join(seed_lines));records=dict(category=label,crop_id=name,position_id=row['position_id'],source_image=source,square=row['chess_square'],true_class_id=int(truth[i]),seeds=seed_records,
            orientation=images[source]['orientation'],annotated_angle=images[source]['camera_angle_degrees'],height_mode=native['height_proxy']['mode'])
        case_records.append(records)
        sections.append(f'<section><h2>{label}: {name} {row["chess_square"]}, label {CLASS_NAMES[truth[i]]}</h2><p>{cases}</p><p>Historical angle {records["annotated_angle"]}°, {records["orientation"]}; {records["height_mode"]}. Green outline is target ground square, not silhouette.</p><img style="max-height:400px;max-width:40%" src="{name}_source.jpg"><img style="max-width:58%" src="{name}.jpg"><details><summary>Uncalibrated 13-class softmax</summary><pre>{html.escape(json.dumps(seed_records,indent=2))}</pre></details></section>')
    (examples/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>C3 matched crop comparison</title><style>body{font:15px system-ui;max-width:1150px;margin:auto;padding:20px}section{border-top:1px solid #aaa;padding:12px 0}img{object-fit:contain}pre{overflow:auto}</style><h1>C.3 matched MobileNetV3 crop experiment</h1><p><a href="../crop_representation_c3.md">All results and limitations</a> · <a href="pretraining/index.html">Fixed pretraining representation audit</a>. These validation examples are selected post-hoc from seed42 errors/corrections; seed43 predictions are also shown. No examples were excluded from aggregate scoring. No silhouette/contact labels were invented.</p>'+''.join(sections))
    all_stats=read(root/'input_hashes.json')['statistics']
    compact_stats={split:{key:np.percentile([r[key] for r in all_stats if r['split']==split],[0,50,100]).tolist()
                          for key in ('target_width_b2','target_width_taller','valid_fraction')} for split in ('train','val')}
    result=dict(schema_version='crop_representation_c3_results_v1',scope='matched_initialization_two_seeds_three_development_validation_positions',
        registration_path=str(root/'registration.json'),registration_sha256=sha(root/'registration.json'),config_sha256=registration['config_sha256'],
        validation_crops=1600,validation_boards=25,validation_positions=3,training_positions=12,training_crops=7168,
        capture_session_status='Historical IDs unavailable; no independent-session claim',models=models,seed_descriptive_summary=aggregate,paired_changes=paired,
        historical_B2_reference=dict(metrics=historical,probabilities_path=str(historical_path),probabilities_sha256=sha(historical_path),
            not_matched_control=True,note='Longer historical chess training/adaptation and different selection; no new inference.'),
        diagnostic_cases=case_records,geometry_preflight=dict(height_modes=registration['height_modes'],failed_targets=0,excluded_targets=0,b2_parity_checks=registration['b2_exact_pixel_parity_checks']),
        crop_statistics_min_median_max=compact_stats,input_hashes_path=str(root/'input_hashes.json'),input_hashes_sha256=registration['input_hashes_sha256'],
        no_test_or_D2_D3c_selection=True,no_production_promotion=True,
        source_sha256={p:sha(p) for p in ['scripts/report_crop_representation_c3.py','scripts/evaluate_crop_representation_c3.py']})
    output.parent.mkdir(parents=True,exist_ok=True);write(output,result)
    csv_path=output.with_suffix('.csv')
    with csv_path.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['arm','seed','class_id','class_name','support','precision','recall'])
        for m in models:
            for r in m['best']['per_class']:writer.writerow([m['arm'],m['seed'],r['class_id'],CLASS_NAMES[r['class_id']],r['support'],r['precision'],r['recall']])
    print(json.dumps({k:v for k,v in aggregate.items()},indent=2));print('Paired:',[{k:v for k,v in p.items() if k!='changes'} for p in paired])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--examples',type=Path,required=True)
    args=parser.parse_args();run(args.run,args.output,args.examples)
