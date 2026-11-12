"""Freeze decisions, measure finalists and audit preservation; offline C11 only."""
import argparse,json,resource,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.benchmark_convnext_c11 import (ROOT,CONFIG,REGISTRY,Corpus,read,value_hash,
    prepare_runtime,parity,metadata)
from src.training.manifest_tools import write_new,labels_to_placement
from src.training.multidataset import digest

def freeze():
    cfg=read(CONFIG);screens={n:read(ROOT/'screens'/f'{n}.json') for n in cfg['candidates']}
    finalists=[]
    for family in ('pytorch','onnx'):
        eligible=[n for n,c in cfg['candidates'].items() if n!='reference' and c['backend'].startswith(family) and screens[n]['top1_disagreements']==0]
        if eligible:finalists.append(min(eligible,key=lambda n:(screens[n]['median_total_seconds'],n)))
    write_new(ROOT/'finalists.json',dict(candidates=finalists,atol=max(1e-6,2*max((screens[n]['max_abs_logit_error'] for n in finalists),default=0)),rtol=1e-5,
        development_only=True,c10_not_used=True,selection='Fastest zero-top1-change screening candidate in each family; full corpus parity still required',
        screen_sha256={n:digest(ROOT/'screens'/f'{n}.json') for n in screens}))

def choose():
    finalists=read(ROOT/'finalists.json');screens={n:read(ROOT/'screens'/f'{n}.json') for n in ['reference']+finalists['candidates']}
    eligible=['reference']
    for n in finalists['candidates']:
        r=read(ROOT/f'{n}_full_parity.json')
        if r['top1_disagreements']==0 and r['board_disagreements']==0 and r['numeric_pass']:eligible.append(n)
    primary=min(eligible,key=lambda n:(screens[n]['median_total_seconds'],n))
    write_new(ROOT/'primary_selection.json',dict(candidate=primary,eligible=eligible,development_only=True,c10_not_used=True,
        rule='Fastest full-development-parity-qualified candidate by registered screening latency. Frozen before C10 regression.'))
    config=dict(screens[primary]['config'])
    # This is a candidate contract, not a recommendation until the regression
    # veto and complete-pipeline preservation checks have passed.
    write_new(Path('configs/evaluation/convnext_c11_candidate_v1.json'),config)

def confirm(candidate):
    if (ROOT/f'{candidate}_confirmation.json').exists():raise ValueError('Confirmation already recorded')
    corpus=Corpus('development');ids=read(REGISTRY)['screening_boards'];rt,imports=prepare_runtime(candidate)
    arrays={bid:corpus.load(bid,preallocate=True)[0] for bid in ids}
    for _ in range(read(CONFIG)['warmup_passes']):rt.predict_logits(arrays[ids[0]])
    samples=[]
    for repeat in range(read(CONFIG)['confirmation_rounds']):
        for bid in ids[repeat%len(ids):]+ids[:repeat%len(ids)]:
            rt.predict_logits(arrays[bid]);samples.append(dict(id=bid,repeat=repeat,**rt.last_timings))
    times=[s['model_forward_seconds'] for s in samples]
    write_new(ROOT/f'{candidate}_confirmation.json',dict(candidate=candidate,samples=samples,median_forward_seconds=float(np.median(times)),
        median_total_seconds=float(np.median([s['total_seconds'] for s in samples])),std_forward_seconds=float(np.std(times)),
        min_forward_seconds=min(times),max_forward_seconds=max(times),**metadata(rt,imports)))

def regression(candidate):
    if (ROOT/f'{candidate}_c10_regression.json').exists() or (ROOT/f'{candidate}_c10_logits.npz').exists():raise ValueError('Regression already recorded')
    # May veto promotion; may not retune or select a different candidate.
    if not (ROOT/'primary_selection.json').exists():raise ValueError('Freeze primary choice before C10 replay')
    corpus=Corpus('c10');rt,imports=prepare_runtime(candidate);old=np.load(ROOT/'reference_c10_logits.npz')['logits'];rows=[];values=[];tol=read(ROOT/'finalists.json')
    for i,bid in enumerate(corpus.ids):
        x,prep=corpus.load(bid,preallocate=True);z=rt.predict_logits(x);values.append(z)
        rows.append(dict(id=bid,**prep,parity=parity(old[i],z,tol['atol'],tol['rtol']),timings=dict(rt.last_timings)))
    np.savez_compressed(ROOT/f'{candidate}_c10_logits.npz',logits=np.stack(values))
    write_new(ROOT/f'{candidate}_c10_regression.json',dict(candidate=candidate,rows=rows,regression_only=True,
        passed=all(not r['parity']['top1_disagreements'] and r['parity']['numeric_within_tolerance'] for r in rows),**metadata(rt,imports)))

def pipeline(candidate):
    """Replay unchanged automatic geometry/crops; never inject verified corners."""
    if (ROOT/f'{candidate}_pipeline.json').exists():raise ValueError('Pipeline timing already recorded')
    from scripts.evaluate_convnext_c10 import timed_preparation,canonical_inputs
    from scripts.run_square_evidence import make_inputs
    from src.vision.convnext_runtime import prepare_board_inputs
    from src.vision.position_pipeline import PreparationError
    cfg=read('configs/evaluation/convnext_independent_c10_v1.json');manifest=read(read(REGISTRY)['regression']['manifest'])
    rt,imports=prepare_runtime(candidate);rows=[];reference=read(ROOT/'reference_c10.json');old=np.load(ROOT/'reference_c10_logits.npz')['logits'];tol=read(ROOT/'finalists.json')
    # Two full sequential passes, no discarded failures. Pass0 is cold image processing,
    # pass1 warm. Session and library initialization are reported separately.
    for repeat in range(2):
        for i,photo in enumerate(manifest['photos']):
            start=time.perf_counter();row=dict(id=photo['id'],repeat=repeat,completed=False)
            try:
                (_,_,turns,prov),image,times=timed_preparation(Path(photo['path']),cfg['geometry']);original=photo['automatic_geometry']
                if turns!=original['rotation_quarters_ccw'] or not np.allclose(prov['source_corners'],original['returned_polygon'],atol=1e-5):raise ValueError('Geometry replay differs')
                t=time.perf_counter();inputs=canonical_inputs(make_inputs(image,original,cfg['crop']));packed=np.stack([p['packed'] for p in inputs]);times['crop_extraction']=time.perf_counter()-t
                if [value_hash(p) for p in packed]!=photo['inputs']['packed_sha256']:raise ValueError('Crop/mask pixels changed')
                t=time.perf_counter();x=prepare_board_inputs(packed);times['normalization']=time.perf_counter()-t
                if value_hash(x)!=reference['rows'][i]['input_sha256']:raise ValueError('Normalized input changed')
                z=rt.predict_logits(x);times.update(rt.last_timings);check=parity(old[i],z,tol['atol'],tol['rtol'])
                t=time.perf_counter();prediction=z.argmax(1);placement=labels_to_placement(prediction.tolist());times['prediction_and_placement_decoding']=time.perf_counter()-t
                # Full FEN uses existing owner-supplied game state, never repaired.
                t=time.perf_counter();fen=placement+' '+' '.join(photo['fen'].split()[1:]);import chess
                chess.Board(fen);times['fen_serialization']=time.perf_counter()-t
                row.update(completed=True,timings=times,parity=check,predicted_placement=placement,geometry_owner_verified=False,
                    source_corners=np.asarray(prov['source_corners']).tolist(),rotation_quarters_ccw=int(turns),
                    source_corners_max_abs_difference=float(np.max(np.abs(np.asarray(prov['source_corners'])-np.asarray(original['returned_polygon'])))),
                    packed_inputs_match=True,normalized_input_sha256=value_hash(x))
            except PreparationError as e:row.update(failure_stage=e.stage,error=str(e))
            row['end_to_end_seconds']=time.perf_counter()-start;rows.append(row)
            print(candidate,photo['id'],repeat,row['end_to_end_seconds'],flush=True)
    write_new(ROOT/f'{candidate}_pipeline.json',dict(candidate=candidate,rows=rows,**metadata(rt,imports)))

def profile():
    if (ROOT/'operator_profile.json').exists():raise ValueError('Operator profile already recorded')
    import torch
    rt,imports=prepare_runtime('reference');x=Corpus('development').load(read(REGISTRY)['screening_boards'][0])[0]
    rt.predict_logits(x)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,profile_memory=True) as prof:rt.predict_logits(x)
    events=sorted(prof.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)
    write_new(ROOT/'operator_profile.json',dict(instrumented_not_latency_benchmark=True,operators=[dict(name=e.key,self_cpu_ms=e.self_cpu_time_total/1000,total_cpu_ms=e.cpu_time_total/1000,calls=e.count,self_cpu_memory_bytes=e.self_cpu_memory_usage) for e in events[:25]]))

def integrity():
    expected=read(ROOT/'protected_inputs.json')['sha256'];bad=[]
    for p,h in expected.items():
        if 'holdout' in p.lower():raise ValueError('Holdout access prohibited')
        if not Path(p).is_file() or digest(p)!=h:bad.append(p)
    result=dict(checked=len(expected),mismatches=bad,final_holdout_access=False,training=False,cloud_jobs=False)
    write_new(ROOT/'integrity.json',result)
    if bad:raise ValueError('Protected artifacts changed: '+str(bad))
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','choose','confirm','regression','pipeline','profile','integrity']);p.add_argument('--candidate');a=p.parse_args()
    if a.action in ('confirm','regression','pipeline'):
        if not a.candidate:p.error('--candidate required')
        globals()[a.action](a.candidate)
    else:globals()[a.action]()
