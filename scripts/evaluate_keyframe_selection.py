"""Compare D.1 strategies without reading piece labels or running recognition."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read, write, check_hashes
from src.scanning.native_pilot import sha256_file
from src.scanning.video_keyframes import (STRATEGIES, board_features, geometry_state,
    select_keyframes, selection_metrics, validate_settings, view_distance)


def binding(f):
    return {k:f[k] for k in ('session_id','frame_id','pts','time_base','source_video_sha256','derivative_sha256')}


def review_map(path):
    if path is None:return {}
    obj=read(path)
    if obj['schema_version']!='offline_keyframe_geometry_reviews_v1':raise ValueError('Unknown review schema')
    output={}
    for r in obj['reviews']:
        b=r['binding'];key=(b['source_video_sha256'],b['frame_id'],r['policy'])
        if key in output:raise ValueError('Duplicate geometry review')
        output[key]=r
    return output


def describe(frame, g, supplied_binding, pointer, policy, settings, reviews):
    started=time.perf_counter()
    if sha256_file(frame['path'])!=frame['derivative_sha256']:raise ValueError('Image changed')
    image=cv2.imread(frame['path'])
    if image is None or list(image.shape[:2])!=[frame['height'],frame['width']]:raise ValueError('Wrong image dimensions')
    if Fraction(frame['pts'])*Fraction(*frame['time_base'])!=Fraction(*frame['pts_time']):raise ValueError('PTS mismatch')
    review=reviews.get((frame['source_video_sha256'],frame['frame_id'],policy))
    state,reasons=geometry_state(frame,g,supplied_binding,review)
    r={k:frame[k] for k in ['session_id','frame_id','pts','time_base','pts_time','timestamp_ms',
        'relative_timestamp_ms','path','width','height','source_video_sha256','derivative_sha256','decoded_pixel_sha256']}
    r.update(policy=policy,geometry_completed=g['completed'],geometry_state=state,
             orientation_validated=g['completed'] and g.get('rotation_quarters_ccw') in (0,1,2,3),
             geometry_reference=pointer,geometry_reference_sha256=sha256_file(pointer),
             geometry_failure_stage=g['failure_stage'],geometry_error=g.get('error'),
             recovery_attempted=g['recovery_attempted'],recovery_reason=g.get('recovery_reason'),
             geometry_seconds=g['elapsed_seconds'],unusable_reasons=reasons,eligible=False,
             review_annotation=review,quality_score=None,quality_rejection_reasons=[],
             predicted_orientation=None,view_descriptor=None,appearance_dhash=None,
             elevation_degrees=None,visibility='unknown',geometry_evidence_source=g.get('provenance',{}).get('orientation_source'))
    if state in ('completed_unverified','visually_verified'):
        try:
            r.update(board_features(image,g,settings))
            r['unusable_reasons']+=r['quality_rejection_reasons']
            r['eligible']=not r['quality_rejection_reasons']
        except ValueError as error:
            r.update(geometry_state='unresolved',unusable_reasons=['selector_invalid_outer_boundary',str(error)])
    r['feature_seconds']=time.perf_counter()-started
    return r


def apparent_motion(rows):
    """Sparse image-space motion diagnostic, not tracking drift or rejection ground truth."""
    previous=None
    for r in sorted(rows,key=lambda r:r['relative_timestamp_ms']):
        r['adjacent_view_proxy_change']=None;r['apparent_board_centre_speed_per_second']=None
        if not r['eligible']:continue
        if previous is not None:
            dt=(r['relative_timestamp_ms']-previous['relative_timestamp_ms'])/1000
            if 0<dt<=7:
                r['adjacent_view_proxy_change']=view_distance(r,previous)
                a=np.mean(r['outer_quad'],axis=0)/[r['width'],r['height']]
                b=np.mean(previous['outer_quad'],axis=0)/[previous['width'],previous['height']]
                r['apparent_board_centre_speed_per_second']=float(np.linalg.norm(a-b)/dt)
        previous=r


def primary_dataset(run, manifest, settings, reviews):
    sessions=[]
    for s in manifest['sessions']:
        records={p:[] for p in ['legacy','marked-board']}
        for f in s['candidates']:
            p=run/'geometry'/f['session_id']/f['frame_id']/'pair.json'
            if not p.exists():raise ValueError('Missing geometry pair: '+str(p))
            pair=read(p)
            if pair['geometry_config_sha256']!=manifest['geometry_config_sha256']:raise ValueError('Changed geometry config')
            if pair['evaluator_sha256']!=manifest['critical_sha256']['scripts/evaluate_pilot_geometry.py']:raise ValueError('Changed evaluator')
            for policy,g in pair['policies'].items():records[policy].append(describe(f,g,pair['binding'],str(p),policy,settings,reviews))
        for rows in records.values():apparent_motion(rows)
        meta={k:s[k] for k in ['capture_session_id','position_id','position_group_id','independence_group_id']}
        sessions.append(dict(meta,source_video=s['source_video'],duration_seconds=s['duration_seconds'],policies=records))
    return dict(name='primary_3s_high',candidate_generation='Regular native source-relative temporal grid; not A.3 quality preselection.',sessions=sessions)


def paired_datasets(manifest, settings, reviews):
    catalog=read(manifest['source_catalog']);datasets=[]
    for role,native_path,report_path in [
        ('paired_frozen_high',catalog['primary_manifest'],'outputs/scanning/pilot_01/hd_geometry_final_run1/results.json'),
        ('paired_frozen_compressed',catalog['comparison_manifest'],'outputs/scanning/pilot_01/geometry_run1/results.json')]:
        native=read(native_path);report=read(report_path)
        if (report['native_manifest_sha256']!=sha256_file(native_path) or
            report['config_sha256']!=manifest['geometry_config_sha256'] or
            report['source_sha256']!=manifest['critical_sha256']['scripts/evaluate_pilot_geometry.py']):
            raise ValueError('Cached evaluation does not match registered source/config/code')
        lookup={(f['session_id'],f['candidate_number']):f for f in report['frames']};sessions=[]
        for s in native['sessions']:
            records={p:[] for p in ['legacy','marked-board']}
            for n,c in enumerate(s['candidates'],1):
                f=dict(c,session_id=s['capture_session_id']);old=lookup[(s['capture_session_id'],n)]
                if any(old[k]!=f[k] for k in ['frame_id','pts','time_base','derivative_sha256']):raise ValueError('Mismatched cached frame')
                for policy,g in old['policies'].items():records[policy].append(describe(f,g,binding(f),report_path,policy,settings,reviews))
            for rows in records.values():apparent_motion(rows)
            meta={k:s[k] for k in ['capture_session_id','position_id','position_group_id','independence_group_id']}
            sessions.append(dict(meta,source_video=s['source_video'],duration_seconds=s['source_video']['container_duration_seconds'],policies=records))
        datasets.append(dict(name=role,candidate_generation='Previous 12 A.3 quality-preselected timestamps/session. Paired secondary analysis only; budgets >12 saturate.',sessions=sessions))
    return datasets


def selected_manifest(dataset,policy,strategy,budget,selections,manifest,settings):
    sessions=[]
    for s,result in zip(dataset['sessions'],selections):
        lookup={r['frame_id']:r for r in s['policies'][policy]}
        chosen=[]
        for fid in result['selected']:
            r=lookup[fid]
            chosen.append({k:r[k] for k in ['frame_id','pts','time_base','pts_time','timestamp_ms','relative_timestamp_ms',
                'path','width','height','source_video_sha256','derivative_sha256','decoded_pixel_sha256',
                'geometry_reference','geometry_reference_sha256','geometry_state','review_annotation','quality_score','quality_components',
                'view_descriptor','predicted_orientation','visibility','elevation_degrees']})
        sessions.append({k:s[k] for k in ['capture_session_id','position_id','position_group_id','independence_group_id']}|
                        dict(source_video=s['source_video'],selected=chosen,selection_trace=result['selection_trace'],
                             candidate_decisions=result['candidate_decisions']))
    return dict(schema_version='offline_selected_keyframes_d1_v1',dataset=dataset['name'],source_catalog=manifest['source_catalog'],
                source_catalog_sha256=manifest['source_catalog_sha256'],designation='development_only',split='pilot',
                training_eligible=False,final_holdout=False,policy=policy,strategy=strategy,budget=budget,
                settings=settings,sessions=sessions,ground_truth_FEN_used_for_selection=False,
                D2_input_role='primary_candidates' if dataset['name']=='primary_3s_high' else 'robustness_comparison_only',
                quality_score_kind='uncalibrated_image_heuristic',
                geometry_config_sha256=manifest['geometry_config_sha256'],
                geometry_evaluator_sha256=manifest['critical_sha256']['scripts/evaluate_pilot_geometry.py'],
                piece_recognition_evaluated=False,per_square_evidence_or_fusion_implemented=False,
                geometry_warning='Completed unverified geometry is provisional usability, not certified correctness.',
                representation_warning='High and compressed copies share recording/session/position groups; never independent samples.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--reviews',type=Path);args=p.parse_args()
    if args.output_dir.exists():p.error('Fresh output directory required')
    manifest=read(args.run/'candidates.json');settings=validate_settings(read(args.run/'settings.json'));reviews=review_map(args.reviews)
    check_hashes(manifest['critical_sha256']);check_hashes({manifest['source_catalog']:manifest['source_catalog_sha256']})
    cv2.setNumThreads(1);start=time.perf_counter()
    datasets=[primary_dataset(args.run,manifest,settings,reviews)]+paired_datasets(manifest,settings,reviews)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    write(args.output_dir/'features.json',dict(schema_version='offline_keyframe_features_d1_v1',datasets=datasets,
           feature_source_sha256=sha256_file('src/scanning/video_keyframes.py'),config=settings,
           reviews_path=str(args.reviews) if args.reviews else None,reviews_sha256=sha256_file(args.reviews) if args.reviews else None))
    metrics=[];replays=0
    for dataset in datasets:
        for policy in ['legacy','marked-board']:
            for strategy in STRATEGIES:
                for budget in settings['budgets']:
                    selections=[]
                    for s in dataset['sessions']:
                        rows=s['policies'][policy];begin=time.perf_counter();sel=select_keyframes(rows,strategy,budget,settings)
                        elapsed=time.perf_counter()-begin
                        if sel!=select_keyframes(list(reversed(rows)),strategy,budget,settings):raise ValueError('Nondeterministic selection')
                        replays+=1;selections.append(sel)
                        m=selection_metrics(rows,sel,settings)
                        metrics.append(dict(dataset=dataset['name'],session_id=s['capture_session_id'],policy=policy,strategy=strategy,budget=budget,
                                            candidates=len(rows),geometry_completed=sum(r['geometry_completed'] for r in rows),
                                            geometry_eligible=sum(r['eligible'] for r in rows),states=dict(Counter(r['geometry_state'] for r in rows)),
                                            quality_rejected=sum(bool(r['quality_rejection_reasons']) for r in rows),selection_seconds=elapsed,**m))
                    write(args.output_dir/'selected'/dataset['name']/policy/strategy/f'budget_{budget}'/'selected.json',
                          selected_manifest(dataset,policy,strategy,budget,selections,manifest,settings))
    timing=dict(metrics=metrics,deterministic_replays_passed=replays,
          feature_selection_seconds=time.perf_counter()-start,sampling_geometry_candidates_sha256=sha256_file(args.run/'candidates.json'),
          selector_sha256=sha256_file('src/scanning/video_keyframes.py'),evaluation_script_sha256=sha256_file(Path(__file__)),
          selection_uses_FEN=False,ground_truth_geometry_available=False)
    fields=list(metrics[0]);
    with (args.output_dir/'metrics.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in metrics)
    from scripts.render_keyframe_diagnostics import render
    render_start=time.perf_counter()
    render(args.output_dir,datasets,metrics,settings)
    timing['diagnostics_seconds']=time.perf_counter()-render_start
    preservation_start=time.perf_counter()
    check_hashes(manifest['protected_sha256']);check_hashes(manifest['critical_sha256'])
    timing['preservation_seconds']=time.perf_counter()-preservation_start
    timing['analysis_total_seconds']=time.perf_counter()-start
    write(args.output_dir/'metrics.json',timing)
    write(args.output_dir/'preservation.json',dict(protected_files_unchanged=True,protected_file_count=len(manifest['protected_sha256']),
          original_video_hashes_unchanged=True,production_and_browser_unchanged=True,selected_manifests_replayed=replays))
    print('Strategies evaluated; deterministic replays',replays,flush=True)

if __name__=='__main__':main()
