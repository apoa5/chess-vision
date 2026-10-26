"""D.3 evaluation after GT-blind prediction files exist; no prediction selection."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.select_video_keyframes import read,write,check_hashes
from scripts.characterize_square_evidence import evaluation_truths
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import SQUARES,classification_metrics,multi_view_metrics
from src.scanning.fusion_metrics import reliability,legality


def run(run_dir,output):
    if output.exists():raise ValueError('Fresh evaluation directory required')
    started=time.perf_counter();inputs=read(run_dir/'inputs.json');prediction=read(run_dir/'predictions.json');config=read(run_dir/'settings.json')
    check_hashes(inputs['source_sha256']);check_hashes(inputs['protected_sha256']);check_hashes({inputs['evidence_path']:inputs['evidence_sha256'],inputs['observations_path']:inputs['observations_sha256']})
    if prediction['ground_truth_loaded_for_predictions'] or prediction['config_sha256']!=inputs['config_sha256']:raise ValueError('Prediction provenance mismatch')
    index=read(config['evidence_index']);check_hashes(index['artifacts'])
    if config['ground_truth_corrections']!=index['evaluation_ground_truth_correction']:raise ValueError('Unregistered label-version choice')
    evidence=read(inputs['evidence_path']);packet=read(evidence['packet_path']);catalog_path=read(packet['config_path'])['source_catalog']
    check_hashes({catalog_path:packet['protected_sha256'][catalog_path]})
    correction=read(config['ground_truth_corrections']);truth,fens=evaluation_truths(read(catalog_path),sha256_file(catalog_path),correction)
    obs=[json.loads(x) for x in Path(inputs['observations_path']).read_text().splitlines()];by_id={o['observation_id']:o for o in obs}
    # Additional explicit camera-index -> canonical-square and orientation check.
    mappings=0
    for o in obs:
        g=read(o['geometry_reference'])['policies'][o['geometry_policy']]
        if g['semantic_square_ids'][o['input_index']]!=o['canonical_square'] or g['rotation_quarters_ccw']!=o['rotation_quarters_ccw']:raise ValueError('Canonical crop/index/orientation mismatch')
        mappings+=1
    output.mkdir(parents=True,exist_ok=False);boards=[];minority=[];oracle=[]
    for b in prediction['boards']:
        if [s['canonical_square'] for s in b['squares']]!=list(SQUARES):raise ValueError('Noncanonical output board order')
        y=truth[b['session_id']];p=[s['predicted_class_id'] for s in b['squares']];scores=[s['score_distribution'] for s in b['squares']];abstain=[s['abstained'] for s in b['squares']]
        metrics=classification_metrics(y,p);rel=reliability(y,scores,p,abstain,config)
        for i,s in enumerate(b['squares']):
            s['true_class_id']=y[i];s['correct']=p[i]==y[i]
            correct_views=sum(by_id[oid]['predicted_class_id']==y[i] for oid in s['observation_ids'])
            s['correct_input_argmax_count_diagnostic_only']=correct_views
            if b['method']!='single_quality' and p[i]!=y[i] and 0<correct_views<len(s['observation_ids'])/2:
                minority.append(dict(condition=b['condition'],session_id=b['session_id'],method=b['method'],canonical_square=s['canonical_square'],
                    true_class_id=y[i],fused_class_id=p[i],correct_views=correct_views,input_views=len(s['observation_ids']),
                    fused_score=s['maximum_score'],abstained=s['abstained'],observation_ids=s['observation_ids']))
        boards.append(dict(**b,metrics=metrics,reliability=rel,exact_board=p==y,legality=legality(p)))
    aggregates=[]
    for condition in config['conditions']:
        for method in config['methods']:
            selected=[b for b in boards if b['condition']==condition and b['method']==method]
            y=[t for b in selected for t in truth[b['session_id']]];p=[s['predicted_class_id'] for b in selected for s in b['squares']]
            score=[s['score_distribution'] for b in selected for s in b['squares']];ab=[s['abstained'] for b in selected for s in b['squares']]
            aggregates.append(dict(condition=condition,method=method,**classification_metrics(y,p),reliability=reliability(y,score,p,ab,config),
                exact_boards=sum(b['exact_board'] for b in selected),board_denominator=len(selected),
                exact_session_ids=[b['session_id'] for b in selected if b['exact_board']],
                legal_under_either_assumed_turn=sum(b['legality']['valid_under_either_assumed_turn'] for b in selected),
                strong_conflict_squares=sum(s['strong_conflict'] for b in selected for s in b['squares'])))
    comparisons=[]
    for b in boards:
        references=[('same_condition_single',b['condition'],'single_quality'),('uniform5_mean_probabilities','uniform_5','mean_probabilities'),('uniform5_majority','uniform_5','majority_vote')]
        for name,condition,method in references:
            ref=next(x for x in boards if x['session_id']==b['session_id'] and x['condition']==condition and x['method']==method)
            rescued=[s['canonical_square'] for s,t in zip(b['squares'],ref['squares']) if s['correct'] and not t['correct']]
            broken=[s['canonical_square'] for s,t in zip(b['squares'],ref['squares']) if not s['correct'] and t['correct']]
            comparisons.append(dict(session_id=b['session_id'],condition=b['condition'],method=b['method'],reference=name,
                correct_change=b['metrics']['correct']-ref['metrics']['correct'],occupied_correct_change=b['metrics']['occupied_correct']-ref['metrics']['occupied_correct'],
                exact_board_change=int(b['exact_board'])-int(ref['exact_board']),rescued=rescued,broken=broken))
    # Oracle diagnostics are computed only after predictions, and cannot affect them.
    for condition in config['conditions']:
        for session in evidence['sessions']:
            sid=session['capture_session_id'];rows=[o for o in obs if o['session_id']==sid and any(m['condition']==condition for m in o['condition_memberships'])]
            fids=sorted(set(o['frame_id'] for o in rows));vectors=[];individual=[]
            for fid in fids:
                lookup={o['canonical_square']:o for o in rows if o['frame_id']==fid};v=[lookup[s]['predicted_class_id'] for s in SQUARES];vectors.append(v)
                individual.append(dict(frame_id=fid,timestamp_ms=lookup[SQUARES[0]]['timestamp_ms'],**classification_metrics(truth[sid],v)))
            best=min(individual,key=lambda m:(-m['correct'],-m['occupied_correct'],m['timestamp_ms'],m['frame_id']))
            ceiling=multi_view_metrics(truth[sid],np.asarray(vectors));oracle.append(dict(condition=condition,session_id=sid,
                oracle_kind='GT diagnostic, never a method or prediction input; per-view argmax-selection ceiling only',
                ever_correct_squares=ceiling['ever_correct_squares'],oracle_exact_board=ceiling['oracle_exact_board'],best_individual_with_GT=best,
                average_individual_metrics=classification_metrics(np.tile(truth[sid],(len(vectors),1)),vectors)))
    result=dict(schema_version='offline_fusion_results_d3_v1',designation='development_only',final_holdout=False,
        predictions_path=str(run_dir/'predictions.json'),predictions_sha256=sha256_file(run_dir/'predictions.json'),evidence_sha256=inputs['evidence_sha256'],
        config=config,config_sha256=inputs['config_sha256'],source_sha256={p:sha256_file(p) for p in ['scripts/evaluate_fusion_experiments.py','src/scanning/fusion_metrics.py']},
        label_version=dict(source_catalog=catalog_path,source_catalog_sha256=sha256_file(catalog_path),correction_path=config['ground_truth_corrections'],
                           correction_sha256=sha256_file(config['ground_truth_corrections']),evaluated_fens=fens),
        validated_camera_index_canonical_mappings=mappings,exclusions=prediction['excluded'],aggregates=aggregates,boards=boards,
        paired_session_comparisons=comparisons,correct_minority_overwhelmed=minority,oracle_diagnostics_only=oracle,
        independence_warning='Five correlated development sessions; no frame/square-IID significance or calibrated reliability claim.',
        evaluation_seconds=time.perf_counter()-started)
    write(output/'results.json',result)
    flat=[]
    for a in aggregates:
        rel=a['reliability'];flat.append({k:a[k] for k in ('condition','method','overall_accuracy','occupied_accuracy','empty_false_positives','occupied_to_empty_misses','incorrect_piece_identities','exact_boards','board_denominator')}|
            {k:rel[k] for k in ('ece','nll_with_floor','multiclass_brier','high_score_wrong_count','coverage','accepted_error_count','accepted_error_rate')})
    with (output/'comparison.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    check_hashes(inputs['protected_sha256']);print('Evaluated',len(boards),'boards; labels loaded only after prediction artifact existed')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.run,a.output_dir)
