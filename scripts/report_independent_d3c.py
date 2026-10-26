"""Evaluate frozen provisional D.3c predictions; FENs loaded here for evaluation only."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read,write,check_hashes
from src.scanning.interim_video import write_gzip
from src.scanning.interim_evidence import fuse_provisional_board
from src.scanning.square_evidence import SQUARES,classification_metrics
from src.scanning.fusion_metrics import reliability
from src.scanning.native_pilot import sha256_file
from src.vision.dataset import labels_from_fen
from src.chess.labels import CLASS_NAMES


def gzread(p):return json.loads(gzip.decompress(Path(p).read_bytes()))


def run(run_dir,output):
    if output.exists():raise ValueError('Fresh report output required')
    started=time.perf_counter();registration=read(run_dir/'registration.json');rec=read(run_dir/'recognition/sessions.json')
    check_hashes(registration['protected_sha256']);check_hashes(rec['source_sha256'])
    prediction=gzread(run_dir/'recognition/predictions.json.gz');obs=gzread(run_dir/'recognition/observations.json.gz')['observations']
    config=read('configs/experiments/fusion_d3.json');lookup={s['session_id']:s for s in registration['sessions']}
    truth={s['session_id']:[v.class_id for row in labels_from_fen(s['fen']) for v in row] for s in registration['sessions']}
    boards=[];paired=[];prefixes=[]
    for p in prediction['boards']:
        if [s['canonical_square'] for s in p['squares']]!=list(SQUARES):raise ValueError('Noncanonical prediction board')
        y=truth[p['session_id']];pred=[s['predicted_class_id'] for s in p['squares']];scores=[s['score_distribution'] for s in p['squares']];ab=[s['abstained'] for s in p['squares']]
        for t,s in zip(y,p['squares']):s.update(true_class_id=t,correct=s['predicted_class_id']==t)
        boards.append(dict(**p,metrics=classification_metrics(y,pred),reliability=reliability(y,scores,pred,ab,config),exact_board=y==pred,
            all_squares_retained=not any(ab),mean_probability=sum(map(max,scores))/64,
            confident_error_squares=[s['canonical_square'] for s in p['squares'] if not s['correct'] and s['maximum_score']>=.9],
            model_disagreement_squares=sum(len(set(s['input_class_ids']))>1 for s in p['squares']),
            truth_status='supplied_FEN_not_photo_verified',geometry_status='completed_unverified',
            novel_placement=not lookup[p['session_id']]['known_exact_placement_overlap']))
    for b in boards:
        ref=next(x for x in boards if x['session_id']==b['session_id'] and x['method']=='single_quality')
        paired.append(dict(session_id=b['session_id'],method=b['method'],reference='same_uniform5_quality_single',
            rescued=[s['canonical_square'] for s,t in zip(b['squares'],ref['squares']) if s['correct'] and not t['correct']],
            regressed=[s['canonical_square'] for s,t in zip(b['squares'],ref['squares']) if not s['correct'] and t['correct']]))
    for sid in lookup:
        rows=[o for o in obs if o['session_id']==sid];ids=sorted(set(o['frame_id'] for o in rows),key=lambda f:next(o['timestamp_ms'] for o in rows if o['frame_id']==f))
        if len(ids)!=5:continue
        y=truth[sid]
        for method in ('mean_probabilities','mean_logits'):
            previous=None
            for k in range(1,6):
                b=fuse_provisional_board([o for o in rows if o['frame_id'] in ids[:k]],method,config,explicit_review_waiver=True)
                pred=[s['predicted_class_id'] for s in b['squares']];ab=[s['abstained'] for s in b['squares']]
                change=dict(rescued=[],regressed=[],withheld_to_retained_correct=[],withheld_to_retained_incorrect=[])
                if previous:
                    for square,t,a,c in zip(SQUARES,y,previous['squares'],b['squares']):
                        if a['predicted_class_id']!=t and c['predicted_class_id']==t:change['rescued'].append(square)
                        if a['predicted_class_id']==t and c['predicted_class_id']!=t:change['regressed'].append(square)
                        if a['abstained'] and not c['abstained']:change['withheld_to_retained_correct' if c['predicted_class_id']==t else 'withheld_to_retained_incorrect'].append(square)
                prefixes.append(dict(session_id=sid,method=method,k=k,frame_ids=ids[:k],metrics=classification_metrics(y,pred),
                    withheld_squares=sum(ab),retained_errors=sum(not s['abstained'] and s['predicted_class_id']!=t for s,t in zip(b['squares'],y)),
                    exact_board=y==pred,all_squares_retained=not any(ab),**change));previous=b
    totals=[]
    for scope in ('all_10_sessions','novel_placement_subset'):
        for method in registration['config']['methods']:
            chosen=[b for b in boards if b['method']==method and (scope=='all_10_sessions' or b['novel_placement'])]
            y=[t for b in chosen for t in truth[b['session_id']]];p=[s['predicted_class_id'] for b in chosen for s in b['squares']]
            score=[s['score_distribution'] for b in chosen for s in b['squares']];ab=[s['abstained'] for b in chosen for s in b['squares']]
            denominator=10 if scope=='all_10_sessions' else sum(not s['known_exact_placement_overlap'] for s in lookup.values())
            if not chosen:totals.append(dict(scope=scope,method=method,eligible_sessions=0,registered_denominator=denominator));continue
            totals.append(dict(scope=scope,method=method,**classification_metrics(y,p),reliability=reliability(y,score,p,ab,config),
                exact_boards=sum(b['exact_board'] for b in chosen),registered_denominator=denominator,eligible_sessions=len(chosen),
                retained_boards=sum(b['all_squares_retained'] for b in chosen),incorrect_retained_boards=sum(b['all_squares_retained'] and not b['exact_board'] for b in chosen),
                incorrect_boards_mean_score_ge_98=sum(not b['exact_board'] and b['mean_probability']>=.98 for b in chosen)))
    geom=read(run_dir/'geometry_summary.json')
    result=dict(schema_version='limited_independent_fusion_results_d3c_v1',scope=registration['scope'],strict_preregistered_gate_passed=False,
        strict_verified_geometry_sessions=0,provisional_only=True,registered_sessions=10,novel_placements=sum(not s['known_exact_placement_overlap'] for s in lookup.values()),
        observed_independent_environments=1,truth_verification='owner supplied; missing reference photos',label_mapping_sha256=sha256_file(registration['config']['mapping']),
        report_source_sha256=sha256_file(Path(__file__)),class_names=list(CLASS_NAMES),model_sha256=rec['model_sha256'],input_sha256={str(run_dir/'registration.json'):sha256_file(run_dir/'registration.json'),str(run_dir/'recognition/predictions.json.gz'):sha256_file(run_dir/'recognition/predictions.json.gz'),
            str(run_dir/'recognition/observations.json.gz'):sha256_file(run_dir/'recognition/observations.json.gz')},
        method_config=config,boards=boards,aggregates=totals,paired_changes=paired,nested_prefix_sensitivity=prefixes,
        geometry_counts=dict(candidates=sum(s['candidate_count'] for s in geom['sessions']),completed=sum(s['geometry_completed'] for s in geom['sessions']),legacy_completed=sum(s['legacy_completed'] for s in geom['sessions']),
            selector_eligible=sum(s['selector_eligible'] for s in geom['sessions']),failures=dict(sum((Counter(s['failures']) for s in geom['sessions']),Counter()))),
        session_geometry=[{k:s[k] for k in ('session_id','candidate_count','geometry_completed','legacy_completed','selector_eligible','failures','metadata')} for s in geom['sessions']],
        inference=rec,elapsed_seconds=time.perf_counter()-started,warning='Provisional metrics conditional on unverified geometry/FEN; same room/light cohort, no IID square/frame inference')
    output.mkdir();write_gzip(output/'results.json.gz',result)
    compact={k:v for k,v in result.items() if k not in ('boards','nested_prefix_sensitivity')};compact['boards']=[{k:v for k,v in b.items() if k!='squares'} for b in boards]
    write(output/'summary.json',compact)
    fields=['session_id','method','correct','square_count','occupied_correct','occupied_count','empty_false_positives','occupied_to_empty_misses','incorrect_piece_identities','exact_board','mean_probability','retained_squares','retained_errors','confident_errors']
    with (output/'per_session.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for b in boards:w.writerow(dict(session_id=b['session_id'],method=b['method'],**{k:b['metrics'][k] for k in fields[2:9]},exact_board=b['exact_board'],mean_probability=b['mean_probability'],retained_squares=b['reliability']['accepted_count'],retained_errors=b['reliability']['accepted_error_count'],confident_errors=len(b['confident_error_squares'])))
    with (output/'per_class.csv').open('w',newline='') as h:
        fields=['scope','method','class_id','class_name','support','predicted_count','precision','recall'];w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for a in totals:
            for c in a.get('per_class',[]):w.writerow(dict(scope=a['scope'],method=a['method'],class_name=CLASS_NAMES[c['class_id']],**c))
    print('Evaluated all registered sessions; provisional eligibility',len(boards)//3,'/10. Strict verified gate not passed.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.run,a.output_dir)
