"""Register/evaluate reviewed new-session D.2 evidence with frozen D.3 methods."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read, write, check_hashes
from scripts.run_fusion_experiments import load_evidence
from scripts.validate_native_pilot import validate_session
from src.scanning.independent_fusion import validate_registry
from src.scanning.native_pilot import sha256_file
from src.scanning.offline_fusion import fuse_board
from src.scanning.square_evidence import classification_metrics
from src.scanning.fusion_metrics import reliability
from src.chess.fen import board_from_fen
from src.chess.labels import CLASSES

PROTOCOL=Path('configs/experiments/fusion_independent_d3b_v1.json')
PREREGISTRATION=Path('data/metadata/fusion_independent_d3b_preregistration_v1.json')
METHODS=('single_quality','mean_probabilities','mean_logits')
CROP_KEYS=('checkpoint_sha256','crop_spec','board_size','context_margin','initial_context_padding','calibration')


def known_history():
    old=read('data/metadata/a3_pilot_01_sources_v2.json'); d3=read('docs/evaluation/fusion_d3_summary_v1.json')
    placements={f.split()[0] for f in d3['label_version']['evaluated_fens'].values()}
    placements.update(s['fen'].split()[0] for s in old['sessions'])
    # Historical position labels are used only as a provenance exclusion list.
    historic=read('data/metadata/positions.json')
    def walk(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in ('fen','placement_fen') and isinstance(v,str) and '/' in v: placements.add(v.split()[0])
                else: walk(v)
        elif isinstance(x,list):
            for v in x: walk(v)
    walk(historic)
    return dict(sessions={s['capture_session_id'] for s in old['sessions']},
                sources={s[k]['sha256'] for s in old['sessions'] for k in ('primary_source','compressed_comparison_source')},placements=placements)


def frozen_sources():
    paths=[PROTOCOL,Path('configs/experiments/fusion_d3.json'),Path('configs/experiments/keyframes_d1.json'),Path('configs/experiments/evidence_d2.json'),
           Path(__file__),Path('src/scanning/independent_fusion.py'),Path('src/scanning/offline_fusion.py'),Path('src/scanning/fusion_metrics.py'),
           Path('src/scanning/square_evidence.py'),Path('src/scanning/video_keyframes.py'),Path('scripts/run_fusion_experiments.py'),
           Path('scripts/run_square_evidence.py'),Path('scripts/select_video_keyframes.py'),Path('scripts/evaluate_keyframe_selection.py'),
           Path('src/vision/inference.py'),Path('src/vision/crop_experiments.py'),Path('data/metadata/positions.json'),Path('data/metadata/a3_pilot_01_sources_v2.json'),Path('docs/evaluation/fusion_d3_summary_v1.json'),Path('docs/evaluation/fusion_independent_d3b_protocol_v1.md')]
    paths += [p for p in Path('src/vision').glob('*.py')]
    return {str(p):sha256_file(p) for p in paths if p.exists()}


def register(registry_path, output, functional=False):
    if output.exists(): raise ValueError('Fresh registration path required')
    preregistration=read(PREREGISTRATION);check_hashes(preregistration['source_sha256'])
    registry=read(registry_path); protocol=read(PROTOCOL); summary=validate_registry(registry,known_history(),functional_development=functional)
    check_hashes({protocol['fusion_config']:protocol['fusion_config_sha256']})
    media={}
    for s in registry['sessions']:
        for key in ('source_video','reference','native_index'): media[s[key]['path']]=s[key]['sha256']
        check_hashes(media)
        native=read(s['native_index']['path']);validate_session(native,check_files=True)
        if native['session_id']!=s['session_id'] or native['source_video']['sha256']!=s['source_video']['sha256']:
            raise ValueError('Native timeline/source/session mismatch')
    lock=dict(schema_version='fusion_independent_registration_d3b_v1',registered_at=datetime.now(timezone.utc).isoformat(),
        preregistration_path=str(PREREGISTRATION),preregistration_sha256=sha256_file(PREREGISTRATION),
        protocol_path=str(PROTOCOL),protocol_sha256=sha256_file(PROTOCOL),protocol=protocol,registry_path=str(registry_path),registry_sha256=sha256_file(registry_path),
        registry=registry,group_summary=summary,source_sha256=frozen_sources(),media_sha256=media,
        functional_development=functional,independent_validation='pending' if functional else 'registered_pending_evaluation',
        registration_is_not_a_claim_of_safe_thresholds=True)
    write(output,lock); print('Registered',summary,'functional development:',functional)


def evaluate(registration, manifest_path, output, functional=False):
    if output.exists(): raise ValueError('Fresh evaluation directory required')
    start=time.perf_counter();lock=read(registration)
    if lock.get('schema_version')!='fusion_independent_registration_d3b_v1': raise ValueError('Unknown registration schema')
    if functional!=lock['functional_development']: raise ValueError('Development mode mismatch')
    check_hashes(lock['source_sha256']);check_hashes(lock['media_sha256']);check_hashes({lock['registry_path']:lock['registry_sha256']})
    if lock['registry']!=read(lock['registry_path']) or lock['protocol']!=read(lock['protocol_path']): raise ValueError('Embedded preregistration metadata changed')
    check_hashes({lock['preregistration_path']:lock['preregistration_sha256']})
    validate_registry(lock['registry'],known_history(),functional_development=functional)
    protocol=lock['protocol'];check_hashes({lock['protocol_path']:lock['protocol_sha256']})
    manifest=read(manifest_path)
    if manifest.get('final_holdout') is not False: raise ValueError('Final holdout prohibited')
    extraction=read(manifest['config_path']);reference=read('configs/experiments/evidence_d2.json')
    if any(extraction[k]!=reference[k] for k in CROP_KEYS): raise ValueError('Frozen model/crop/preprocessing changed')
    if not functional and (extraction.get('independent_registration_sha256')!=sha256_file(registration) or extraction.get('independent_protocol_sha256')!=lock['protocol_sha256']):
        raise ValueError('Evidence was not bound to preregistration before extraction')
    if not functional:
        for condition in extraction['conditions']:
            if condition['name']=='uniform_5':
                selected=read(condition['manifest'])
                check_hashes({selected['source_catalog']:selected['source_catalog_sha256']})
                catalog=read(selected['source_catalog'])
                check_hashes({catalog['owner_attestations']:catalog['owner_attestations_sha256']})
                if selected['settings']!=read(protocol['D1_config']): raise ValueError('D.1 selection settings changed')
                if selected['strategy']!='uniform' or selected['budget']!=5 or selected['policy']!='marked-board': raise ValueError('Selection protocol changed')
    e,packet,obs,exclusions=load_evidence(manifest_path)
    registered={s['session_id']:s for s in lock['registry']['sessions']}
    uniform=next(c for c in packet['conditions'] if c['name']=='uniform_5')
    check_hashes({uniform['manifest']:uniform['manifest_sha256']})
    selection=read(uniform['manifest'])
    if selection['settings']!=read(protocol['D1_config']) or selection['strategy']!='uniform' or selection['budget']!=5: raise ValueError('Frozen uniform selection changed')
    if set(registered)!={s['capture_session_id'] for s in e['sessions']}: raise ValueError('Registered session denominator changed')
    for s in e['sessions']:
        r=registered[s['capture_session_id']]
        if any(s[k]!=r[k] for k in ('position_id','position_group_id','independence_group_id')) or s['source_video']['sha256']!=r['source_video']['sha256']:
            raise ValueError('Registered session/source grouping mismatch')
    config=read(protocol['fusion_config']);predictions=[];unusable=[];times=[]
    # Ground truth is not used until all predictions have been written.
    for sid,r in registered.items():
        rows=[o for o in obs if o['session_id']==sid and any(m['condition']=='uniform_5' for m in o['condition_memberships'])]
        if any(o['geometry_policy']!='marked-board' for o in rows): raise ValueError('Geometry policy changed')
        if any(m['keyframe_strategy']!='uniform' or m['selection_budget']!=5 for o in rows for m in o['condition_memberships'] if m['condition']=='uniform_5'):
            raise ValueError('Uniform-5 membership changed')
        selected_session=next(s for s in selection['sessions'] if s['capture_session_id']==sid)
        nominated={f['frame_id'] for f in selected_session['selected']}
        if not {o['frame_id'] for o in rows} <= nominated: raise ValueError('Evidence silently substituted selected frames')
        for o in rows:
            membership=next(m for m in o['condition_memberships'] if m['condition']=='uniform_5')
            if membership['selected_manifest_sha256']!=uniform['manifest_sha256']: raise ValueError('Selection provenance mismatch')
        if len({o['frame_id'] for o in rows})!=5 or len(nominated)!=5:
            unusable.append(dict(session_id=sid,reason='requires_five_verified_original_selected_frames',usable_frames=len({o['frame_id'] for o in rows})));continue
        for method in METHODS:
            t=time.perf_counter();p=fuse_board(rows,method,config);times.append(time.perf_counter()-t)
            if p!=fuse_board(rows[::-1],method,config): raise ValueError('Deterministic replay failed')
            predictions.append(p)
    output.mkdir(parents=True)
    write(output/'predictions.json',dict(boards=predictions,ground_truth_used_for_predictions=False,registration_sha256=sha256_file(registration)))
    symbol_ids={c.symbol:i for i,c in enumerate(CLASSES)};boards=[]
    for p in predictions:
        truth=[symbol_ids[s] for row in board_from_fen(registered[p['session_id']]['fen']).rows for s in row]
        pred=[s['predicted_class_id'] for s in p['squares']];scores=[s['score_distribution'] for s in p['squares']];ab=[s['abstained'] for s in p['squares']]
        boards.append(dict(session_id=p['session_id'],position_id=p['position_id'],position_group_id=registered[p['session_id']]['position_group_id'],independence_group_id=registered[p['session_id']]['independence_group_id'],method=p['method'],**classification_metrics(truth,pred),
            exact_board=truth==pred,reliability=reliability(truth,scores,pred,ab,config),
            confidently_wrong_occupied=sum(t!=0 and t!=v and max(s)>=.9 for t,v,s in zip(truth,pred,scores)),
            all_squares_retained=not any(ab),maximum_entropy=max(s['entropy_nats'] for s in p['squares']),
            maximum_disagreement=max(s['jensen_shannon_disagreement_nats'] for s in p['squares']),minimum_margin=min(s['top_two_margin'] for s in p['squares']),mean_confidence=sum(map(max,scores))/64,minimum_confidence=min(map(max,scores))))
    totals=[dict(method=m,exact_boards=sum(b['exact_board'] for b in boards if b['method']==m),exact_board_accuracy=sum(b['exact_board'] for b in boards if b['method']==m)/len(registered),registered_denominator=len(registered),
        usable_denominator=sum(b['method']==m for b in boards),correct_squares=sum(b['correct'] for b in boards if b['method']==m),
        occupied_correct=sum(b['occupied_correct'] for b in boards if b['method']==m),occupied_count=sum(b['occupied_count'] for b in boards if b['method']==m),
        retained_boards=sum(b['all_squares_retained'] for b in boards if b['method']==m),
        retained_incorrect_boards=sum(b['all_squares_retained'] and not b['exact_board'] for b in boards if b['method']==m)) for m in METHODS]
    for total in totals:
        total['overall_accuracy_usable']=total['correct_squares']/(64*total['usable_denominator']) if total['usable_denominator'] else None
        total['occupied_accuracy_usable']=total['occupied_correct']/total['occupied_count'] if total['occupied_count'] else None
    position_groups=[]
    for group in sorted({r['position_group_id'] for r in registered.values()}):
        for method in METHODS:
            group_sessions=[sid for sid,r in registered.items() if r['position_group_id']==group]
            selected_boards=[b for b in boards if b['position_group_id']==group and b['method']==method]
            position_groups.append(dict(position_group_id=group,method=method,registered_sessions=group_sessions,usable_sessions=len(selected_boards),
                exact_boards=sum(b['exact_board'] for b in selected_boards),exact_fraction_registered=sum(b['exact_board'] for b in selected_boards)/len(group_sessions)))
    write(output/'evaluation.json',dict(schema_version='fusion_independent_evaluation_d3b_v1',
        independent_validation='pending' if functional else 'evaluated_registered_cohort_not_safety_certification',functional_development=functional,
        registration_sha256=sha256_file(registration),protocol_sha256=lock['protocol_sha256'],evidence_sha256=sha256_file(manifest_path),
        position_group_results=position_groups,registered_sessions=len(registered),minimum_cohort_collected=(lock['group_summary']['sessions']>=protocol['minimum_planned_sessions'] and lock['group_summary']['distinct_position_groups']>=protocol['minimum_distinct_positions'] and lock['group_summary']['capture_dates']>=protocol['minimum_capture_dates']),group_summary=lock['group_summary'],boards=boards,totals=totals,unusable_sessions=unusable,exclusions=exclusions,
        label_provenance={s['session_id']:dict(fen=s['fen'],label_version=s['label_version'],reference=s['reference'],attestation=s['attestation']) for s in registered.values()},
        fusion_call_seconds=times,evidence_inference_seconds=e.get('inference_loop_seconds'),elapsed_seconds=time.perf_counter()-start,
        completion_enabled=False,score_calibrated=False,final_holdout=False))
    print('Evaluated',len(registered),'registered sessions; independent validation:', 'pending' if functional else 'cohort evaluated')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('register');a.add_argument('--registry',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('evaluate');a.add_argument('--registration',type=Path,required=True);a.add_argument('--evidence-manifest',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True)
    for a in sub.choices.values(): a.add_argument('--functional-development',action='store_true')
    a=p.parse_args()
    if a.command=='register':register(a.registry,a.output,a.functional_development)
    else:evaluate(a.registration,a.evidence_manifest,a.output_dir,a.functional_development)

if __name__=='__main__':main()
