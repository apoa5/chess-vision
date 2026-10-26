"""D.2 ground-truth diagnostics only: no combined class distribution or fusion."""
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write
from src.chess.labels import CLASS_NAMES
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import (SQUARES,classification_metrics,multi_view_metrics,paired_errors,descriptive_correlation,validate_observation,validate_bundle)
from src.scanning.video_keyframes import view_distance
from src.vision.dataset import labels_from_fen


def paired_coverage(a,b):
    """Physical-square deltas, not independent-crop statistical tests."""
    if a.get('squares') is None or b.get('squares') is None:return dict(status='missing_verified_views')
    aa={s['canonical_square']:s['ever_correct'] for s in a['squares']};bb={s['canonical_square']:s['ever_correct'] for s in b['squares']}
    return dict(status='descriptive_paired_session',b_minus_a_ever_correct=b['ever_correct_squares']-a['ever_correct_squares'],
        rescued_by_b=[s for s in SQUARES if not aa[s] and bb[s]],lost_by_b=[s for s in SQUARES if aa[s] and not bb[s]],
        a_usable_views=a['usable_views'],b_usable_views=b['usable_views'],
        equal_usable_frame_budget=a['usable_views']==b['usable_views'],
        oracle_ceiling_change=int(b['oracle_exact_board'])-int(a['oracle_exact_board']))


def draw_crop_distributions(observations,title,path):
    """Actual packed B.2 RGB contexts, binary-mask outline and all 13 raw probabilities."""
    cells=[]
    for o in observations:
        packed=np.load(o['input_cache'])['packed'][o['input_index']]
        import hashlib
        if hashlib.sha256(packed.tobytes()).hexdigest()!=o['input_packed_sha256']:raise ValueError('Input cache changed')
        canvas=np.full((570,320,3),248,np.uint8);rgb=packed[:3].transpose(1,2,0);image=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        contours,_=cv2.findContours(packed[3].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image,contours,-1,(60,255,60),1);canvas[45:269,48:272]=image
        for i,txt in enumerate([f"{o['timestamp_ms']/1000:.3f}s {o['semantic_orientation']}",f"pred {o['predicted_class']} max {o['maximum_probability']:.3f}"]):
            cv2.putText(canvas,txt,(5,15+18*i),cv2.FONT_HERSHEY_SIMPLEX,.35,(20,20,20),1,cv2.LINE_AA)
        for k,p in enumerate(o['probabilities']):
            y=290+k*20;cv2.putText(canvas,CLASS_NAMES[k],(5,y),cv2.FONT_HERSHEY_SIMPLEX,.32,(20,20,20),1)
            cv2.rectangle(canvas,(120,y-10),(120+int(p*150),y),(180,100,40),-1)
            cv2.putText(canvas,f'{p:.3f}',(275,y),cv2.FONT_HERSHEY_SIMPLEX,.3,(20,20,20),1)
        cells.append(canvas)
    if not cells:return
    body=cv2.hconcat(cells);banner=np.full((35,body.shape[1],3),255,np.uint8)
    cv2.putText(banner,title,(5,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
    cv2.imwrite(str(path),cv2.vconcat([banner,body]))



def evaluation_truths(catalog,catalog_sha256,corrections=None):
    """Only explicit versioned human corrections can supplement immutable FENs."""
    registered={s['capture_session_id']:s for s in catalog['sessions']}
    fens={sid:s['fen'] for sid,s in registered.items()}
    if corrections is not None:
        if corrections['schema_version']!='offline_pilot_ground_truth_corrections_v1' or corrections['source_catalog_sha256']!=catalog_sha256:
            raise ValueError('Ground-truth correction source mismatch')
        seen=set()
        for c in corrections['corrections']:
            sid=c['session_id']
            if sid in seen:raise ValueError('Duplicate correction')
            seen.add(sid);original=registered[sid]
            if c['position_id']!=original['position_id'] or c['original_fen']!=original['fen']:raise ValueError('Correction does not match original label')
            if c['reviewer_kind']!='human' or not c.get('approved_by') or not c.get('approval_statement'):raise ValueError('Explicit human correction approval required')
            fens[sid]=c['corrected_fen']
    return {sid:[v.class_id for row in labels_from_fen(fen) for v in row] for sid,fen in fens.items()},fens


def run(evidence_path,output,corrections_path=None):
    if output.exists():raise ValueError('Fresh analysis directory required')
    started=time.perf_counter();evidence=read(evidence_path)
    if evidence['schema_version']!='offline_square_evidence_d2_v1' or evidence['fusion_implemented'] or evidence['final_holdout']:raise ValueError('Invalid evidence scope')
    if sha256_file(evidence['observations_path'])!=evidence['observations_sha256']:raise ValueError('Observations changed')
    packet=read(evidence['packet_path'])
    if sha256_file(evidence['packet_path'])!=evidence['packet_sha256']:raise ValueError('Review packet changed')
    catalog_path=read(packet['config_path'])['source_catalog']
    if sha256_file(catalog_path)!=packet['protected_sha256'][catalog_path]:raise ValueError('Ground-truth source changed')
    catalog=read(catalog_path)
    truths,evaluated_fens=evaluation_truths(catalog,sha256_file(catalog_path),read(corrections_path) if corrections_path else None)
    obs=[json.loads(line) for line in Path(evidence['observations_path']).read_text().splitlines()]
    validate_bundle(evidence,obs)
    by_frame={}
    for o in obs:
        validate_observation(o);by_frame.setdefault((o['session_id'],o['frame_id']),{})[o['canonical_square']]=o
    for o in by_frame.values():
        if set(o)!=set(SQUARES):raise ValueError('Incomplete canonical frame')
    output.mkdir(parents=True,exist_ok=False);(output/'examples').mkdir();results=[];pairs=[];comparisons=[];diagnostics=[];contacts=[]
    pooled={c:dict(truth=[],predictions=[]) for c in ('diversity_5','uniform_5','diversity_10')}
    for session in evidence['sessions']:
        sid=session['capture_session_id'];truth=truths[sid];session_results={}
        for condition in ('diversity_5','uniform_5','diversity_10'):
            selected=[f for f in evidence['frames'] if f['session_id']==sid and condition in f['conditions']]
            usable=[f for f in selected if f['observations']]
            p=[[by_frame[(sid,f['frame_id'])][s]['predicted_class_id'] for s in SQUARES] for f in usable]
            metrics=multi_view_metrics(truth,np.asarray(p,dtype=int).reshape(-1,64))
            for pred in p:pooled[condition]['truth'].extend(truth);pooled[condition]['predictions'].extend(pred)
            frame_metrics=[]
            for frame,pred in zip(usable,p):
                fm=classification_metrics(truth,pred);frame_metrics.append(dict(frame_id=frame['frame_id'],timestamp_ms=frame['timestamp_ms'],**fm))
            metrics.update(session_id=sid,position_id=session['position_id'],condition=condition,selected_views=len(selected),
                           geometry_status_counts=dict(Counter(f['geometry_status'] for f in selected)),
                           excluded_frames=[dict(frame_id=f['frame_id'],status=f['geometry_status']) for f in selected if not f['observations']],
                           requested_budget=10 if condition=='diversity_10' else 5,frame_metrics=frame_metrics,
                           inference_frame_ids=[f['frame_id'] for f in usable],all_selected_human_verified=len(usable)==len(selected))
            results.append(metrics);session_results[condition]=metrics
            these_pairs=[]
            for i,a in enumerate(usable):
                for j in range(i+1,len(usable)):
                    b=usable[j];row=dict(session_id=sid,condition=condition,frame_a=a['frame_id'],frame_b=b['frame_id'],
                        view_distance=view_distance(a['d1_metadata'],b['d1_metadata']),
                        temporal_gap_seconds=abs(a['timestamp_ms']-b['timestamp_ms'])/1000,
                        same_predicted_orientation=by_frame[(sid,a['frame_id'])][SQUARES[0]]['semantic_orientation']==by_frame[(sid,b['frame_id'])][SQUARES[0]]['semantic_orientation'],
                        **paired_errors(truth,p[i],p[j]))
                    occupied=np.asarray(truth)!=0
                    row['occupied_squares']=paired_errors(np.asarray(truth)[occupied],np.asarray(p[i])[occupied],np.asarray(p[j])[occupied]) if occupied.any() else None
                    these_pairs.append(row);pairs.append(row)
            metrics['view_distance_vs_error_overlap']=descriptive_correlation([x['view_distance'] for x in these_pairs],[x['error_jaccard'] for x in these_pairs])
            metrics['view_distance_vs_error_association']=descriptive_correlation([x['view_distance'] for x in these_pairs],[x['binary_error_phi'] for x in these_pairs])
            metrics['view_distance_vs_prediction_disagreement']=descriptive_correlation([x['view_distance'] for x in these_pairs],[x['prediction_disagreement_rate'] for x in these_pairs])
            metrics['temporal_gap_vs_error_association']=descriptive_correlation([x['temporal_gap_seconds'] for x in these_pairs],[x['binary_error_phi'] for x in these_pairs])
            metrics['pairwise_summary']=dict(pair_count=len(these_pairs),mean_prediction_disagreement_rate=float(np.mean([x['prediction_disagreement_rate'] for x in these_pairs])) if these_pairs else None,
                mean_error_jaccard=float(np.mean([x['error_jaccard'] for x in these_pairs if x['error_jaccard'] is not None])) if any(x['error_jaccard'] is not None for x in these_pairs) else None)
            # All-square evidence matrix: counts and view-by-view labels, never a fused prediction.
            table=['<tr><th>square / truth</th>'+''.join('<th>'+html.escape(f"{f['timestamp_ms']/1000:.3f}s")+'</th>' for f in usable)+'<th>correct / views</th></tr>']
            for k,square in enumerate(SQUARES):
                vals=[pred[k] for pred in p];n=sum(v==truth[k] for v in vals)
                table.append('<tr><td>'+square+' / '+CLASS_NAMES[truth[k]]+'</td>'+''.join(f'<td style="background:{"#dbf4dc" if v==truth[k] else "#ffd8d8"}">{CLASS_NAMES[v]}</td>' for v in vals)+f'<td>{n}/{len(vals)}</td></tr>')
            name=sid+'_'+condition+'.html';(output/name).write_text('<!doctype html><meta charset="utf-8"><h1>'+html.escape(sid+' '+condition)+'</h1><p>Human-verified geometry only; per-view predictions, not fusion. Green=GT correct, red=wrong. Blank views have no evidence.</p><table border="1">'+''.join(table)+'</table>')
            contacts.append(name)
            # Contact sheets annotate per-frame errors; original photos retain camera-facing appearance.
            tiles=[]
            for frame,fm in zip(usable,frame_metrics):
                image=cv2.imread(frame['d1_metadata']['path']);image=cv2.resize(image,(270,480));tile=np.full((520,270,3),250,np.uint8);tile[40:]=image
                cv2.putText(tile,f"{frame['timestamp_ms']/1000:.3f}s errors {64-fm['correct']}/64",(3,15),cv2.FONT_HERSHEY_SIMPLEX,.4,(10,10,10),1)
                cv2.putText(tile,f"occupied {fm['occupied_correct']}/{fm['occupied_count']}",(3,33),cv2.FONT_HERSHEY_SIMPLEX,.4,(10,10,10),1);tiles.append(tile)
            if tiles:cv2.imwrite(str(output/(sid+'_'+condition+'_contact.jpg')),cv2.hconcat(tiles))
            if condition=='diversity_5' and metrics.get('squares'):
                candidates=sorted(metrics['squares'],key=lambda x:(x['true_class_id']==0,x['correct_view_count'],SQUARES.index(x['canonical_square'])))
                for category,criterion in [
                    ('rescue',lambda s:0<s['correct_view_count']<len(usable)),
                    ('persistent',lambda s:s['wrong_in_every_view'] and len(set(s['predictions']))==1),
                    ('black_rook',lambda s:s['true_class_id']==10),
                    ('empty_false_positive',lambda s:s['true_class_id']==0 and any(p!=0 for p in s['predictions'])),
                    ('same_prediction_despite_diversity',lambda s:s['true_class_id']!=0 and len(set(s['predictions']))==1 and len(usable)>1)]:
                    hit=next((s for s in candidates if criterion(s)),None)
                    if hit:
                        square=hit['canonical_square'];filename=sid+'_'+category+'_'+square+'.png'
                        draw_crop_distributions([by_frame[(sid,f['frame_id'])][square] for f in usable],
                            f"{category} {square} truth={CLASS_NAMES[hit['true_class_id']]} — raw uncalibrated probabilities",output/'examples'/filename)
                        diagnostics.append(dict(session_id=sid,category=category,canonical_square=square,true_class_id=hit['true_class_id'],
                                                correct_views=hit['correct_view_count'],observations=[by_frame[(sid,f['frame_id'])][square]['observation_id'] for f in usable],path=str(output/'examples'/filename)))
                    else:diagnostics.append(dict(session_id=sid,category=category,status='no_matching_example; none fabricated'))
        comparisons.append(dict(session_id=sid,position_id=session['position_id'],
            uniform5_to_diversity5=paired_coverage(session_results['uniform_5'],session_results['diversity_5']),
            diversity5_to_diversity10=paired_coverage(session_results['diversity_5'],session_results['diversity_10'])))
    aggregates=[]
    for condition,pooled_rows in pooled.items():
        rows=[r for r in results if r['condition']==condition];observed=[r for r in rows if r['usable_views']>0]
        classification=classification_metrics(pooled_rows['truth'],pooled_rows['predictions']) if pooled_rows['truth'] else None
        total_ever=sum(r['ever_correct_squares'] for r in observed);square_denominator=64*len(observed)
        class_coverage=[]
        for k in range(13):
            groups=[r['physical_class_coverage'][k] for r in observed]
            support=sum(g['physical_square_support'] for g in groups);ever=sum(g['ever_correct'] for g in groups)
            class_coverage.append(dict(class_id=k,physical_square_support=support,ever_correct=ever,
                                       ever_correct_fraction=ever/support if support else None,persistently_wrong=support-ever))
        aggregates.append(dict(condition=condition,physical_session_count=len(rows),physical_sessions_with_verified_views=len(observed),
            selected_frame_memberships=sum(r['selected_views'] for r in rows),usable_frame_memberships=sum(r['usable_views'] for r in rows),
            classification_metrics=classification,ever_correct_squares=total_ever,physical_square_denominator=square_denominator,
            ever_correct_fraction=total_ever/square_denominator if square_denominator else None,
            persistently_wrong_squares=sum(r['persistently_wrong_squares'] for r in observed),
            oracle_exact_board_count=sum(r['oracle_exact_board'] for r in observed),oracle_board_denominator=len(observed),
            exact_single_frame_boards=sum(r.get('exact_single_frame_boards',0) for r in rows),physical_class_coverage=class_coverage,
            interpretation='Pooled descriptive counts only; per-session comparisons are primary, not independent frames or squares.'))
    report=dict(schema_version='offline_evidence_characterization_d2_v1',designation='development_only',evidence_path=str(evidence_path),
        evidence_sha256=sha256_file(evidence_path),analysis_source_sha256=sha256_file(Path(__file__)),
        metrics_source_sha256=sha256_file('src/scanning/square_evidence.py'),source_catalog_sha256=sha256_file(read(packet['config_path'])['source_catalog']),
        ground_truth_corrections_artifact=str(corrections_path) if corrections_path else None,
        ground_truth_corrections_sha256=sha256_file(corrections_path) if corrections_path else None,evaluated_fens=evaluated_fens,
        conditions=results,condition_aggregates=aggregates,paired_session_comparisons=comparisons,pairwise_diagnostics=pairs,crop_examples=diagnostics,
        uncertainty='Five correlated physical sessions; descriptive paired results only, no frame/square-IID significance claims.',
        oracle_warning='GT upper bound only. Not implemented fusion, attainable performance, or deployment logic.',
        independence_warning='Probabilities from correlated video frames must not be treated as conditionally independent; do not multiply them.',
        analysis_seconds=time.perf_counter()-started,storage_bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
    write(output/'characterization.json',report)
    fields=['session_id','condition','selected_views','usable_views','all_selected_human_verified','overall_accuracy','occupied_accuracy',
            'empty_false_positives','occupied_to_empty_misses','incorrect_piece_identities','exact_single_frame_boards',
            'ever_correct_squares','persistently_wrong_squares','oracle_incorrect_squares','oracle_exact_board']
    with (output/'comparison.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in results)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>D.2 evidence characterization — no fusion</h1><p>'+html.escape(report['independence_warning'])+'</p><ul>'+''.join(f'<li><a href="{p}">{p}</a></li>' for p in contacts)+'</ul><h2>Crop distributions</h2>'+''.join(f'<p>{html.escape(str(d))}</p>' for d in diagnostics))
    print('Characterized',len(results),'paired session/condition groups')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--ground-truth-corrections',type=Path);a=p.parse_args();run(a.evidence,a.output_dir,a.ground_truth_corrections)
