"""Summarize fixed C.2 predictions and publish bounded visual diagnostics."""
import argparse
import csv
import gzip
import html
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from scripts.benchmark_chessqueries_c2 import configuration, load_cohort, read, write_fresh, checked
from src.chess.labels import CLASS_NAMES
from src.scanning.square_evidence import SQUARES, classification_metrics
from src.vision.chessqueries_offline import map_outputs, sha256


def placement(ids):
    symbols='.PNBRQKpnbrqk'
    ranks=[]
    for start in range(0,64,8):
        text='';empty=0
        for k in ids[start:start+8]:
            if k==0:empty+=1
            else:
                if empty:text+=str(empty);empty=0
                text+=symbols[k]
        if empty:text+=str(empty)
        ranks.append(text)
    return '/'.join(ranks)


def metrics(frames, predictions, key):
    y=np.asarray([f['truth'] for f in frames]);p=np.asarray([predictions[f['frame_id']][key] for f in frames])
    return dict(classification_metrics(y,p),frames=len(frames),exact_boards=int((y==p).all(axis=1).sum()))


def paired(truth,a,b):
    y=np.asarray(truth);a=np.asarray(a);b=np.asarray(b)
    return dict(corrected=int(((a!=y)&(b==y)).sum()),regressed=int(((a==y)&(b!=y)).sum()),
                both_correct=int(((a==y)&(b==y)).sum()),both_wrong=int(((a!=y)&(b!=y)).sum()),
                predictions_changed=int((a!=b).sum()),net_correct=int((b==y).sum()-(a==y).sum()))


def board_svg(ids, truth, baseline=None, cq=None):
    unicode_pieces=['','♙','♘','♗','♖','♕','♔','♟','♞','♝','♜','♛','♚']
    parts=['<svg viewBox="0 0 350 350" role="img" aria-label="Canonical board a8 to h1">']
    for i,k in enumerate(ids):
        row,col=divmod(i,8);x,y=15+40*col,15+40*row
        fill='#ece1cc' if (row+col)%2==0 else '#ac9577'
        parts.append(f'<rect x="{x}" y="{y}" width="40" height="40" fill="{fill}"/>')
        color=None
        if baseline is not None and cq is not None:
            if baseline[i]!=truth[i] and cq[i]==truth[i]:color='#078547'
            elif baseline[i]==truth[i] and cq[i]!=truth[i]:color='#de2939'
            elif baseline[i]!=truth[i] and cq[i]!=truth[i]:color='#bd8700'
        elif k!=truth[i]:color='#de2939'
        if color:parts.append(f'<rect x="{x+2}" y="{y+2}" width="36" height="36" fill="none" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+20}" y="{y+30}" font-size="32" text-anchor="middle" fill="#111">{unicode_pieces[k]}</text>')
    for i in range(8):
        parts.append(f'<text x="{35+40*i}" y="348" font-size="11" text-anchor="middle">{"abcdefgh"[i]}</text>')
        parts.append(f'<text x="6" y="{40+40*i}" font-size="11">{8-i}</text>')
    return ''.join(parts)+'</svg>'


def visual_packet(output, frames, predictions, case_studies):
    if output.exists():raise ValueError('Fresh visual packet required')
    output.mkdir(parents=True)
    by_id={f['frame_id']:f for f in frames};selected=[];reasons={}
    for sid in sorted({f['session_id'] for f in frames}):
        these=[f for f in frames if f['session_id']==sid]
        row=max(these,key=lambda f:(sum(a!=b for a,b in zip(f['baseline_class_ids'],predictions[f['frame_id']]['cq'])),f['frame_id']))
        selected.append(row['frame_id']);reasons[row['frame_id']]='Largest prediction disagreement in this session; illustrative, not model selection.'
    for c in case_studies:
        if c['case_id'] not in ('D2_king_rook','D2_knight_rook','D2_black_rook_king','D2_empty_rook','D2_empty_correct','D2_pawn_missed'):continue
        fid=c['frame_id']
        if fid not in selected:selected.append(fid)
        reasons[fid]=reasons.get(fid,'')+' C.1 case: '+c['case_id']+'.'
    sections=[]
    for fid in selected:
        f=by_id[fid];p=predictions[fid];truth=f['truth'];b=f['baseline_class_ids'];q=p['cq']
        image=cv2.imread(f['frame_path']);height,width=image.shape[:2];scale=min(1.,640/max(height,width))
        image=cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        photo=output/(fid+'.jpg');cv2.imwrite(str(photo),image,[cv2.IMWRITE_JPEG_QUALITY,86])
        grid=read(f['geometry_reference'])['policies']['marked-board'];labels=grid['semantic_square_ids']
        overlay=[f'<svg viewBox="0 0 {width} {height}" style="max-height:550px;width:100%"><image href="{photo.name}" width="{width}" height="{height}"/>']
        for square,polygon in zip(labels,grid['source_square_polygons']):
            pts=' '.join(f'{x},{y}' for x,y in polygon);center=np.mean(polygon,axis=0)
            overlay.append(f'<polygon points="{pts}" stroke="lime" stroke-width="1.4" fill="none"/><text x="{center[0]}" y="{center[1]}" fill="cyan" font-size="15" text-anchor="middle" stroke="#111" stroke-width=".3">{square}</text>')
        overlay.append('</svg>')
        rows=[]
        for i,square in enumerate(SQUARES):
            if b[i]!=truth[i] or q[i]!=truth[i]:
                status='corrected' if b[i]!=truth[i] and q[i]==truth[i] else ('introduced' if b[i]==truth[i] else 'both wrong')
                rows.append(f'<tr><td>{square}</td><td>{CLASS_NAMES[truth[i]]}</td><td>{CLASS_NAMES[b[i]]}</td><td>{CLASS_NAMES[q[i]]}</td><td>{status}</td></tr>')
        cases=[]
        for c in case_studies:
            if c['frame_id']!=fid:continue
            index=SQUARES.index(c['square']);scores=p['probabilities'][index]
            bars=''.join(f'<div>{CLASS_NAMES[k]} <meter min="0" max="1" value="{v}"></meter> {v:.3f}</div>' for k,v in enumerate(scores))
            cases.append(f'<details><summary>{c["case_id"]} {c["square"]}: B2 {CLASS_NAMES[c["b2"]]}, CQ {CLASS_NAMES[c["cq"]]}, label {CLASS_NAMES[c["truth"]]}</summary><p>Uncalibrated CQ softmax. Reused C.1 crop visualization, not CQ input:</p><img style="width:100%" src="../input_representation_c1_examples/{c["case_id"]}_variants.jpg">{bars}</details>')
        delta=paired(truth,b,q)
        sections.append(f'<section id="{fid}"><h2>{html.escape(fid)}</h2><p>{f["timestamp_ms"]/1000:.3f}s · reviewed {f["semantic_orientation"]} · {html.escape(reasons[fid])} Corrected {delta["corrected"]}; introduced {delta["regressed"]}; both wrong {delta["both_wrong"]}.</p><div class="panels"><div><h3>Original photo</h3><img style="width:100%;max-height:500px;object-fit:contain" src="{photo.name}"></div><div><h3>Registered ground truth</h3>{board_svg(truth,truth)}</div><div><h3>Frozen B.2</h3>{board_svg(b,truth)}</div><div><h3>ChessQueries</h3>{board_svg(q,truth,b,q)}</div></div><p>Placement FENs: GT <code>{placement(truth)}</code><br>B2 <code>{placement(b)}</code><br>CQ <code>{placement(q)}</code>. No game-state fields inferred.</p><details><summary>Reviewed source grid (diagnostic only; not CQ input)</summary>{"".join(overlay)}</details><details><summary>All incorrect or changed squares</summary><table><tr><th>Square</th><th>Label</th><th>B2</th><th>CQ</th><th>Comparison</th></tr>{"".join(rows)}</table></details>{"".join(cases)}</section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>C.2 zero-shot comparison</title><style>body{font:15px system-ui;max-width:1500px;margin:auto;padding:20px}.panels{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px}svg{width:100%}table{border-collapse:collapse}td,th{padding:4px;border:1px solid #ccc}section{border-top:1px solid #aaa;padding:15px 0}code{overflow-wrap:anywhere}meter{width:200px}@media(max-width:800px){.panels{grid-template-columns:repeat(2,1fr)}}</style><h1>C.2 frozen B.2 versus zero-shot ChessQueries ViT-S</h1><p><a href="../chessqueries_zero_shot_c2.md">Methodology and all-frame results</a>. All 55 frames were evaluated. This bounded gallery selects session disagreement examples and registered C.1 cases after inference; it is not an accuracy sample. CQ board outlines: green corrected B2 error, red introduced error, amber both wrong. B2 outlines: red error. Qualitative D.2 grid review is not subpixel/piece-contact verification; h4 boundary ambiguity remains. Labels use the owner-approved c7 supplement. All source images are unchanged; only small previews are stored.</p>'+''.join(sections))
    return selected


def run(config_path, predictions_path, result_path, examples):
    if result_path.exists() or examples.exists():raise ValueError('Fresh report paths required')
    config=configuration(config_path);manifest,frames,fens=load_cohort(config)
    raw=json.loads(gzip.decompress(predictions_path.read_bytes()))
    if raw['config_sha256']!=sha256(config_path) or raw['release_sha256']!=sha256(config['release']):raise ValueError('Prediction configuration mismatch')
    lookup={p['frame_id']:p for p in raw['predictions']}
    if len(lookup)!=len(raw['predictions']) or set(lookup)!={f['frame_id'] for f in frames}:raise ValueError('No missing, duplicate or extra frames permitted')
    predictions={};records=[]
    for f in frames:
        p=lookup[f['frame_id']]
        if p['frame_sha256']!=f['frame_sha256'] or p['session_id']!=f['session_id']:raise ValueError('Mismatched source frame/session')
        recomputed=map_outputs([p['raw_logits']],squares=p['canonical_squares'])
        if recomputed['predicted_class_ids']!=p['predicted_class_ids'] or not np.allclose(recomputed['probabilities'],p['probabilities'],atol=1e-12,rtol=0):raise ValueError('Invalid stored logits/probabilities')
        predictions[f['frame_id']]=dict(b2=f['baseline_class_ids'],cq=p['predicted_class_ids'],probabilities=p['probabilities'])
        records.append(dict(f, cq_class_ids=p['predicted_class_ids'],
                            b2=classification_metrics(f['truth'],f['baseline_class_ids']),
                            cq=classification_metrics(f['truth'],p['predicted_class_ids']),
                            paired=paired(f['truth'],f['baseline_class_ids'],p['predicted_class_ids']),
                            b2_exact=f['truth']==f['baseline_class_ids'],cq_exact=f['truth']==p['predicted_class_ids']))
    aggregate={key:metrics(frames,predictions,key) for key in ('b2','cq')}
    sessions=[]
    for sid in sorted({f['session_id'] for f in frames}):
        these=[f for f in frames if f['session_id']==sid]
        sessions.append(dict(session_id=sid,position_id=these[0]['position_id'],frame_count=len(these),
                             **{key:metrics(these,predictions,key) for key in ('b2','cq')}))
    groups=[]
    for orientation in sorted({f['semantic_orientation'] for f in frames}):
        these=[f for f in frames if f['semantic_orientation']==orientation]
        groups.append(dict(semantic_orientation=orientation,frames=len(these),
                           **{key:metrics(these,predictions,key) for key in ('b2','cq')}))
    y=[f['truth'] for f in frames];b=[f['baseline_class_ids'] for f in frames];q=[predictions[f['frame_id']]['cq'] for f in frames]
    cases=[]
    for c in read('docs/evaluation/input_representation_c1_examples/audit.json')['cases']:
        if not c['case_id'].startswith('D2_'):continue
        f=next(f for f in frames if f['frame_id']==c['frame_id']);i=SQUARES.index(c['canonical_square'])
        cases.append(dict(case_id=c['case_id'],frame_id=c['frame_id'],square=c['canonical_square'],
                          truth=f['truth'][i],b2=f['baseline_class_ids'][i],cq=predictions[c['frame_id']]['cq'][i]))
    timings={name:dict(median=statistics.median(p[name] for p in raw['predictions']),
                       mean=statistics.mean(p[name] for p in raw['predictions']),
                       p95=float(np.percentile([p[name] for p in raw['predictions']],95)))
             for name in ('forward_seconds','preprocessing_seconds','inference_and_preprocessing_seconds')}
    historical=read('outputs/scanning/d2_pilot_run1/evidence.json')
    if historical['observations_sha256']!=manifest['observations_sha256']:raise ValueError('Historical B2 timing population mismatch')
    old={f['frame_id']:f for f in historical['frames']}
    timings['b2_historical_forward_seconds_median']=statistics.median(old[f['frame_id']]['forward_seconds'] for f in frames)
    timings['b2_historical_extraction_seconds_median']=statistics.median(old[f['frame_id']]['extraction_seconds'] for f in frames)
    timings['b2_historical_normalization_seconds_median']=statistics.median(old[f['frame_id']]['normalization_seconds'] for f in frames)
    selected=visual_packet(examples,frames,predictions,cases)
    result=dict(schema_version='chessqueries_zero_shot_c2_results_v1',scope='five_correlated_development_sessions_not_independent_generalization',
                attempted_frames=55,evaluated_frames=55,excluded_frames=[],session_count=5,
                label_version='owner_corrected_c7_supplement_v1',ground_truth_fens=fens,
                label_corrections_path=config['ground_truth_corrections'],label_corrections_sha256=sha256(config['ground_truth_corrections']),
                geometry_caveat='qualitatively human-reviewed corners/orientation/mapping; not pixel-perfect geometry or per-piece contact; h4 ambiguity preserved',
                preprocessing_confounds=['full image versus homography and square crops','different encoders, parameters, pretraining data and training distributions','different effective target detail and semantic-orientation mechanism'],
                orientation_policy=config['orientation_policy'],orientation_realignment_attempted=False,
                aggregate=aggregate,per_session=sessions,by_reviewed_camera_orientation=groups,
                session_macro_average={key:{metric:statistics.mean(s[key][metric] for s in sessions) for metric in ('overall_accuracy','occupied_accuracy')} for key in ('b2','cq')},
                paired=paired(y,b,q),frames=records,c1_case_comparison=cases,
                timings=timings,model_load_seconds=raw['model_load_seconds'],benchmark_elapsed_seconds=raw['elapsed_seconds'],
                environment=raw['environment'],predictions_path=str(predictions_path),predictions_sha256=sha256(predictions_path),
                predictions_include='full 13-class logits and stable uncalibrated softmax; no fused/legality-repaired predictions',
                b2_predictions_reused=True,observations_sha256=manifest['observations_sha256'],
                inference_failures=0,external_geometry_used_by_chessqueries=False,
                chessqueries_internal_geometry_or_orientation_output_available=False,
                visual_frame_ids=selected,visual_selection='post-inference diagnostic only, no evaluation exclusions',
                source_sha256={p:sha256(p) for p in ['scripts/report_chessqueries_c2.py',str(config_path),'src/vision/chessqueries_offline.py','scripts/benchmark_chessqueries_c2.py']})
    write_fresh(result_path,result)
    csv_path=result_path.with_suffix('.csv')
    with csv_path.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['model','class_id','class_name','support','predicted_count','precision','recall'])
        for key,m in aggregate.items():
            for row in m['per_class']:writer.writerow([key,row['class_id'],CLASS_NAMES[row['class_id']],row['support'],row['predicted_count'],row['precision'],row['recall']])
    print(json.dumps(dict(aggregate={k:{n:m[n] for n in ('correct','square_count','overall_accuracy','occupied_correct','occupied_count','occupied_accuracy','empty_false_positives','empty_count','empty_false_positive_rate','exact_boards')} for k,m in aggregate.items()},paired=result['paired'],timings=timings,diagnostic_frames=len(selected)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/experiments/chessqueries_c2.json'))
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--examples',type=Path,required=True)
    a=parser.parse_args();run(a.config,a.predictions,a.results,a.examples)
