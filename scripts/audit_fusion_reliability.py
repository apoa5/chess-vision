"""Reproduce D.3 exactly, then audit preserved predictions without fitting thresholds."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read, write, check_hashes
from src.scanning.fusion_reliability import SIGNALS, uncertainty_signals, risk_curve, matched_coverage, board_summary
from src.scanning.fusion_metrics import reliability
from src.scanning.native_pilot import sha256_file
from src.chess.labels import CLASS_NAMES


def run(replay, output):
    if output.exists(): raise ValueError('Fresh audit output directory required')
    start = time.perf_counter()
    original = Path('outputs/scanning/fusion_d3_run1')
    index = read('data/metadata/fusion_d3_index_v1.json'); check_hashes(index['artifacts'])
    check_hashes(read(original/'inputs.json')['protected_sha256'])
    a = read(original/'predictions.json'); b = read(replay/'predictions.json')
    if a['boards'] != b['boards']: raise ValueError('D.3 prediction replay differs')
    result = read(original/'evaluation_v1/results.json'); reproduced = read(replay/'evaluation_v1/results.json')
    for key in ('aggregates','boards','paired_session_comparisons','correct_minority_overwhelmed','oracle_diagnostics_only','label_version'):
        if result[key] != reproduced[key]: raise ValueError('D.3 evaluation differs: '+key)
    settings = read('configs/experiments/fusion_reliability_d3b.json')
    evidence = read(result['config']['evidence_manifest']); check_hashes({evidence['observations_path']:evidence['observations_sha256']})
    obs = {o['observation_id']:o for o in map(json.loads,Path(evidence['observations_path']).read_text().splitlines())}
    rows=[]; boards=[]
    for board in result['boards']:
        if board['condition'] != 'uniform_5' or board['method'] not in settings['methods']: continue
        these=[]
        for s in board['squares']:
            views=[obs[i]['probabilities'] for i in s['observation_ids']]
            signals=uncertainty_signals(s['score_distribution'],views,s['predicted_class_id'])
            consensus=signals['strong_support_fraction'] >= settings['consensus_fraction']
            row=dict(method=board['method'],session_id=board['session_id'],position_id=board['position_id'],
                     canonical_square=s['canonical_square'],true_class_id=s['true_class_id'],predicted_class_id=s['predicted_class_id'],
                     correct=s['correct'],occupied=s['true_class_id']!=0,abstained=s['abstained'],score_distribution=s['score_distribution'],
                     observation_ids=s['observation_ids'],input_class_ids=s['input_class_ids'],input_frame_ids=s['input_frame_ids'],
                     high_confidence_consensus=consensus,incorrect_high_confidence_consensus=consensus and not s['correct'],
                     confidently_wrong=signals['maximum_probability']>=settings['high_confidence'] and not s['correct'],**signals)
            rows.append(row);these.append(row)
        boards.append(dict(method=board['method'],session_id=board['session_id'],**board_summary(these)))
    summaries=[]; curves=[];matched=[];board_probes=[]
    for method in settings['methods']:
        allrows=[r for r in rows if r['method']==method]
        strata=[('all',allrows),('occupied',[r for r in allrows if r['occupied']]),('empty',[r for r in allrows if not r['occupied']])]
        strata += [('class_'+str(k),[r for r in allrows if r['true_class_id']==k]) for k in range(13)]
        strata += [('session_'+sid,[r for r in allrows if r['session_id']==sid]) for sid in sorted({r['session_id'] for r in allrows})]
        for name,subset in strata:
            rel=reliability([r['true_class_id'] for r in subset],[r['score_distribution'] for r in subset],
                            [r['predicted_class_id'] for r in subset],[r['abstained'] for r in subset],result['config'])
            summaries.append(dict(method=method,stratum=name,count=len(subset),errors=sum(not r['correct'] for r in subset),
                confidently_wrong=sum(r['confidently_wrong'] for r in subset),wrong_consensus=sum(r['incorrect_high_confidence_consensus'] for r in subset),reliability=rel,
                error_groups={name:dict(count=sum(test(r) for r in subset),mean_probability=sum(r['maximum_probability'] for r in subset if test(r))/max(1,sum(test(r) for r in subset)))
                    for name,test in [('correct',lambda r:r['correct']),('confidently_incorrect',lambda r:r['confidently_wrong']),('low_confidence',lambda r:r['maximum_probability']<.8)]}))
            for signal,high in SIGNALS.items():
                curves.append(dict(method=method,stratum=name,signal=signal,higher_is_better=high,points=risk_curve(subset,signal,high)))
                if name in ('all','occupied','empty'):
                    for coverage in settings['matched_square_coverages']:
                        matched.append(dict(method=method,stratum=name,signal=signal,**matched_coverage(subset,signal,high,coverage)))
        for threshold in settings['exploratory_thresholds']:
            for signal in ('mean_probability','minimum_probability'):
                retained=[b for b in boards if b['method']==method and b[signal]>=threshold]
                board_probes.append(dict(method=method,signal=signal,threshold=threshold,exploratory=True,
                    retained=len(retained),total=5,incorrect_retained=sum(not b['exact_board'] for b in retained),session_ids=[b['session_id'] for b in retained]))
    boardcurves=[]
    for method in settings['methods']:
        subset=[dict(b,canonical_square='board',correct=b['exact_board']) for b in boards if b['method']==method]
        for signal,high in [('mean_probability',True),('minimum_probability',True),('minimum_margin',True),('maximum_entropy',False),('maximum_disagreement',False)]:
            boardcurves.append(dict(method=method,signal=signal,points=risk_curve(subset,signal,high),
                                    matched=[matched_coverage(subset,signal,high,k/5) for k in range(1,6)]))
    sensitivity=[]
    for method in settings['methods']:
        for sid in sorted({b['session_id'] for b in boards}):
            bs=[b for b in result['boards'] if b['method']==method and b['session_id']==sid and b['condition'] in ('uniform_5','diversity_10')]
            five=next(b for b in bs if b['condition']=='uniform_5');ten=next(b for b in bs if b['condition']=='diversity_10')
            sensitivity.append(dict(method=method,session_id=sid,five_frames=five['input_frame_count'],larger_frames=ten['input_frame_count'],
                nested=set(five['input_frame_ids'])<=set(ten['input_frame_ids']),
                rescued=[x['canonical_square'] for x,y in zip(five['squares'],ten['squares']) if not x['correct'] and y['correct']],
                regressed=[x['canonical_square'] for x,y in zip(five['squares'],ten['squares']) if x['correct'] and not y['correct']],
                newly_confident_wrong=[x['canonical_square'] for x,y in zip(five['squares'],ten['squares']) if not y['correct'] and y['maximum_score']>=.9 and (x['correct'] or x['maximum_score']<.9)],
                caveat='Different nonnested selectors; cannot isolate benefit of adding frames'))
    output.mkdir(parents=True)
    diagnostics=dict(schema_version='fusion_reliability_d3b_v1',independent_validation='pending',designation='development_only',
        exact_D3_replay=True,matched_prediction_boards=75,matched_evaluation_aggregates=15,label_version=result['label_version'],
        input_sha256={str(original/'predictions.json'):sha256_file(original/'predictions.json'),str(original/'evaluation_v1/results.json'):sha256_file(original/'evaluation_v1/results.json'),
            evidence['observations_path']:evidence['observations_sha256'],'configs/experiments/fusion_reliability_d3b.json':sha256_file('configs/experiments/fusion_reliability_d3b.json')},
        source_sha256={p:sha256_file(p) for p in ['scripts/audit_fusion_reliability.py','src/scanning/fusion_reliability.py']},
        session_count=5,class_names=list(CLASS_NAMES),square_diagnostics=rows,board_diagnostics=boards,strata=summaries,
        selective_risk_coverage=curves,matched_coverage=matched,board_probes=board_probes,board_curves=boardcurves,
        frame_budget_sensitivity=sensitivity,elapsed_seconds=time.perf_counter()-start,
        warning='Squares and frames are correlated; descriptive five-session development audit, not validated operating thresholds')
    write(output/'reliability_v1.json',diagnostics)
    for name,data in [('boards',boards),('matched_coverage',matched),('board_probes',board_probes)]:
        with (output/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print('Exact D.3 replay confirmed; audited 960 method/square diagnostics and 15 boards.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.replay,a.output_dir)
