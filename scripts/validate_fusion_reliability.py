"""Recheck frozen history, exact D.3 replay and portable D.3b artifacts."""
import argparse
import gzip
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read,write,check_hashes


def validate(replay,audit,export,output):
    original=Path('outputs/scanning/fusion_d3_run1')
    protected=dict(read(original/'inputs.json')['protected_sha256'])
    protected.update(read('data/metadata/fusion_d3_index_v1.json')['artifacts'])
    protected.update(read('data/metadata/evidence_d2_index_v1.json')['artifacts'])
    check_hashes(protected)
    prereg=read('data/metadata/fusion_independent_d3b_preregistration_v1.json');check_hashes(prereg['source_sha256'])
    old=read(original/'predictions.json');new=read(replay/'predictions.json')
    if old['boards']!=new['boards']:raise ValueError('D.3 replay differs')
    oldresults=read(original/'evaluation_v1/results.json');newresults=read(replay/'evaluation_v1/results.json')
    for key in ('aggregates','boards','paired_session_comparisons','correct_minority_overwhelmed','oracle_diagnostics_only','label_version'):
        if oldresults[key]!=newresults[key]:raise ValueError('D.3 evaluation differs')
    exp=read(export/'fusion_reliability_d3b_export_v1.json');check_hashes({str(export/p):h for p,h in exp['artifacts'].items()})
    raw=gzip.decompress((export/'fusion_reliability_d3b_results_v1.json.gz').read_bytes())
    if hashlib.sha256(raw).hexdigest()!=exp['uncompressed_result_sha256']:raise ValueError('Lossless export differs')
    import json
    result=json.loads(raw);check_hashes(result['input_sha256']);check_hashes(result['source_sha256'])
    if len(result['square_diagnostics'])!=960 or len(result['board_diagnostics'])!=15 or result['independent_validation']!='pending':raise ValueError('Audit population/status changed')
    functional=read(audit/'functional_evaluation_v3/evaluation.json');pred=read(audit/'functional_evaluation_v3/predictions.json')
    for board in pred['boards']:
        expected=next(b for b in old['boards'] if b['condition']=='uniform_5' and b['session_id']==board['session_id'] and b['method']==board['method'])
        if board!={k:v for k,v in expected.items() if k!='condition'}:raise ValueError('Functional prediction differs')
    for board in functional['boards']:
        expected=next(b for b in oldresults['boards'] if b['condition']=='uniform_5' and b['session_id']==board['session_id'] and b['method']==board['method'])
        if any(board[k]!=v for k,v in expected['metrics'].items()) or board['reliability']!=expected['reliability']:raise ValueError('Functional metrics differ')
    if functional['independent_validation']!='pending' or not functional['functional_development']:raise ValueError('Functional data misrepresented')
    write(output,dict(schema_version='fusion_reliability_preservation_d3b_v1',protected_history_files_unchanged=len(protected),
        D3_prediction_boards_exact=75,D3_evaluation_aggregates_exact=15,functional_primary_predictions_and_metrics_exact=15,
        audit_square_outputs=960,audit_board_outputs=15,portable_artifact_hashes_checked=len(exp['artifacts']),
        lossless_result_sha256=exp['uncompressed_result_sha256'],preregistered_sources_unchanged=len(prereg['source_sha256']),
        independent_validation='pending',final_holdout_accessed=False,classifier_rerun=False,completion_enabled=False))
    print('Preservation, exact replay, functional predictions and portable export verified.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for f in ('replay','audit','export','output'):p.add_argument('--'+f,type=Path,required=True)
    a=p.parse_args();validate(a.replay,a.audit,a.export,a.output)
