"""Bounded/resumable offline pilot evaluation using the unchanged geometry evaluator."""
import argparse,copy,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from scripts.evaluate_pilot_geometry import evaluate_frame
from scripts.ingest_native_pilot import contact_sheet
from src.scanning.native_pilot import sha256_file
from src.utils.config import load_config


def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')


def check(manifest):
    for path,h in (manifest['protected_sha256']|manifest['media_sha256']).items():
        if sha256_file(path)!=h:raise ValueError('Protected input changed: '+path)


def counts(frames):
    return {p:dict(completed=sum(f['policies'][p]['completed'] for f in frames),failures=dict(Counter(f['policies'][p]['failure_stage'] for f in frames if not f['policies'][p]['completed'])),recovery_attempts=sum(f['policies'][p]['recovery_attempted'] for f in frames)) for p in ['legacy','marked-board']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--session-id');p.add_argument('--reuse-run',type=Path);p.add_argument('--assemble-parts',type=Path,nargs='+');a=p.parse_args()
    if a.output_dir.exists():p.error('Fresh output directory required')
    m=json.loads((a.native_run/'manifest.json').read_text())
    if m['designation']!='development_only' or m['training_eligible'] or m['final_holdout']:raise ValueError('Pilot only')
    check(m);cv2.setNumThreads(1);native_sha=sha256_file(a.native_run/'manifest.json');config_sha=sha256_file('config/settings.yaml');evaluator_sha=sha256_file('scripts/evaluate_pilot_geometry.py')
    report=dict(schema_version='pilot_geometry_audit_v1',native_manifest_sha256=native_sha,config_sha256=config_sha,source_sha256=evaluator_sha,bounded_runner_sha256=sha256_file(Path(__file__)),policies=['legacy','marked-board'],frames=[],reference_geometry_available=False,limitation='Completion is not verified correctness. No classifier loaded.')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    if a.assemble_parts:
        allrows={}
        for part in a.assemble_parts:
            d=json.loads((part/'results.json').read_text())
            if any(d[k]!=report[k] for k in ['native_manifest_sha256','config_sha256','source_sha256']):raise ValueError('Incompatible part')
            for f in d['frames']:
                key=(f['session_id'],f['candidate_number'])
                if key in allrows:raise ValueError('Duplicate pair across parts')
                allrows[key]=f
        for s in m['sessions']:
            for n,c in enumerate(s['candidates'],1):
                row=allrows.pop((s['capture_session_id'],n))
                if row['derivative_sha256']!=c['derivative_sha256']:raise ValueError('Mismatched frame')
                report['frames'].append(row)
        if allrows:raise ValueError('Extra frames')
        report['part_reports']={str(q/'results.json'):sha256_file(q/'results.json') for q in a.assemble_parts}
    else:
        sessions=[s for s in m['sessions'] if s['capture_session_id']==a.session_id]
        if len(sessions)!=1:p.error('Exact --session-id required')
        cache={}
        if a.reuse_run:
            d=json.loads((a.reuse_run/'results.json').read_text())
            if any(d[k]!=report[k] for k in ['native_manifest_sha256','config_sha256','source_sha256']):raise ValueError('Incompatible cache')
            cache={(r['session_id'],r['candidate_number']):r for r in d['frames']};report['reuse_report']=dict(path=str(a.reuse_run/'results.json'),sha256=sha256_file(a.reuse_run/'results.json'))
        s=sessions[0];config=load_config()
        for n,c in enumerate(s['candidates'],1):
            if sha256_file(c['path'])!=c['derivative_sha256']:raise ValueError('Changed input image')
            key=(s['capture_session_id'],n)
            if key in cache:
                row=copy.deepcopy(cache[key]);row['reused_unchanged_result']=True
                if row['derivative_sha256']!=c['derivative_sha256']:raise ValueError('Incompatible cache frame')
            else:
                row={k:c[k] for k in ['frame_id','path','pts','time_base','timestamp_ms','relative_timestamp_ms','derivative_sha256']};row.update(session_id=s['capture_session_id'],position_id=s['position_id'],candidate_number=n,policies={},reused_unchanged_result=False)
                for policy in ['legacy','marked-board']:row['policies'][policy]=evaluate_frame(c['path'],config,a.output_dir/f'candidate_{n:02d}'/policy,policy)
            if row['policies']['legacy']['completed']:
                if not row['policies']['marked-board']['completed'] or row['policies']['legacy']['source_square_polygons']!=row['policies']['marked-board']['source_square_polygons']:raise ValueError('Legacy-success parity failed')
            report['frames'].append(row);write(a.output_dir/'results.json',report);print(s['position_category'],n,[(k,v['completed'],v['failure_stage']) for k,v in row['policies'].items()],flush=True)
    report['counts']=counts(report['frames']);write(a.output_dir/'results.json',report)
    for s in m['sessions']:
        fs=[f for f in report['frames'] if f['session_id']==s['capture_session_id']]
        if fs:
            for policy in ['legacy','marked-board']:contact_sheet([(f['policies'][policy]['overlay_path'],f"{f['candidate_number']:02d} {f['relative_timestamp_ms']/1000:.2f}s") for f in fs],a.output_dir/f"{s['capture_session_id']}_{policy}_contact.png")
    check(m);write(a.output_dir/'preservation.json',dict(protected_inputs_unchanged=True,original_media_unchanged=True,legacy_success_grid_parity=True));print(json.dumps(report['counts']),flush=True)

if __name__=='__main__':main()
