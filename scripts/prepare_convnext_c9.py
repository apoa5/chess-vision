"""C9 source inventory and bounded original-PTS candidate analysis; no auto ground truth."""
import argparse,copy,gzip,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.training.manifest_tools import read,write_new
from src.training.multidataset import digest,position_key
from src.scanning.interim_video import decode_exact,pixel_hash
from src.scanning.video_keyframes import temporal_candidates,board_features,geometry_state,near_duplicate,view_distance
from src.utils.config import load_config
ROOT=Path('outputs/evaluation/convnext_c9')

def inventory():
    if (ROOT/'inventory_v1.json').exists():raise ValueError('Fresh inventory required')
    photos=read('data/metadata/images.json');prepared={b['image_id']:b for b in read('data/classification/prepared_boards.json')};approved={b['source_id']:b for b in read('outputs/evaluation/multidataset_c4/personal_manifest_v1.json')['boards']}
    records=[]
    from PIL import Image
    for name,m in photos.items():
        if m.get('final_holdout') or m.get('split')=='final_holdout':
            records.append(dict(id=name,status='excluded_final_holdout',metadata_only=True));continue
        p=prepared[name];path=Path(m['path'])
        if not path.is_file():records.append(dict(id=name,status='missing',source_path=str(path)));continue
        sha=digest(path)
        if sha!=p['source_sha256']:raise ValueError('Original photo changed: '+name)
        with Image.open(path) as image:resolution=list(image.size)
        records.append(dict(id=name,source_path=str(path),sha256=sha,resolution=resolution,position_id=m['position_id'],session_id=m.get('capture_session_id'),
            fen=p['fen'],placement_key=position_key(p['fen']),orientation=m['orientation'],historical_split=approved.get(name,{}).get('split','excluded'),
            status='historical_owner_verified' if name in approved else 'quarantine_geometry_unverified',historical_exclusion_reason=p.get('reason'),
            geometry_source='data/classification/prepared_boards.json',geometry_detected=p['board_detected']))
    a3=read('data/metadata/a3_pilot_01_sources_v2.json');native=read(a3['primary_manifest']);fixes={x['position_id']:x for x in read('data/metadata/evidence_d2_ground_truth_corrections_v1.json')['corrections']}
    sessions=[]
    for s,n in zip(a3['sessions'],native['sessions']):
        source=s['primary_source'];fen=fixes.get(s['position_id'],{}).get('corrected_fen',s['fen'])
        sessions.append(dict(id=s['capture_session_id'],dataset='a3_pilot',position_id=s['position_id'],source=source,fen=fen,original_fen=s['fen'],
            label_version='owner_corrected_c7_d2_v1' if fen!=s['fen'] else 'original_owner_fen',placement_key=position_key(fen),index_path=n['index_path'],
            independence_group_id=s['independence_group_id'],reference=s['reference'],timeline_kind='a3',
            historical_use='development_only_previously_training_prohibited',new_use='C9_authorized_candidate_only_requires_frame_review'))
    d3=read('outputs/scanning/d3c_interim_v1/registration.json')
    for s in d3['sessions']:
        sessions.append(dict(id=s['session_id'],dataset='d3c_retired',position_id=s['position_id'],source=s['source'],fen=s['fen'],original_fen=s['fen'],
            label_version='owner_supplied_verified_C4',placement_key=position_key(s['fen']),index_path=f"outputs/scanning/d3c_interim_v1/{s['session_id']}/index.json.gz",
            independence_group_id=None,reference=None,timeline_kind='d3c',historical_use='retired_independent_C4_C7_training',new_use='C9_authorized_candidate_requires_frame_review'))
    for s in sessions:
        if s['source'].get('final_holdout'):raise ValueError('Protected source metadata encountered')
        path=Path(s['source']['path']);s['available']=path.is_file()
        if s['available'] and digest(path)!=s['source']['sha256']:raise ValueError('Original video changed')
    manifest=dict(schema_version='convnext_c9_inventory_v1',photos=records,sessions=sessions,final_holdout_access=False,compressed_substitution=False)
    write_new(ROOT/'inventory_v1.json',manifest)
    write_new('data/metadata/convnext_c9_data_use_v1.json',dict(schema_version='convnext_c9_authority_v1',authority='Explicit owner C9 authorization',
        scope='New dataset only: all180 photos and15 original videos may be considered; historical validation/test training restrictions retired for C9, not historical results.',
        historical_files_modified=False,final_holdout_protected=True,independent_generalization=False,inventory_path=str(ROOT/'inventory_v1.json'),inventory_sha256=digest(ROOT/'inventory_v1.json'),
        a3_c7_correction='Reuses explicit owner-approved D2 c7 black pawn correction; original FEN/results preserved.',frame_review_required=True))
    print('Inventoried',len(records),'photos and',len(sessions),'original recordings',flush=True)


def candidates(args):
    session,interval=args;cv2.setNumThreads(1)
    from scripts.evaluate_independent_d3c import geometry_record
    settings=read('configs/experiments/keyframes_d1.json');dest=ROOT/'candidates_v1'/session['id'];dest.mkdir(parents=True,exist_ok=True)
    idx=Path(session['index_path']);index=json.loads(gzip.decompress(idx.read_bytes())) if idx.suffix=='.gz' else read(idx)
    timeline=index['frames'] if session['timeline_kind']=='d3c' else index['original_frame_index']
    chosen=temporal_candidates(timeline,interval,1024);source=session['source']['path'];cached={}
    if session['timeline_kind']=='d3c':
        for p in Path('outputs/scanning/d3c_interim_v1',session['id']).glob('candidate_*.json'):
            row=read(p);cached[tuple(row['pts_time'])]=row
    else:
        # Existing D1 reviewed frame candidates contain exactly the same native display convention.
        f=read('outputs/scanning/d1_pilot_run1/selection_run2/features.json')
        for d in f['datasets']:
            if d['name']=='primary_3s_high':
                for s in d['sessions']:
                    if s['capture_session_id']==session['id']:
                        for row in s['policies']['marked-board']:
                            if row.get('geometry_reference'):
                                pair=read(row['geometry_reference']);g=pair['policies']['marked-board']
                                cached[tuple(row['pts_time'])]=dict(row,geometry=g)
    rows=[]
    for i,frame in enumerate(chosen):
        frame=dict(frame)
        if session['timeline_kind']=='a3':
            frame['historical_index_display_rotation_degrees']=frame.get('display_rotation_degrees')
            frame['display_rotation_degrees']=frame.get('encoded_rotation_degrees',session['source']['encoded_rotation_degrees'])
        target=dest/f'candidate_{i:04d}.json'
        if target.exists():rows.append(read(target));continue
        prior=cached.get(tuple(frame['pts_time']))
        if prior is not None:
            row=copy.deepcopy(prior);row.update(frame);row['reused_geometry']=True
            # D1 feature records already carry structural quality eligibility and source-pixel hash.
            if 'source_video_sha256' not in row:row['source_video_sha256']=session['source']['sha256']
        else:
            image=decode_exact(source,frame);row=dict(frame,session_id=session['id'],source_video_sha256=session['source']['sha256'],
                decoded_pixel_sha256=pixel_hash(image),derivative_sha256=pixel_hash(image))
            g=geometry_record(image,row,load_config());binding={k:row[k] for k in ['session_id','frame_id','pts','time_base','source_video_sha256','derivative_sha256']}
            state,reasons=geometry_state(row,g,binding);row.update(geometry=g,geometry_state=state,unusable_reasons=reasons,eligible=False,quality_score=None,reused_geometry=False)
            if state=='completed_unverified':
                try:row.update(board_features(image,g,settings));row['eligible']=not row['quality_rejection_reasons']
                except ValueError as e:row.update(eligible=False,unusable_reasons=[str(e)])
        row['session_id']=session['id'];row['geometry_verification_status']='unreviewed_C9_candidate';write_new(target,row);rows.append(row)
        if (i+1)%10==0:print(session['id'],i+1,'/',len(chosen),flush=True)
    # Adaptive greedy novelty, no fixed 5/10/20 budget or quota. Views must pass
    # unchanged D1 quality checks. The novelty threshold is a proxy, not a pose.
    eligible=[r for r in rows if r.get('eligible') and r.get('quality_score') is not None];selected=[]
    for row in sorted(eligible,key=lambda r:(-r['quality_score'],r['timestamp_ms'])):
        if any(abs(row['timestamp_ms']-s['timestamp_ms'])<1500 for s in selected):continue
        if any(near_duplicate(row,s,settings) for s in selected):continue
        if selected and min(view_distance(row,s) for s in selected)<.035:continue
        selected.append(row)
    summary=dict(session_id=session['id'],source_sha256=session['source']['sha256'],candidates=len(rows),eligible=len(eligible),adaptive_selected=[r['frame_id'] for r in selected],
        selected_count=len(selected),target_range=[20,40],target_is_quota=False,interval_ms=interval,quality_thresholds_changed=False,
        verification='All newly selected frames require explicit review; completion is not correctness')
    write_new(dest/'summary.json',summary);print('SESSION',session['id'],summary['selected_count'],'adaptive selected',flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','candidates']);p.add_argument('--workers',type=int,default=2);p.add_argument('--interval-ms',type=int,default=1000);a=p.parse_args()
    if a.action=='inventory':inventory()
    else:
        sessions=[s for s in read(ROOT/'inventory_v1.json')['sessions'] if s['available']]
        with ProcessPoolExecutor(max_workers=a.workers) as pool:result=list(pool.map(candidates,[(s,a.interval_ms) for s in sessions]))
        write_new(ROOT/'candidate_summary_v1.json',result)
