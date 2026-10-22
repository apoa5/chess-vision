"""Index higher-quality pilot representations and match the frozen 60 timestamps."""
import argparse,copy,hashlib,json,sys,time
from collections import Counter
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.ingest_native_pilot import contact_sheet
from scripts.validate_native_pilot import validate
from src.scanning.native_pilot import decoder,pts_record,sha256_file
from src.scanning.pilot_comparison import display_image,display_transform,thumbnail,correspondence_score,nearest_time,register_pair


def write(p,value):Path(p).write_text(json.dumps(value,indent=2)+'\n')


def ingest(old,source,dest):
    session_id=old['capture_session_id'];dest.mkdir(exist_ok=False)
    target=[Fraction(str(x['relative_timestamp_ms']))/1000 for x in old['candidates']]
    thumbs=[thumbnail(cv2.imread(x['path'])) for x in old['candidates']]
    best=[None]*len(target);best_images={};timeline=[];first=None;previous=None;rotation_seen=None
    start=time.perf_counter();av=decoder(); source_sha=sha256_file(source)
    with av.open(str(source)) as c:
        s=c.streams.video[0];s.thread_type='SLICE';s.codec_context.thread_count=1
        meta=dict(path=str(source),sha256=source_sha,bytes=source.stat().st_size,container_format=c.format.name,container_metadata=dict(c.metadata),container_duration_seconds=c.duration/av.time_base,
                  codec=s.codec_context.name,pixel_format=s.codec_context.format.name,encoded_width=s.width,encoded_height=s.height,declared_frames=s.frames,stream_time_base=str(s.time_base),stream_start_pts=s.start_time,stream_duration_pts=s.duration,stream_metadata=dict(s.metadata),average_rate=str(s.average_rate),nominal_rate=str(s.base_rate),streams=[dict(index=z.index,type=z.type,codec=z.codec_context.name if z.codec_context is not None else None,metadata=dict(z.metadata)) for z in c.streams])
        for i,f in enumerate(c.decode(s)):
            if f.pts is None:raise ValueError('Missing PTS')
            absolute=Fraction(f.pts)*f.time_base
            if first is None:first=absolute
            if previous is not None and absolute<=previous:raise ValueError('Nonmonotonic PTS')
            previous=absolute;relative=absolute-first;rotation=int(f.rotation)
            if rotation_seen is None:rotation_seen=rotation
            if rotation != rotation_seen:raise ValueError('Changing display transform')
            rotated=rotation in (-90,90,270,-270); w,h=(f.height,f.width) if rotated else (f.width,f.height)
            row=dict(frame_id=f'{session_id}_hd_frame_{i:06d}',decode_index=i,width=w,height=h,duration_pts=f.duration,display_rotation_degrees=0,encoded_rotation_degrees=rotation,**pts_record(f.pts,f.time_base,first));timeline.append(row)
            nearby=[j for j,t in enumerate(target) if abs(relative-t)<=Fraction(1,2)]
            if not nearby:continue
            image=display_image(f.to_ndarray(format='bgr24'),rotation); thumb=thumbnail(image)
            for j in nearby:
                score=correspondence_score(thumb,thumbs[j])
                if best[j] is None or score>best[j]['correspondence_score']:
                    best[j]=row|dict(correspondence_score=score,matched_low_frame_id=old['candidates'][j]['frame_id'],target_low_relative_timestamp_ms=float(target[j]*1000),match_offset_ms=float((relative-target[j])*1000));best_images[j]=image.copy()
    if len(timeline)!=meta['declared_frames']:raise ValueError('Incomplete decode')
    matrix=display_transform(meta['encoded_width'],meta['encoded_height'],rotation_seen)
    meta.update(display_width=timeline[0]['width'],display_height=timeline[0]['height'],encoded_rotation_degrees=rotation_seen,encoded_to_display_matrix=matrix.tolist())
    selected=[]
    for j,row in enumerate(best):
        if row is None or row['correspondence_score']<.85:raise ValueError('Insufficient cross-version correspondence')
        im=best_images[j];p=dest/(row['frame_id']+'.png');cv2.imwrite(str(p),im)
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        row.update(path=str(p),derivative_sha256=sha256_file(p),decoded_pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),source_video_sha256=source_sha,split='pilot',training_eligible=False,verified_corners_px=None,verified_semantic_orientation=None,photographic_placement_review='pending',camera_orientation=None,camera_elevation=None,review_status='pending',blur_score=float(cv2.Laplacian(gray,cv2.CV_64F).var()),mean_luma=float(gray.mean()),pair_registration=register_pair(im,cv2.imread(old['candidates'][j]['path'])))
        selected.append(row)
    times=[Fraction(*r['pts_time']) for r in timeline];intervals=[b-a for a,b in zip(times,times[1:])]
    result=dict(schema_version='native_pilot_session_v1',session_id=session_id,split='pilot',designation='development_only',source_video=meta,decoded_frames=len(timeline),all_frames_decoded=True,pts_strictly_increasing=True,first_pts_time=str(times[0]),last_pts_time=str(times[-1]),interval_seconds_histogram=dict(Counter(map(str,intervals))),observed_mean_fps=float((len(times)-1)/(times[-1]-times[0])),max_frame_gap_ms=float(max(intervals)*1000),original_frame_index=timeline,candidates=selected,verified_ground_truth=False,unchanged_position_review=old['unchanged_position_review'],display_transform_applied=True,selection_settings=dict(method='Maximum whole-frame correlation in +/-0.5s window around frozen low-resolution selection; no geometry/model feedback',max_frames=12),elapsed_seconds=time.perf_counter()-start)
    write(dest/'session.json',result)
    contact_sheet([(r['path'],f"{i+1:02d} {r['relative_timestamp_ms']/1000:.3f}s") for i,r in enumerate(selected)],dest/'candidates_contact.png')
    if sha256_file(source)!=source_sha:raise ValueError('Original changed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--low-manifest',type=Path,default=Path('data/metadata/a3_pilot_01_native.json'));p.add_argument('--source-dir',type=Path,default=Path('data/raw/a3_pilot_01'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():p.error('Fresh output directory required')
    cv2.setNumThreads(1);old=json.loads(a.low_manifest.read_text());manifest=copy.deepcopy(old);manifest.update(source_revision='hd_v1',primary_representation='high_quality',low_resolution_manifest=str(a.low_manifest),low_resolution_manifest_sha256=sha256_file(a.low_manifest),sessions=[])
    # Freeze all old artifacts plus original media and production/checkpoint hashes.
    protected=dict(old['protected_sha256'])
    for d in [Path('outputs/scanning/pilot_01/native_run1'),Path('outputs/scanning/pilot_01/geometry_run1'),Path('docs/evaluation/pilot_01_examples')]:
        protected.update({str(q):sha256_file(q) for q in d.rglob('*') if q.is_file()})
    protected.update(old['media_sha256']);protected[str(a.low_manifest)]=sha256_file(a.low_manifest)
    for q in ['docs/evaluation/pilot_01_quality.md','docs/evaluation/pilot_01_geometry.csv','docs/evaluation/pilot_01_results.json','docs/evaluation/geometry_alignment_audit.md','docs/evaluation/geometry_reliability_b1b.md']:
        protected[q]=sha256_file(q)
    for path,digest in protected.items():
        if sha256_file(path)!=digest:raise ValueError('Protected baseline changed: '+path)
    manifest['protected_sha256']=protected;manifest['media_sha256']=dict(old['media_sha256']);manifest['source_sha256']={str(q):sha256_file(q) for q in [Path(__file__),Path('src/scanning/pilot_comparison.py')]}
    manifest['representation_relationship']='Same recording/session and dataset groups, not independent samples. Existing FEN/semantic metadata copied without upgrading review.'
    a.output_dir.mkdir(exist_ok=False);write(a.output_dir/'low_manifest_snapshot.json',old)
    for i,s in enumerate(old['sessions'],1):
        source=a.source_dir/f'new_position_{i:03d}_hd.MOV'
        result=ingest(s,source,a.output_dir/s['capture_session_id']);new=copy.deepcopy(s);new.update(source_video=result['source_video'],index_path=str(a.output_dir/s['capture_session_id']/'session.json'),candidates=result['candidates'],decoded_frames=result['decoded_frames'],source_role='primary_high_quality',same_recording_low_resolution_source=s['source_video'],same_recording_low_index_path=s['index_path'])
        manifest['sessions'].append(new);manifest['media_sha256'][str(source)]=result['source_video']['sha256'];write(a.output_dir/'manifest.json',manifest)
        print(s['position_category'],result['decoded_frames'],'offsets_ms',[r['match_offset_ms'] for r in result['candidates']],'correlation',[round(r['correspondence_score'],4) for r in result['candidates']],flush=True)
    validation=validate(manifest);write(a.output_dir/'integrity_validation.json',validation)
    for path,digest in (manifest['protected_sha256']|manifest['media_sha256']|manifest['source_sha256']).items():
        if sha256_file(path)!=digest:raise ValueError('Preservation failure: '+path)
    write(a.output_dir/'preservation.json',dict(old_artifacts_original_media_and_production_unchanged=True,protected_file_count=len(protected)))

if __name__=='__main__':main()
