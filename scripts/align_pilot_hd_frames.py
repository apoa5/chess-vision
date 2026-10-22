"""Freeze sequence-level offsets, then extract nearest original PTS for all 60 pairs."""
import argparse,copy,hashlib,json,sys,time
from fractions import Fraction
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.ingest_native_pilot import contact_sheet
from scripts.validate_native_pilot import validate
from src.scanning.native_pilot import decoder,sha256_file
from src.scanning.pilot_comparison import display_image,thumbnail,correspondence_score,nearest_time,register_pair


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--correspondence-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():p.error('Fresh output directory required')
    initial=json.loads((a.correspondence_run/'manifest.json').read_text());old=json.loads((a.correspondence_run/'low_manifest_snapshot.json').read_text());manifest=copy.deepcopy(initial);manifest['sessions']=[]
    manifest['source_revision']='hd_v2_sequence_aligned';manifest['alignment_input_manifest']=str(a.correspondence_run/'manifest.json');manifest['alignment_input_sha256']=sha256_file(a.correspondence_run/'manifest.json');manifest['source_sha256'][str(Path(__file__))]=sha256_file(Path(__file__))
    for path,h in (initial['protected_sha256']|initial['media_sha256']|initial['source_sha256']).items():
        if sha256_file(path)!=h:raise ValueError('Input/protected source changed: '+path)
    cv2.setNumThreads(1);a.output_dir.mkdir(exist_ok=False);(a.output_dir/'low_manifest_snapshot.json').write_text(json.dumps(old,indent=2)+'\n');av=decoder()
    for original,low in zip(initial['sessions'],old['sessions']):
        start=time.perf_counter();idx=json.loads(Path(original['index_path']).read_text());frames=idx['original_frame_index'];times=[Fraction(*r['pts_time']) for r in frames];first=times[0]
        offsets=[r['match_offset_ms'] for r in original['candidates']];offset_ms=float(np.median(offsets));offset=Fraction(str(offset_ms))/1000
        dest=a.output_dir/original['capture_session_id'];dest.mkdir(exist_ok=False);selected=[]
        with av.open(original['source_video']['path']) as c:
            stream=c.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
            for lf in low['candidates']:
                target=first+Fraction(str(lf['relative_timestamp_ms']))/1000+offset;row=copy.deepcopy(frames[nearest_time(times,target)])
                c.seek(row['pts'],stream=stream,backward=True);image=None
                for f in c.decode(stream):
                    actual=Fraction(f.pts)*f.time_base
                    if actual>Fraction(*row['pts_time']):break
                    if actual==Fraction(*row['pts_time']):
                        if int(f.rotation)!=row['encoded_rotation_degrees']:raise ValueError('Display metadata changed')
                        image=display_image(f.to_ndarray(format='bgr24'),int(f.rotation));break
                if image is None:raise ValueError('Seek failed to recover exact indexed PTS')
                q=dest/(row['frame_id']+'.png');cv2.imwrite(str(q),image);low_image=cv2.imread(lf['path']);score=correspondence_score(thumbnail(image),thumbnail(low_image));gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                row.update(path=str(q),derivative_sha256=sha256_file(q),decoded_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),source_video_sha256=original['source_video']['sha256'],split='pilot',training_eligible=False,verified_corners_px=None,verified_semantic_orientation=None,photographic_placement_review='pending',camera_orientation=None,camera_elevation=None,review_status='pending',matched_low_frame_id=lf['frame_id'],target_low_relative_timestamp_ms=lf['relative_timestamp_ms'],sequence_offset_ms=offset_ms,match_offset_ms=row['relative_timestamp_ms']-lf['relative_timestamp_ms'],alignment_residual_ms=float((Fraction(*row['pts_time'])-target)*1000),correspondence_score=score,pair_registration=register_pair(image,low_image),blur_score=float(cv2.Laplacian(gray,cv2.CV_64F).var()),mean_luma=float(gray.mean()))
                if abs(row['alignment_residual_ms'])>18 or score<.85:raise ValueError('Timestamp-aligned correspondence needs manual review')
                selected.append(row)
        idx.update(candidates=selected,alignment=dict(method='Median of 12 geometry-independent initial best-correlation time offsets; nearest actual PTS thereafter, one offset per recording.',sequence_offset_ms=offset_ms,initial_offsets_ms=offsets,initial_offset_mad_ms=float(np.median(np.abs(np.array(offsets)-offset_ms))),max_abs_nearest_pts_residual_ms=max(abs(r['alignment_residual_ms']) for r in selected),limitation='Inferred temporal offset, not hardware synchronization or proof of exact same exposure.'),selection_settings=dict(method='Frozen 12 low-resolution timestamps + sequence-level offset, nearest real source PTS; no new quality/geometry selection.'),alignment_extraction_seconds=time.perf_counter()-start)
        (dest/'session.json').write_text(json.dumps(idx,indent=2)+'\n');s=copy.deepcopy(original);s.update(index_path=str(dest/'session.json'),candidates=selected,temporal_alignment=idx['alignment']);manifest['sessions'].append(s)
        contact_sheet([(r['path'],f"{i+1:02d} {r['relative_timestamp_ms']/1000:.3f}s") for i,r in enumerate(selected)],dest/'candidates_contact.png')
        (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(s['position_category'],'sequence offset',offset_ms,'max residual',idx['alignment']['max_abs_nearest_pts_residual_ms'],'seconds',idx['alignment_extraction_seconds'],flush=True)
    result=validate(manifest);(a.output_dir/'integrity_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    for path,h in (manifest['protected_sha256']|manifest['media_sha256']|manifest['source_sha256']).items():
        if sha256_file(path)!=h:raise ValueError('Preservation failure: '+path)
    (a.output_dir/'preservation.json').write_text(json.dumps(dict(old_artifacts_original_media_and_production_unchanged=True,sequence_aligned_extraction_validated=True),indent=2)+'\n')

if __name__=='__main__':main()
