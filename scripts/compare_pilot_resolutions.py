"""Paired high/Telegram pilot diagnostics; completion and agreement are not truth."""
import argparse,csv,json,sys
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.scanning.native_pilot import sha256_file


def stage(result):
    raw=result['failure_stage']
    return 'rectification_validation' if raw=='dependency_or_runtime' and 'Corners must be ordered' in result.get('error','') else raw


def project(points,h):
    points=np.asarray(points,float);shape=points.shape
    return cv2.perspectiveTransform(points.reshape(-1,1,2),np.asarray(h,float)).reshape(shape)


def agreement(low,high,registration):
    if not low['completed'] or not high['completed'] or registration['status']!='registered':return None
    h=registration['high_display_to_low_pixels'];a=np.asarray(low['returned_polygon']);b=project(high['returned_polygon'],h)
    rms=min(float(np.sqrt(np.mean(np.sum((a-np.roll(b,k,axis=0))**2,axis=1)))) for k in range(4))
    # Compare same semantic square identities, even if camera-relative row order differs.
    la=dict(zip(low['semantic_square_ids'],np.asarray(low['source_square_polygons']).mean(axis=1)))
    ha=dict(zip(high['semantic_square_ids'],project(high['source_square_polygons'],h).mean(axis=1)))
    centre=np.array([np.linalg.norm(la[k]-ha[k]) for k in sorted(la)])
    return dict(corner_rms_low_pixels=rms,semantic_square_centre_rms_low_pixels=float(np.sqrt(np.mean(centre**2))),max_semantic_square_centre_shift_low_pixels=float(centre.max()),rectification_quarter_turn_agrees=low['rotation_quarters_ccw']==high['rotation_quarters_ccw'],meaning='Representation agreement, not ground-truth geometry error. Registration residual and timestamp mismatch also contribute.')


def diagnostic_image(record):
    image=cv2.imread(record['overlay_path'])
    scale=image.shape[0]/848
    polygon=record.get('returned_polygon') if record['completed'] else record.get('localization_proposal')
    if polygon is None and not record['completed']:
        polygon=record.get('leading_rejected_polygon')
        if polygon is not None:
            cv2.polylines(image,[np.rint(polygon).astype(np.int32)],True,(0,140,255),3)
            cv2.putText(image,'REJECTED leading candidate',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,100,255),2)
    if polygon is not None:
        for i,point in enumerate(polygon):
            xy=tuple(np.rint(point).astype(int));cv2.circle(image,xy,round(6*scale),(255,70,0),max(1,round(2*scale)))
            cv2.putText(image,f'C{i}',xy,cv2.FONT_HERSHEY_SIMPLEX,.65*scale,(255,70,0),max(1,round(2*scale)))
    if record['completed']:
        for square,poly in zip(record['semantic_square_ids'],record['source_square_polygons']):
            if square in ('a1','h1','a8','h8'):
                xy=tuple(np.rint(np.asarray(poly).mean(axis=0)).astype(int))
                cv2.putText(image,square,xy,cv2.FONT_HERSHEY_SIMPLEX,.7*scale,(0,0,0),max(2,round(4*scale)))
                cv2.putText(image,square,xy,cv2.FONT_HERSHEY_SIMPLEX,.7*scale,(0,255,0),max(1,round(2*scale)))
    return image


def panel(image,caption,width=464):
    resized=cv2.resize(image,(width,848),interpolation=cv2.INTER_AREA)
    canvas=np.full((900,width,3),245,np.uint8);canvas[52:]=resized
    for i,line in enumerate(caption):cv2.putText(canvas,line,(6,20+i*20),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,0),1)
    return canvas


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--hd-run',type=Path,required=True);p.add_argument('--hd-geometry',type=Path,required=True);p.add_argument('--low-geometry',type=Path,default=Path('outputs/scanning/pilot_01/geometry_run1'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    if a.output_dir.exists():p.error('Fresh output directory required')
    hd=json.loads((a.hd_run/'manifest.json').read_text());lo=json.loads((a.hd_run/'low_manifest_snapshot.json').read_text());lr=json.loads((a.low_geometry/'results.json').read_text());hr=json.loads((a.hd_geometry/'results.json').read_text())
    lmap={(f['session_id'],f['candidate_number']):f for f in lr['frames']};hmap={(f['session_id'],f['candidate_number']):f for f in hr['frames']}
    if set(lmap)!=set(hmap):raise ValueError('Unpaired evaluation population')
    oldside={r['frame_id']:r['observed_side_unverified'] for r in csv.DictReader(Path('docs/evaluation/pilot_01_geometry.csv').open())}
    rows=[];frames=[];a.output_dir.mkdir(exist_ok=False)
    for hs,ls in zip(hd['sessions'],lo['sessions']):
        if any(hs[k]!=ls[k] for k in ['capture_session_id','position_id','fen','placement_fen','position_group_id','independence_group_id','split']):raise ValueError('Session identity/label/group mutation')
        for n,(hf,lf) in enumerate(zip(hs['candidates'],ls['candidates']),1):
            low=lmap[(hs['capture_session_id'],n)];high=hmap[(hs['capture_session_id'],n)]
            if hf['matched_low_frame_id']!=lf['frame_id']:raise ValueError('Broken timestamp pairing')
            frame=dict(position=hs['position_category'],session_id=hs['capture_session_id'],candidate=n,observed_side_unverified=oldside[lf['frame_id']],low_timestamp_seconds=lf['relative_timestamp_ms']/1000,high_timestamp_seconds=hf['relative_timestamp_ms']/1000,offset_ms=hf['match_offset_ms'],correlation=hf['correspondence_score'],registration=hf['pair_registration'],policies={})
            tiles=[]
            for policy in ['legacy','marked-board']:
                l=low['policies'][policy];h=high['policies'][policy]
                transition=('retained' if h['completed'] else 'regressed') if l['completed'] else ('rescued' if h['completed'] else 'still_unresolved')
                pair=dict(low_completed=l['completed'],high_completed=h['completed'],transition=transition,low_stage=stage(l),high_stage=stage(h),high_error=h.get('error'),high_recovery_reason=h.get('recovery_reason'),low_seconds=l['elapsed_seconds'],high_seconds=h['elapsed_seconds'],geometry_agreement=agreement(l,h,hf['pair_registration']))
                frame['policies'][policy]=pair
                rows.append({k:frame[k] for k in ['position','session_id','candidate','observed_side_unverified','low_timestamp_seconds','high_timestamp_seconds','offset_ms']}|dict(policy=policy,**{k:v for k,v in pair.items() if k!='geometry_agreement'},**(pair['geometry_agreement'] or {})))
                for label,record in [('Telegram',l),('HD',h)]:tiles.append(panel(diagnostic_image(record),[f"{hs['position_category']} #{n:02d} {label} {policy}",'complete (unverified)' if record['completed'] else str(stage(record))]))
            imagepath=a.output_dir/f"{hs['position_category']}_{n:02d}_paired.jpg";cv2.imwrite(str(imagepath),np.hstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,88]);frame['paired_overlay']=str(imagepath);frames.append(frame)
    # Uniform CSV columns even where one policy does not complete.
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (a.output_dir/'paired.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=columns);w.writeheader();w.writerows(rows)
    summary={}
    for policy in ['legacy','marked-board']:
        relevant=[f['policies'][policy] for f in frames];summary[policy]=dict(low_completed=sum(x['low_completed'] for x in relevant),high_completed=sum(x['high_completed'] for x in relevant),transitions=dict(Counter(x['transition'] for x in relevant)),high_failure_stages=dict(Counter(x['high_stage'] for x in relevant if not x['high_completed'])),low_runtime_seconds=sum(x['low_seconds'] for x in relevant),high_runtime_seconds=sum(x['high_seconds'] for x in relevant),low_runtime_median_seconds=float(np.median([x['low_seconds'] for x in relevant])),high_runtime_median_seconds=float(np.median([x['high_seconds'] for x in relevant])))
    report=dict(schema_version='pilot_resolution_comparison_v1',comparison_script_sha256=sha256_file(Path(__file__)),designation='development_only',same_recordings_not_independent=True,ground_truth_geometry_available=False,input_sha256={str(q):sha256_file(q) for q in [a.hd_run/'manifest.json',a.hd_geometry/'results.json',a.low_geometry/'results.json']},summary=summary,frames=frames,limitations=['Timestamp match chosen by visual correlation only; no geometry-conditioned selection.','Completion and cross-representation agreement do not establish geometric correctness.','Old and new wall times are descriptive runs, not controlled benchmark or mobile latency.','HEVC/H.264, color/exposure processing, timebase and resampling differ; not a pure resolution-only intervention.'])
    (a.output_dir/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
