"""Select bounded original-PTS frames; prepare review mosaics, never auto-approve."""
import argparse,hashlib,html,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.training.manifest_tools import read,write_new
from src.training.multidataset import PROFILE,regular_observations,digest,split_audit
from src.training.video_review import binding,CHECKS,apply_reviews
from src.scanning.video_keyframes import select_keyframes,selection_metrics
from src.scanning.interim_video import decode_exact,pixel_hash
from scripts.run_square_evidence import make_inputs
from src.chess.labels import CLASS_NAMES

def prepare(output,target):
    if output.exists():raise ValueError('Fresh review output required')
    if not 10<=target<=20:raise ValueError('Review budget 10–20 per eligible session')
    output.mkdir(parents=True);(output/'images').mkdir()
    policy=read('data/metadata/d3c_data_use_c4_v1.json');config=read('configs/experiments/keyframes_d1.json');crop=read('configs/experiments/evidence_d2.json')
    frames=[];summaries=[];started=time.perf_counter()
    for s in policy['sessions']:
        if digest(s['source']['path'])!=s['source']['sha256']:raise ValueError('Video changed')
        paths=sorted((Path('outputs/scanning/d3c_interim_v1')/s['session_id']).glob('candidate_*.json'))
        rows=[read(p) for p in paths];paths_by_id={r['frame_id']:p for r,p in zip(rows,paths)}
        selection=select_keyframes(rows,'quality_diverse',target,config)
        summaries.append(dict(session_id=s['session_id'],protected_overlap=s['protected_overlap'],selection=selection,metrics=selection_metrics(rows,selection,config),candidates=len(rows)))
        for row in rows:
            if row['frame_id'] not in selection['selected']:continue
            if row['source_video_sha256']!=s['source']['sha256']:raise ValueError('Cached source mismatch')
            image=decode_exact(s['source']['path'],row)
            if pixel_hash(image)!=row['decoded_pixel_sha256']:raise ValueError('Decoded pixels changed')
            g=row['geometry'];inputs=make_inputs(image,g,crop);lookup={r['canonical_square']:r for r in inputs}
            preview=image.copy()
            for square,poly in zip(g['semantic_square_ids'],g['source_square_polygons']):
                pts=np.int32(poly);cv2.polylines(preview,[pts],True,(0,230,40),2)
                p=np.mean(pts,axis=0).astype(int);cv2.putText(preview,square,tuple(p),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            preview=cv2.resize(preview,(round(preview.shape[1]*720/preview.shape[0]),720))
            mosaic=np.zeros((8*112,8*112,3),np.uint8)
            observations=regular_observations(s['fen'],0)
            camera={sq:i for i,sq in enumerate(g['semantic_square_ids'])}
            for o in observations:
                square=o['square'];packed=lookup[square]['packed'];r,c=divmod('abcdefgh'.index(square[0])+(8-int(square[1]))*8,8)
                tile=cv2.resize(cv2.cvtColor(packed[:3].transpose(1,2,0),cv2.COLOR_RGB2BGR),(112,112));mask=cv2.resize(packed[3],(112,112),interpolation=cv2.INTER_NEAREST)
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(tile,contours,-1,(255,0,255),1)
                cv2.putText(tile,f'{square}:{o["class_id"]}',(3,14),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,255),1)
                mosaic[r*112:(r+1)*112,c*112:(c+1)*112]=tile
                o.update(camera_index=camera[square],visibility='pending_human_crop_review',packed_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
            ident=row['frame_id'];cv2.imwrite(str(output/'images'/f'{ident}_grid.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,83]);cv2.imwrite(str(output/'images'/f'{ident}_crops.jpg'),mosaic,[cv2.IMWRITE_JPEG_QUALITY,83])
            board=dict(id='d3c:'+ident,dataset='d3c_retired',source_id=ident,position_id=s['position_id'],session_id=s['session_id'],game_id=None,
                source=dict(path=s['source']['path'],filename=Path(s['source']['path']).name,sha256=s['source']['sha256']),placement=s['fen'].split()[0],original_fen=s['fen'],
                semantic_orientation=g['predicted_camera_orientation'],split='quarantine' if s['protected_overlap'] else 'train',representation=PROFILE,
                geometry=dict(status='completed_unverified',kind='video_original_pts',candidate_path=str(paths_by_id[ident]),candidate_sha256=digest(paths_by_id[ident])),
                annotation=dict(status='unverified',provenance='owner-supplied filename/FEN association; missing reference photo; crop review pending'),
                inclusion=dict(status='quarantine',reason='protected_position_overlap' if s['protected_overlap'] else 'awaiting_human_FEN_geometry_crop_review'),observations=observations)
            frame=dict(frame_id=ident,source_sha256=s['source']['sha256'],pts_time=row['pts_time'],timestamp_ms=row['timestamp_ms'],decoded_pixel_sha256=row['decoded_pixel_sha256'],
                geometry_sha256=hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest(),fen=s['fen'],protected_overlap=s['protected_overlap'],board=board)
            frame['binding']=binding(frame);frames.append(frame)
        print(s['session_id'],len(selection['selected']),'pending review',flush=True)
    packet=dict(schema_version='c4_video_review_v1',authority='owner C4 authorization; no automatic ground truth verification',target=target,
                crop_config_sha256=digest('configs/experiments/evidence_d2.json'),selection_config_sha256=digest('configs/experiments/keyframes_d1.json'),sessions=summaries,frames=frames,elapsed_seconds=time.perf_counter()-started)
    write_new(output/'packet.json',packet)
    template=dict(packet_sha256=hashlib.sha256(json.dumps(packet,sort_keys=True).encode()).hexdigest(),reviewer='',frames={f['frame_id']:dict(binding=f['binding'],status='pending',checks={c:False for c in CHECKS},ambiguous_squares=[]) for f in frames})
    write_new(output/'review_template.json',template)
    cards=[]
    for f in frames:
        i=f['frame_id'];cards.append(f'<section><h2>{i} — {f["timestamp_ms"]/1000:.2f}s</h2><code>{html.escape(f["fen"])}</code><p>UNVERIFIED. Protected overlap: {f["protected_overlap"]}</p><img height="600" src="images/{i}_grid.jpg"><img width="700" src="images/{i}_crops.jpg"></section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>C4 video training review</title><h1>Pending human review</h1><p>Complete review_template.json with reviewer, status verified/rejected, all seven checks, and ambiguous square flags. Verify FEN placement, unchanged position, orientation, corners, canonical mapping, crop alignment and all 64 crops. Labels shown are owner-provided, never predictions. Magenta outlines identify target masks. Do not approve indistinguishable or misaligned crops. Position 03 stays quarantined.</p>'+''.join(cards))
    return packet

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--target',type=int,default=15);p.add_argument('--packet',type=Path);p.add_argument('--reviews',type=Path);a=p.parse_args()
    if a.reviews:
        boards=apply_reviews(read(a.packet),read(a.reviews));split_audit(boards)
        write_new(a.output,dict(schema_version='multidataset_c4_v1',representation=PROFILE,class_names=list(CLASS_NAMES),final_holdout=False,review_path=str(a.reviews),boards=boards))
    else:prepare(a.output,a.target)
