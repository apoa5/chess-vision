"""C.1 fixed-geometry, no-inference representation audit on verified D2 and flagged D3c cases."""
import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write,check_hashes
from scripts.characterize_square_evidence import evaluation_truths
from src.scanning.native_pilot import sha256_file
from src.scanning.interim_video import decode_exact
from src.scanning.square_evidence import SQUARES
from src.vision.input_representations import extract_representations,RepresentationSettings,jacobian_anisotropy


def select_cases():
    e=read('data/metadata/evidence_d2_manifest_v1.json');check_hashes({e['observations_path']:e['observations_sha256']})
    obs=[json.loads(x) for x in Path(e['observations_path']).read_text().splitlines()]
    catalog=Path('data/metadata/a3_pilot_01_sources_v2.json');truth,_=evaluation_truths(read(catalog),sha256_file(catalog),read('data/metadata/evidence_d2_ground_truth_corrections_v1.json'))
    requests=[('king_rook','puzzle','b3',6,4),('king_correct','puzzle','b3',6,6),('knight_rook','opening','f3',2,4),('knight_correct','opening','f3',2,2),
        ('black_rook_king','middlegame','f8',10,12),('black_rook_correct','middlegame','f8',10,10),('empty_rook','middlegame','b1',0,4),('empty_correct','middlegame','b1',0,0),
        ('pawn_missed','endgame','h4',1,0),('pawn_correct','endgame','h4',1,1)]
    cases=[]
    for name,position,square,t,p in requests:
        candidates=[o for o in obs if o['position_id'].endswith(position) and o['canonical_square']==square and o['predicted_class_id']==p and truth[o['session_id']][SQUARES.index(square)]==t]
        if not candidates:raise ValueError('Required existing diagnostic unavailable: '+name)
        o=max(candidates,key=lambda o:(o['maximum_probability'],o['observation_id']))
        cases.append(dict(case_id='D2_'+name,observation=o,true_class_id=t,verification='D2_human_geometry_and_owner_corrected_FEN'))
    root=Path('outputs/scanning/d3c_interim_v1');d3=json.loads(gzip.decompress((root/'recognition/observations.json.gz').read_bytes()))['observations']
    for name,sid,sq,t,p in [('king_rook','d3c_pos_17','f3',6,4),('king_correct','d3c_pos_17','f3',6,6),('empty_rook','d3c_pos_05','h1',0,4),('empty_black_rook','d3c_pos_05','h1',0,10)]:
        o=max([o for o in d3 if o['session_id']==sid and o['canonical_square']==sq and o['predicted_class_id']==p],key=lambda o:o['maximum_probability'])
        cases.append(dict(case_id='D3c_'+name,observation=o,true_class_id=t,verification='D3c_provisional_geometry_and_supplied_FEN_not_photo_verified'))
    return cases


def source_case(case):
    o=case['observation']
    if case['case_id'].startswith('D2_'):
        check_hashes({o['frame_path']:o['frame_sha256'],o['geometry_reference']:o['geometry_reference_sha256']})
        return cv2.imread(o['frame_path']),read(o['geometry_reference'])['policies'][o['geometry_policy']],dict(path=o['frame_path'],sha256=o['frame_sha256'])
    root=Path('outputs/scanning/d3c_interim_v1');s=next(s for s in read(root/'registration.json')['sessions'] if s['session_id']==o['session_id'])
    row=next(read(p) for p in (root/o['session_id']).glob('candidate_*.json') if read(p)['frame_id']==o['frame_id'])
    return decode_exact(s['source']['path'],row),row['geometry'],dict(path=s['source']['path'],sha256=s['source']['sha256'],pts=row['pts'],time_base=row['time_base'])


def draw_packed(packed,label):
    im=cv2.cvtColor(packed[:3].transpose(1,2,0),cv2.COLOR_RGB2BGR).copy();contours,_=cv2.findContours(packed[3].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(0,255,0),1)
    panel=np.full((300,260,3),250,np.uint8);panel[35:259,18:242]=im;cv2.putText(panel,label,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(10,10,10),1)
    cv2.putText(panel,'mask = ground square',(4,285),cv2.FONT_HERSHEY_SIMPLEX,.36,(10,10,10),1);return panel


def run(output):
    if output.exists():raise ValueError('Fresh C1 output required')
    config=read('configs/experiments/evidence_d2.json');cases=select_cases()
    protected=read('outputs/scanning/d3c_interim_v1/registration.json')['protected_sha256']
    for path in ['data/metadata/fusion_independent_d3c_index_v1.json','data/metadata/fusion_reliability_d3b_index_v1.json']:
        protected.update(read(path)['artifacts']);protected[path]=sha256_file(path)
    check_hashes(protected);output.mkdir(parents=True);write(output/'preserved_inputs.json',dict(protected_sha256=protected))
    rows=[];sections=[]
    for case in cases:
        image,g,source=source_case(case);o=case['observation'];start=time.perf_counter();r=extract_representations(image,g,o['canonical_square'],config);elapsed=time.perf_counter()-start
        packed=r['b2']['packed'];digest=hashlib.sha256(packed.tobytes()).hexdigest()
        if digest!=o['input_packed_sha256']:raise ValueError('Frozen B.2 pixel/mask mismatch: '+case['case_id'])
        target=r['source_target_polygon'];edge=float(np.linalg.norm(target-np.roll(target,-1,axis=0),axis=1).mean());center=target.mean(axis=0)
        # Native ROI for auditable manual visible-feature annotations; pixels stay original.
        lo=np.maximum(0,np.floor(target.min(axis=0)-[2*edge,3*edge])).astype(int);hi=np.minimum([image.shape[1],image.shape[0]],np.ceil(target.max(axis=0)+[2*edge,2*edge])).astype(int)
        raw=image[lo[1]:hi[1],lo[0]:hi[0]].copy();cv2.polylines(raw,[np.rint(target-lo).astype(np.int32)],True,(0,255,0),1)
        raw_path=output/(case['case_id']+'_raw.jpg');cv2.imwrite(str(raw_path),raw,[cv2.IMWRITE_JPEG_QUALITY,95])
        panels=[draw_packed(r['b2']['packed'],'Frozen B2 3x3'),draw_packed(r['taller']['packed'],'Taller native 2-square'),draw_packed(r['expanded']['packed'],'Expanded 5x5'),draw_packed(r['two_branch']['local']['packed'],'Two-branch: local'),draw_packed(r['two_branch']['context']['packed'],'Two-branch: context')]
        # Explicit binary masks are displayed separately; no model receives outlines.
        masks=np.full((300,260,3),255,np.uint8);cv2.putText(masks,'B2 binary target mask',(3,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1);masks[35:259,18:242]=r['b2']['packed'][3,:,:,None]*255;panels.append(masks)
        tile=output/(case['case_id']+'_variants.jpg');cv2.imwrite(str(tile),cv2.hconcat(panels),[cv2.IMWRITE_JPEG_QUALITY,93])
        overview=image.copy();cv2.polylines(overview,[np.rint(r['source_board_quad']).astype(np.int32)],True,(0,255,0),3)
        cv2.polylines(overview,[np.rint(target).astype(np.int32)],True,(0,0,255),3)
        # One small source overview per case; no full-size raw or tensor cache.
        scale=min(1.,700/max(image.shape[:2]));overview=cv2.resize(overview,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output/(case['case_id']+'_source.jpg')),overview,[cv2.IMWRITE_JPEG_QUALITY,88])
        metrics={}
        for name in ('b2','taller','expanded'):
            v=r[name];mask=v['packed'][3];coords=np.argwhere(mask)
            metrics[name]=dict(packed_sha256=hashlib.sha256(v['packed'].tobytes()).hexdigest(),input_shape=v['input_shape'],
                mask_area=int(mask.sum()),target_equivalent_width=float(np.sqrt(mask.sum())),target_mask_bbox_wh=[int(coords[:,1].max()-coords[:,1].min()+1),int(coords[:,0].max()-coords[:,0].min()+1)],
                raw_to_input=v['raw_to_input'].tolist(),source_support=v['source_support'].tolist(),base_jacobian_anisotropy=jacobian_anisotropy(v['raw_to_input'],center),
                raw_height_proxy=v.get('height_proxy'),valid_fraction=v.get('valid_fraction'),aspect_preserving=v['aspect_preserving'])
        rows.append(dict(case_id=case['case_id'],verification=case['verification'],session_id=o['session_id'],position_id=o['position_id'],canonical_square=o['canonical_square'],
            observation_id=o['observation_id'],frame_id=o['frame_id'],timestamp_ms=o['timestamp_ms'],true_class_id=case['true_class_id'],predicted_class_id=o['predicted_class_id'],
            frozen_probabilities=o['probabilities'],geometry_reference=o.get('geometry_reference'),source=source,source_target_polygon=target.tolist(),source_square_mean_edge_px=edge,
            native_annotation_window_xyxy=[int(lo[0]),int(lo[1]),int(hi[0]),int(hi[1])],source_size_wh=[image.shape[1],image.shape[0]],
            metrics=metrics,extraction_seconds=elapsed,two_branch=dict(local='taller',context='expanded',arrays_shared=True,model_implemented=False),
            piece_visibility='no_verified_piece_segmentation; explicit visible-extent supplement needed for coverage claims'))
        title=f"{case['case_id']} {o['canonical_square']} truth={o['class_names'][case['true_class_id']]} old prediction={o['predicted_class']} {o['maximum_probability']:.3f}"
        sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(case["verification"])}. Original timestamp {o["timestamp_ms"]/1000:.3f}s. No new inference/training.</p><img height="350" src="{case["case_id"]}_source.jpg"><img style="max-height:450px;max-width:65%" src="{raw_path.name}"><p>Source board/target and native target neighborhood. Raw annotation window: {lo.tolist()}..{hi.tolist()}.</p><img style="max-width:100%" src="{tile.name}"><p>Taller assumption: {html.escape(r["taller"]["height_proxy"]["mode"])}. No calibrated camera pose or piece silhouette is claimed. RGB previews contain display-only outlines; arrays/masks are separately hashed.</p></section>')
        print(case['case_id'],'B2 exact',r['taller']['height_proxy']['mode'],flush=True)
    write(output/'audit.json',dict(schema_version='input_representation_c1_v1',cases=rows,settings=RepresentationSettings().__dict__,frozen_config=config,
        no_training=True,new_accuracy_measurements=False,protected_files_unchanged=len(protected),source_sha256={p:sha256_file(p) for p in ['src/vision/input_representations.py','scripts/audit_input_representations.py']},
        limitation='Error-selected diagnostic cases, not performance samples. D3c fine geometry/labels remain provisional.'))
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>C1 input representation audit</title><style>body{font:15px system-ui;max-width:1600px;margin:auto;padding:20px}section{border-top:1px solid #aaa;padding:15px 0}img{object-fit:contain}</style><h1>C.1 input representation audit</h1><p>Ten geometry/FEN-reviewed D2 examples; four explicitly provisional D3c examples. These illustrative cases are selected by historical errors, not independent accuracy estimates. Two-branch panels reuse the taller/local and expanded/context arrays. No models were trained or evaluated on changed representations.</p>'+''.join(sections))
    check_hashes(protected)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.output_dir)
