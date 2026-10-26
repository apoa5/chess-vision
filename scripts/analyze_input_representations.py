"""C.1 visible-feature proxies and synthetic sensitivity, no recognition inference."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write
from scripts.audit_input_representations import source_case,select_cases
from src.vision.input_representations import extract_representations,visible_bbox_coverage
from src.scanning.native_pilot import sha256_file


def contains(support,point):
    return cv2.pointPolygonTest(cv2.convexHull(np.float32(support)),tuple(map(float,point)),False)>=0


def run(audit_path,annotations_path,output):
    if output.exists():raise ValueError('Fresh analysis output required')
    audit=read(audit_path);annotations=read(annotations_path)
    if annotations['audit_sha256']!=sha256_file(audit_path):raise ValueError('Visible-extent annotations bind a different audit')
    records=[];sensitivity=[];costs=[];case_objects={c['case_id']:c for c in select_cases()}
    for case in audit['cases']:
        cid=case['case_id'];o=np.array(case['native_annotation_window_xyxy'][:2]);annotation=annotations['cases'].get(cid)
        if annotation:
            bbox=np.asarray(annotation['visible_bbox_local_xyxy'],float)+np.tile(o,2);crown=np.asarray(annotation['crown_local_xy'],float)+o
            values={}
            for name,m in case['metrics'].items():
                values[name]=dict(visible_bbox_area_coverage_proxy=visible_bbox_coverage(bbox,m['source_support']),crown_inside=contains(m['source_support'],crown))
            records.append(dict(case_id=cid,verification=case['verification'],assistant_estimated_bbox_raw=bbox.tolist(),assistant_estimated_crown_raw=crown.tolist(),
                metrics=values,source_note=annotation['note'],boundary_ambiguous_base=('base_local_xy' in annotation),
                base_inside_target=contains(case['source_target_polygon'],np.asarray(annotation['base_local_xy'])+o) if 'base_local_xy' in annotation else None,
                segmentation_ground_truth=False))
        image,g,_=source_case(case_objects[cid]);sqr=case['canonical_square']
        t=time.perf_counter();baseline=extract_representations(image,g,sqr,audit['frozen_config']);costs.append(time.perf_counter()-t)
        shifts=[]
        for dx,dy in [(-2,0),(2,0),(0,-2),(0,2)]:
            perturbed=copy.deepcopy(g);perturbed['source_square_polygons']=(np.asarray(g['source_square_polygons'])+[dx,dy]).tolist()
            perturbed['provenance']['source_corners']=(np.asarray(g['provenance']['source_corners'])+[dx,dy]).tolist()
            alt=extract_representations(image,perturbed,sqr,audit['frozen_config']);values={}
            for name in ('b2','taller','expanded'):
                a,b=baseline[name]['packed'],alt[name]['packed'];ma=a[3]>0;mb=b[3]>0
                values[name]=dict(rgb_mean_abs_change=float(np.abs(a[:3].astype(float)-b[:3]).mean()),mask_iou=float((ma&mb).sum()/max(1,(ma|mb).sum())))
            shifts.append(dict(dx=dx,dy=dy,values=values,height_mode_changed=baseline['taller']['height_proxy']['mode']!=alt['taller']['height_proxy']['mode']))
        sensitivity.append(dict(case_id=cid,synthetic_only=True,frozen_inputs_modified=False,shifts=shifts))
    result=dict(schema_version='input_representation_analysis_c1_v1',audit_sha256=sha256_file(audit_path),annotations_sha256=sha256_file(annotations_path),
        visibility_measurement='assistant-estimated visible BBOX area and crown inclusion; not segmented silhouette/occlusion completeness',visible_feature_cases=records,
        synthetic_geometry_sensitivity=sensitivity,all_variant_extraction_seconds=costs,source_sha256=sha256_file(Path(__file__)),accuracy_claims=False)
    output.mkdir();write(output/'analysis.json',result)
    rows=['# C.1 visible-feature proxy measurements','', '| Case | B2 bbox / crown | Taller bbox / crown | Expanded bbox / crown |','|---|---:|---:|---:|']
    for c in records:
        cells=[f"{c['metrics'][n]['visible_bbox_area_coverage_proxy']*100:.1f}% / {c['metrics'][n]['crown_inside']}" for n in ('b2','taller','expanded')]
        rows.append('| '+c['case_id']+' | '+' | '.join(cells)+' |')
    (output/'visible_features.md').write_text('\n'.join(rows)+'\n')
    print('\n'.join(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.audit,a.annotations,a.output_dir)
