#!/usr/bin/env python3
"""Image-only engineering check of all camera sides and diagonal board views."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.scanning.bootstrap import bootstrap_image,draw_bootstrap
from src.utils.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/scanning/tracking/orientations'))
    args=parser.parse_args()
    image=cv2.imread(str(args.image))
    if image is None:parser.error('Image cannot be decoded')
    cv2.setNumThreads(1)
    config=load_config()|{'scan_fast_acquisition':True};base=bootstrap_image(image,config)
    if base['status']!='candidate':raise SystemExit('Source needs an unambiguous automatic alignment')
    corners=np.array(base['geometry']['corners']);height,width=image.shape[:2]
    size=int(np.ceil(np.hypot(width,height)))+20;records=[]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for angle in (0,40,45,50,90,135,180,225,270,315):
        matrix=cv2.getRotationMatrix2D((width/2,height/2),angle,1)
        matrix[:,2]+=[(size-width)/2,(size-height)/2]
        rotated=cv2.warpAffine(image,matrix,(size,size),borderValue=(40,40,40))
        truth=np.column_stack([corners,np.ones(4)])@matrix.T
        start=time.monotonic();result=bootstrap_image(rotated,config)
        error=float(np.max(np.linalg.norm(np.array(result['geometry']['corners'])-truth,axis=1))) if result['geometry'] else None
        record={'angle_ccw':angle,'status':result['status'],'reason':result['reason'],'turns':result.get('rotation_quarters_ccw'),'max_corner_error_px':error,'seconds':round(time.monotonic()-start,2)}
        records.append(record);print(record,flush=True)
        if result['geometry']:cv2.imwrite(str(args.output_dir/f'rotation_{angle:03d}.jpg'),draw_bootstrap(rotated,result))
    report={'kind':'synthetic_rotation_of_real_photo','source_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),'baseline':'automatic_geometry_of_source_photo_not_independent_manual_truth','physical_camera_motion_tested':False,'records':records}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if any(r['status']!='candidate' or r['max_corner_error_px']>8 for r in records):raise SystemExit('Orientation check failed; inspect report and overlays')

if __name__=='__main__':main()
