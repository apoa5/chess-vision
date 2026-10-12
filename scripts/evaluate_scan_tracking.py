#!/usr/bin/env python3
"""Controlled tracker test from a reviewed phone image; not a live-video benchmark."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.scanning.replay import replay_session

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--frame',type=int,default=30,help='1-based index of reviewed aligned frame')
    parser.add_argument('--output',type=Path,default=Path('outputs/scanning/tracking/controlled_report.json'))
    parser.add_argument('--node',type=Path,default=ROOT/'tools/node/bin/node')
    args=parser.parse_args()
    replay_session(args.archive)
    with ZipFile(args.archive) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        if not 1<=args.frame<=len(manifest['frames']):parser.error('Frame index outside scan')
        frame=manifest['frames'][args.frame-1]
        if not frame['geometry']:parser.error('Select a reviewed aligned frame')
        image=cv2.imdecode(np.frombuffer(archive.read(frame['path']),np.uint8),0)
    height,width=image.shape
    scale=640/max(height,width)
    image=cv2.resize(image,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
    corners=np.array(frame['geometry']['corners'])*[image.shape[1]/width,image.shape[0]/height]
    frames=[]
    for step in range(5):
        transform=np.array([[1,.0005*step,2*step],[-.0005*step,1,step],[.000002*step,-.000001*step,1.]])
        warped=cv2.warpPerspective(image,transform,(image.shape[1],image.shape[0]))
        truth=cv2.perspectiveTransform(corners[None].astype(np.float64),transform)[0]
        frames.append({'width':image.shape[1],'height':image.shape[0],'pixels':warped.reshape(-1).tolist(),'corners':truth.tolist()})
    fixture={'corners':corners.tolist(),'source_sha256':frame['source_sha256'],'frames':frames}
    result=subprocess.run([str(args.node),'--experimental-strip-types',str(ROOT/'web/scripts/evaluate-tracking.mjs')],input=json.dumps(fixture),text=True,capture_output=True,check=True)
    report=json.loads(result.stdout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if any(record['state']!='tracking' or record['max_corner_error_processing_px']>2 for record in report['records']):raise SystemExit('Controlled tracker check failed; inspect report')
    print(f"Four controlled steps passed. Report: {args.output}")

if __name__=='__main__':main()
