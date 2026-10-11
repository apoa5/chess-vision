#!/usr/bin/env python3
"""Benchmark optional pretrained OCR on one validated scan frame."""
import argparse,json,time,sys,hashlib
from pathlib import Path
from zipfile import ZipFile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.scanning.replay import replay_session
from src.scanning.learned_markings import read_markings


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('archive',type=Path);parser.add_argument('--frame',type=int,default=1);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    replay_session(args.archive);cv2.setNumThreads(1)
    with ZipFile(args.archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        if not 1<=args.frame<=len(manifest['frames']):parser.error('frame must be a valid 1-based index')
        frame=manifest['frames'][args.frame-1];raw=z.read(frame['path']);image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
        start=time.perf_counter();observations=read_markings(image)
    report={'backend':'easyocr_1.7.2_english_g2_craft','source_sha256':hashlib.sha256(raw).hexdigest(),'frame_id':frame['frame_id'],'elapsed_ms':(time.perf_counter()-start)*1000,'observations':observations,'fen_used':False}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(f'{len(observations)} accepted single-character readings; {args.output}')

if __name__=='__main__':main()
