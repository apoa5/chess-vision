#!/usr/bin/env python3
"""Evaluate image-only bootstrap and a separately reported manual-reference path."""
import argparse,json,sys,hashlib,time
from pathlib import Path
from zipfile import ZipFile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.utils.config import load_config
from src.scanning.replay import replay_session
from src.scanning.bootstrap import bootstrap_image,draw_bootstrap


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--stride',type=int,default=1);args=p.parse_args()
    if args.stride<1:p.error('stride must be positive')
    replay_session(args.archive);config=load_config();cv2.setNumThreads(1);args.output_dir.mkdir(parents=True,exist_ok=True);records=[];seen=set()
    with ZipFile(args.archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        for index,frame in enumerate(manifest['frames']):
            if frame['source_sha256'] in seen:continue
            seen.add(frame['source_sha256'])
            if index%args.stride and frame['geometry'] is None:continue
            raw=z.read(frame['path']);image=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
            for mode in (['automatic','assisted'] if frame['geometry'] else ['automatic']):
                start=time.perf_counter();result=bootstrap_image(image,config,frame['geometry']['corners'] if mode=='assisted' else None)
                record={'frame_id':frame['frame_id'],'source_sha256':hashlib.sha256(raw).hexdigest(),'timestamp_ms':frame['timestamp_ms'],'elapsed_ms':round((time.perf_counter()-start)*1000,2),**result}
                if frame['geometry'] and result['geometry']:
                    before=np.asarray(frame['geometry']['corners']);after=np.asarray(result['geometry']['corners']);record['manual_corner_distance_px']=np.linalg.norm(after-before,axis=1).tolist()
                records.append(record)
                if result['geometry']:cv2.imwrite(str(args.output_dir/f'frame_{index+1:04d}_{mode}.jpg'),draw_bootstrap(image,result))
                print(frame['frame_id'],mode,result['status'],result['reason'],flush=True)
    from src.scanning.marking_model import MODEL_PATH
    model_hash=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() if config.get('learned_markings',{}).get('enabled') and MODEL_PATH.exists() else None
    report={'learned_reader_sha256':model_hash,'schema_version':'guided_bootstrap_evaluation_v1','session_id':manifest['session_id'],'ground_truth_used_for_inference':False,'unique_images_evaluated':len({r['source_sha256'] for r in records}),'records':records,'automatic_candidates':sum(r['mode']=='automatic' and r['status']=='candidate' for r in records),'assisted_candidates':sum(r['mode']=='assisted' and r['status']=='candidate' for r in records),'accuracy_claim':None}
    (args.output_dir/'bootstrap.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
