"""Read-only D.2 provenance/pixel checks; optionally reproduce inputs without inference."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,check_hashes,write
from scripts.run_square_evidence import make_inputs
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import verify_reviews,validate_bundle
from src.vision.perspective import warp_board_with_context


def validate(path,reproduce_inputs=False):
    start=time.perf_counter();e=read(path);check_hashes({e['packet_path']:e['packet_sha256'],e['reviews_path']:e['reviews_sha256'],
        e['config_path']:e['config_sha256'],e['observations_path']:e['observations_sha256'],e['model']['checkpoint_path']:e['model']['checkpoint_sha256']})
    check_hashes(e['model']['source_sha256'])
    packet=read(e['packet_path']);reviews=read(e['reviews_path']);states=verify_reviews(packet,reviews,e['packet_sha256']);config=read(e['config_path'])
    observations=[json.loads(x) for x in Path(e['observations_path']).read_text().splitlines()];result=validate_bundle(e,observations)
    rows={r['review_id']:r for r in packet['frames']};pixels=0;contexts=0;cv2.setNumThreads(1)
    for s in e['sessions']:check_hashes({s['source_video']['path']:s['source_video']['sha256']})
    for f in e['frames']:
        if not f['observations']:continue
        row=rows[f['review_id']];original=row['frame']
        if states[f['review_id']]['human_review']!='human_verified':raise ValueError('No human approval')
        check_hashes({original['path']:original['derivative_sha256'],original['geometry_reference']:original['geometry_reference_sha256'],f['input_cache']:f['input_cache_sha256']})
        raw=read(original['geometry_reference']);g=raw['policies']['marked-board']
        if g['source_square_polygons']!=row['source_square_polygons']:raise ValueError('Reviewed grid mismatch')
        if reproduce_inputs:
            image=cv2.imread(original['path']);made=make_inputs(image,g,config);packed=np.load(f['input_cache'])['packed']
            if not np.array_equal(np.stack([x['packed'] for x in made]),packed):raise ValueError('Packed inputs failed pixel replay')
            pixels+=len(made)
            saved=Path(g['overlay_path']).parent/'context.png'
            if not saved.exists():saved=Path(g['overlay_path']).parent/'legacy/context.png'
            context=warp_board_with_context(image,np.float32(g['provenance']['source_corners']),config['board_size'],config['context_margin']).image
            if not np.array_equal(cv2.imread(str(saved)),context):raise ValueError('D.1 context pixels differ')
            contexts+=1
    result.update(human_verified_unique_frames=sum(x['human_review']=='human_verified' for x in states.values()),
                  replayed_packed_square_inputs=pixels,identical_saved_D1_contexts=contexts,validation_seconds=time.perf_counter()-start)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);p.add_argument('--reproduce-inputs',action='store_true');p.add_argument('--report',type=Path);a=p.parse_args();result=validate(a.manifest,a.reproduce_inputs)
    if a.report:write(a.report,result)
    print(json.dumps(result))
