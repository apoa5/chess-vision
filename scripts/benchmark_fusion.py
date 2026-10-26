"""Loaded-evidence D.3 CPU timing without classifier calls or allocation tracing."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import read,write
from src.scanning.native_pilot import sha256_file
from src.scanning.offline_fusion import fuse_board


def run(run_dir,output):
    if output.exists():raise ValueError('Fresh benchmark path required')
    settings=read(run_dir/'settings.json');inputs=read(run_dir/'inputs.json');obs=[json.loads(x) for x in Path(inputs['observations_path']).read_text().splitlines()]
    if sha256_file(inputs['observations_path'])!=inputs['observations_sha256']:raise ValueError('Source changed')
    expected=read(run_dir/'predictions.json');timings=[]
    for b in expected['boards']:
        rows=[o for o in obs if o['session_id']==b['session_id'] and any(x['condition']==b['condition'] for x in o['condition_memberships'])]
        samples=[]
        for repeat in range(3):
            t=time.perf_counter();result=fuse_board(rows,b['method'],settings);samples.append(time.perf_counter()-t)
            if result!={k:v for k,v in b.items() if k!='condition'}:raise ValueError('Prediction mismatch')
        timings.append(dict(condition=b['condition'],session_id=b['session_id'],method=b['method'],seconds=samples,median_seconds=statistics.median(samples)))
    write(output,dict(schema_version='offline_fusion_cost_d3_v1',scope='Loaded raw observations; includes full per-square validation and diagnostics, excludes file I/O/model inference. Three repeats, no tracemalloc.',timings=timings))
    print('Benchmarked 75 board/method/condition cases, 225 identical predictions.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.run,a.output)
