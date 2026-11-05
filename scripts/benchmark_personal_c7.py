"""Bounded interleaved CPU timing on identical fixed validation inputs; no tuning."""
import json
import resource
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.training.manifest_tools import rebase_manifest,write_new
from src.training.multidataset import MultiDatasetSquares,digest
from src.vision.classifier import create_classifier

ROOT=Path('outputs/evaluation/personal_c7')
CHECKPOINTS={'b2':'outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt',
 'chesscog':'outputs/evaluation/chesscog_c5/checkpoints/chesscog_best.pt',
 'B':str(ROOT/'checkpoints/B/chesscog_personal_photos_best.pt'),
 'C':str(ROOT/'checkpoints/C/chesscog_personal_multiview_best.pt')}

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    ds=MultiDatasetSquares(rebase_manifest(ROOT/'bundle_v1'),'val');x=torch.stack([ds[i][0] for i in range(64)])
    models={}
    for name,path in CHECKPOINTS.items():
        state=torch.load(path,map_location='cpu',weights_only=True);model=create_classifier(pretrained=False,mode='fine_tune').eval();model.load_state_dict(state['model_state_dict']);models[name]=model
    samples={k:[] for k in models}
    with torch.inference_mode():
        for repetition in range(7):
            # Rotate order to reduce temporal/system-load bias; first two warmups discarded.
            names=list(models);offset=repetition%len(names);names=names[offset:]+names[:offset]
            for name in names:
                begin=time.perf_counter()
                for i in range(0,64,16):models[name](x[i:i+16])
                seconds=time.perf_counter()-begin
                if repetition>=2:samples[name].append(seconds)
    write_new(ROOT/'cpu_timing_v1.json',{'schema_version':'personal_c7_cpu_timing_v1','threads':2,'interop_threads':1,'batch_size':16,'squares_per_measurement':64,'warmup_rounds':2,'measured_rounds':5,'preprocessing_timed':False,'models':{k:{'samples_seconds':v,'median_seconds':float(np.median(v)),'checkpoint_sha256':digest(CHECKPOINTS[k])} for k,v in samples.items()},'combined_process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'note':'All four loaded together; shared-process RSS is not per-model RAM. No accuracy inference is selected from this timing pass.'})

if __name__=='__main__':main()
