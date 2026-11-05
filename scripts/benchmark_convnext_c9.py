"""Bounded CPU cost on validation inputs only; does not recompute test accuracy."""
import argparse,json,platform,resource,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.training.dataset_c9 import C9Squares
from src.training.convnext_c9 import make_model
from src.training.multidataset import digest
from src.training.manifest_tools import write_new
from src.vision.classifier import create_classifier
ROOT=Path('outputs/evaluation/convnext_c9')

def main(name):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    manifest=json.loads((ROOT/'approved_cache_with_reviews_v1.json').read_text());ds=C9Squares(manifest,'val')
    path=ROOT/'export_v1/convnextv2_tiny_best.pt' if name=='convnext' else Path('outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt')
    begin=time.perf_counter();state=torch.load(path,map_location='cpu',weights_only=True)
    model=make_model() if name=='convnext' else create_classifier(pretrained=False,mode='fine_tune')
    model.load_state_dict(state['model_state_dict']);model.eval();load=time.perf_counter()-begin
    start=time.perf_counter();x=torch.stack([ds[i][0] for i in range(64)]);preprocessing=time.perf_counter()-start
    times=[]
    with torch.inference_mode():
        for step in range(7):
            begin=time.perf_counter()
            for batch in x.split(16):
                z=model(batch)
                if z.shape!=(len(batch),13) or not torch.isfinite(z).all():raise ValueError('Invalid CPU output')
            if step>=2:times.append(time.perf_counter()-begin)
    cpu=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
    result=dict(schema_version='convnext_c9_cpu_cost_v1',model=name,checkpoint_sha256=digest(path),checkpoint_bytes=path.stat().st_size,parameter_count=sum(p.numel() for p in model.parameters()),model_tensor_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),
        loading_seconds=load,median_forward_per_64_squares_seconds=float(torch.tensor(times).median()),forward_board_seconds=times,preprocessing_first_board_seconds=preprocessing,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,torch_version=torch.__version__,cpu_model=cpu,cpu_threads=2,batch_size=16,device='cpu',platform=platform.platform(),
        input_board_id=ds.rows[0][0]['id'],split='validation_only',inference_mode=True,warmup_board_passes=2,measured_board_passes=5,accuracy_recomputed=False)
    write_new(ROOT/('cpu_'+name+'_v1.json'),result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=['b2','convnext'],required=True);a=p.parse_args();main(a.model)
