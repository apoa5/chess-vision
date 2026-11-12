"""One FP32 ONNX export from the exact C9 checkpoint; no architectural rewrites."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.training.multidataset import digest
from src.training.manifest_tools import write_new
from src.vision.convnext_runtime import CHECKPOINT_SHA,ARCHITECTURE,PROFILE,semantic_weight_hash
from scripts.benchmark_convnext_c11 import ROOT,CONFIG,REGISTRY,Corpus,prepare_runtime,read

def main():
    import torch,onnx
    if not read(ROOT/'reference_c10.json')['saved_top1_reproduced']:raise ValueError('C10 reference prerequisite failed')
    path=ROOT/'convnext_fp32.onnx'
    if path.exists():raise ValueError('ONNX artifact already exists; no silent export replacement')
    runtime,_=prepare_runtime('reference');model=runtime.model
    corpus=Corpus('development');bid=read(REGISTRY)['screening_boards'][0];x,_=corpus.load(bid);example=torch.from_numpy(x[:16])
    before=semantic_weight_hash(model);start=time.perf_counter()
    batch=torch.export.Dim('batch',min=1,max=64)
    program=torch.onnx.export(model,(example,),opset_version=18,dynamo=True,input_names=['square_inputs'],output_names=['logits'],
        dynamic_shapes=({0:batch},),external_data=False,optimize=True,report=True,artifacts_dir=str(ROOT/'export_diagnostics'))
    program.save(str(path),external_data=False)
    exported=onnx.load(str(path));onnx.checker.check_model(exported,full_check=True)
    if semantic_weight_hash(model)!=before or digest(runtime.config['checkpoint'])!=CHECKPOINT_SHA:raise ValueError('Export changed frozen weights')
    metadata=dict(schema_version='convnext_c11_onnx_identity_v1',checkpoint_sha256=CHECKPOINT_SHA,onnx_sha256=digest(path),onnx_bytes=path.stat().st_size,
        architecture=ARCHITECTURE,representation=PROFILE,class_names=runtime.config['class_names'],dtype='float32',input_shape=['batch',4,224,224],output_shape=['batch',13],
        normalization='unchanged existing B2 ImageNet RGB mean/std; binary mask untouched',opset_imports=[dict(domain=o.domain,version=o.version) for o in exported.opset_import],
        logical_weights_sha256=before,license='CC BY-NC4.0 pretrained weights; noncommercial research; export does not expand rights',
        exporter='torch.onnx.export dynamo=True; dynamic batch1..64; spatial/channel dimensions fixed',export_seconds=time.perf_counter()-start,
        source_sha256={p:digest(p) for p in ['src/training/convnext_c9.py','scripts/export_convnext_c11.py','src/vision/crop_experiments.py']})
    meta=ROOT/'convnext_fp32.metadata.json';write_new(meta,metadata)
    write_new(ROOT/'onnx_identity.json',dict(checkpoint_sha256=CHECKPOINT_SHA,onnx_sha256=digest(path),onnx_metadata_sha256=digest(meta)))
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
