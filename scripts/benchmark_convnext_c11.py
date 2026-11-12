"""Bounded, sequential CPU performance/parity experiment; no labels used for selection."""
import argparse,copy,hashlib,importlib.metadata,json,platform,resource,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
cv2.setNumThreads(1)
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import digest,SQUARES,position_key
from src.training.manifest_tools import write_new,labels_to_placement
from src.vision.crop_experiments import tensor_from_packed
from src.vision.convnext_runtime import CHECKPOINT_SHA,PROFILE,load_runtime
ROOT=Path('outputs/evaluation/convnext_c11_v1')
REGISTRY=Path('data/metadata/convnext_c11_corpus_v1.json')
CONFIG=Path('configs/evaluation/convnext_c11_v1.json')
CHECKPOINT='outputs/evaluation/convnext_c9/export_v1/convnextv2_tiny_best.pt'

def read(p):return json.loads(Path(p).read_text())
def value_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def register():
    mpath=Path('outputs/evaluation/convnext_c9/approved_cache_with_reviews_v1.json');m=read(mpath);boards=sorted([b for b in m['boards'] if b['split']=='val'],key=lambda b:b['id'])
    source=read('outputs/evaluation/convnext_c9/export_v1/convnext_val.json');margins=[]
    for o in source['observations']:
        v=sorted(o['logits']);margins.append(dict(board_id=o['board_id'],square=o['square'],margin=v[-1]-v[-2]))
    screening=[];seen=set()
    for b in boards:
        key=position_key(b['placement'])
        if key not in seen:screening.append(b['id']);seen.add(key)
    corpus=dict(schema_version='convnext_c11_corpus_v1',main=dict(manifest=str(mpath),manifest_sha256=digest(mpath),split='val',boards=[dict(id=b['id'],position_key=position_key(b['placement']),orientation=b['semantic_orientation']) for b in boards],square_count=len(boards)*64),
        screening_boards=screening,edge_cases=sorted(margins,key=lambda x:(x['margin'],x['board_id'],x['square']))[:16],
        regression=dict(manifest='data/metadata/convnext_independent_c10_manifest_v1.json',manifest_sha256=digest('data/metadata/convnext_independent_c10_manifest_v1.json'),
            predictions='outputs/evaluation/convnext_c10_v1/convnext_result.json',predictions_sha256=digest('outputs/evaluation/convnext_c10_v1/convnext_result.json'),use='fixed reproduction/regression only; not performance/accuracy selection'),
        internal_test_opened=False,final_holdout_access=False,labels_used_for_runtime_selection=False)
    base=dict(checkpoint=CHECKPOINT,checkpoint_sha256=CHECKPOINT_SHA,representation=PROFILE,class_names=list(CLASS_NAMES),dtype='float32',interop_threads=1)
    candidates={'reference':dict(base,backend='pytorch_reference',threads=2,batch_size=16)}
    for batch in (1,8,16,32):candidates[f'pt_cl_b{batch}_t2']=dict(base,backend='pytorch_channels_last',threads=2,batch_size=batch)
    for threads in (1,4):candidates[f'pt_cl_b8_t{threads}']=dict(base,backend='pytorch_channels_last',threads=threads,batch_size=8)
    candidates['pt_nchw_b8_t2']=dict(base,backend='pytorch_reference',threads=2,batch_size=8)
    for batch,threads in ((8,2),(8,4),(16,2)):
        candidates[f'ort_b{batch}_t{threads}']=dict(base,backend='onnx_cpu',threads=threads,batch_size=batch,onnx_path=str(ROOT/'convnext_fp32.onnx'),onnx_metadata=str(ROOT/'convnext_fp32.metadata.json'))
    config=dict(schema_version='convnext_c11_config_v1',candidates=candidates,screening_rounds=3,confirmation_rounds=5,warmup_passes=1,
        selection='Fastest screened zero-top1-change FP32 candidate per runtime family; full development corpus parity gates promotion; C10 is a final regression veto, never tuning.',
        numeric_tolerance='Freeze after screening, before full parity: atol=max(1e-6,2*max screening absolute FP32 logit difference among chosen finalists),rtol=1e-5. No top1 or FEN changes permitted regardless of tolerance. Never relax after full/C10 results.',
        quantization=False,torch_compile=False,gpu=False,cloud=False,preprocessing_changed=False)
    if digest(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Checkpoint changed')
    write_new(REGISTRY,corpus);write_new(CONFIG,config)
    expected=read('outputs/evaluation/convnext_c10_v1/protected_inputs.json')['protected_sha256']
    expected.update(read('outputs/evaluation/convnext_c10_v1/execution_contract.json')['sha256'])
    expected.update(read('data/metadata/convnext_independent_c10_results_v1.json')['artifacts_sha256'])
    for root in ('outputs/evaluation/convnext_c10_v1','outputs/evaluation/convnext_c9'):
        for p in Path(root).rglob('*'):
            if p.is_file() and 'holdout' not in str(p).lower():expected[str(p)]=digest(p)
    for p in ('data/metadata/convnext_independent_c10_results_v1.json',str(REGISTRY),str(CONFIG)):expected[p]=digest(p)
    write_new(ROOT/'protected_inputs.json',dict(sha256=expected,final_holdout_access=False))
    print(json.dumps(dict(development_boards=len(boards),development_squares=len(boards)*64,positions=len(seen),screening=screening,minimum_saved_margin=corpus['edge_cases'][0]['margin']),indent=2))

class Corpus:
    def __init__(self,kind):
        self.kind=kind;registry=read(REGISTRY)
        if kind=='c10':
            c=registry['regression'];self.manifest=read(c['manifest'])
            if digest(c['manifest'])!=c['manifest_sha256']:raise ValueError('C10 manifest changed')
            self.ids=[p['id'] for p in self.manifest['photos']]
        else:
            from src.training.dataset_c9 import C9Squares
            c=registry['main'];self.manifest=read(c['manifest'])
            if digest(c['manifest'])!=c['manifest_sha256']:raise ValueError('C9 corpus changed')
            self.ds=C9Squares(self.manifest,'val');self.index={}
            for i,(b,o) in enumerate(self.ds.rows):self.index.setdefault(b['id'],{})[o['square']]=i
            self.ids=[b['id'] for b in c['boards']]
    def load(self,bid,preallocate=False):
        start=time.perf_counter()
        if self.kind=='c10':
            p=next(p for p in self.manifest['photos'] if p['id']==bid);i=p['inputs']
            if digest(i['path'])!=i['sha256']:raise ValueError('C10 inputs changed')
            packed=np.load(i['path'])['packed']
            if [value_hash(p) for p in packed]!=i['packed_sha256']:raise ValueError('C10 RGB/mask mismatch')
        else:packed=np.stack([self.ds.packed(self.index[bid][s]) for s in SQUARES])
        extraction=time.perf_counter()-start;start=time.perf_counter()
        if preallocate:
            from src.vision.convnext_runtime import prepare_board_inputs
            array=prepare_board_inputs(packed)
        else:array=np.stack([tensor_from_packed(p) for p in packed])
        normalization=time.perf_counter()-start
        return array,dict(extraction_seconds=extraction,normalization_seconds=normalization,input_sha256=value_hash(array))

def prepare_runtime(candidate):
    start=time.perf_counter();config=read(CONFIG)['candidates'][candidate]
    if config['backend']=='onnx_cpu':
        import onnxruntime
        identity=read(ROOT/'onnx_identity.json')
        if identity['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('ONNX export source differs')
        config.update(onnx_sha256=identity['onnx_sha256'],onnx_metadata_sha256=identity['onnx_metadata_sha256'])
    else:import torch
    imports=time.perf_counter()-start;runtime=load_runtime(config)
    return runtime,imports

def parity(reference,actual,atol=0.,rtol=0.):
    ref=np.asarray(reference,np.float32);z=np.asarray(actual,np.float32)
    if ref.shape!=(64,13) or z.shape!=ref.shape or not np.isfinite(z).all():raise ValueError('Invalid logits/order')
    delta=np.abs(z.astype(np.float64)-ref);a,b=ref.argmax(1),z.argmax(1);margin=np.sort(ref,axis=1)[:,-1]-np.sort(ref,axis=1)[:,-2]
    return dict(max_abs_logit_error=float(delta.max()),sum_abs_logit_error=float(delta.sum()),logit_count=int(delta.size),mean_abs_logit_error=float(delta.mean()),
        top1_disagreements=int((a!=b).sum()),changed_squares=[SQUARES[i] for i in np.flatnonzero(a!=b)],board_changed=not np.array_equal(a,b),
        reference_placement=labels_to_placement(a.tolist()),candidate_placement=labels_to_placement(b.tolist()),numeric_within_tolerance=bool(np.allclose(ref,z,atol=atol,rtol=rtol)),
        minimum_reference_top2_logit_margin=float(margin.min()),max_top2_logit_margin_difference=float(abs((np.sort(z,axis=1)[:,-1]-np.sort(z,axis=1)[:,-2])-margin).max()))

def metadata(runtime,imports):
    result=dict(initialization_seconds=runtime.initialization_seconds,backend_library_import_seconds=imports,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        config=runtime.config,cpu_model=next((v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')),'unknown'),platform=platform.platform(),
        torch_version=importlib.metadata.version('torch'),numpy_version=np.__version__,cpu_threads=runtime.config['threads'],interop_threads=1)
    if runtime.config['backend']=='onnx_cpu':
        result['onnxruntime_version']=importlib.metadata.version('onnxruntime')
        result['execution_providers']=runtime.session.get_providers()
    else:result['weight_values_sha256']=runtime.weight_values_sha256
    return result

def reference(kind):
    output=ROOT/f'reference_{kind}.json'
    if output.exists() or (ROOT/f'reference_{kind}_logits.npz').exists():raise ValueError('Reference already recorded; use a fresh versioned run')
    # C10 must reproduce before a new development reference is established.
    if kind=='development' and not read(ROOT/'reference_c10.json')['saved_top1_reproduced']:raise ValueError('C10 reference reproduction gate failed')
    corpus=Corpus('c10' if kind=='c10' else 'development');runtime,imports=prepare_runtime('reference');rows=[];all_z=[]
    saved=read(read(REGISTRY)['regression']['predictions']) if kind=='c10' else read('outputs/evaluation/convnext_c9/export_v1/convnext_val.json')
    old={}
    if kind=='c10':
        for row in saved['rows']:old[row['position_id']]=np.array([o['raw_logits'] for o in row['observations']],np.float32)
    else:
        for bid in corpus.ids:
            by_square={o['square']:o['logits'] for o in saved['observations'] if o['board_id']==bid}
            old[bid]=np.array([by_square[s] for s in SQUARES],np.float32)
    for bid in corpus.ids:
        x,prep=corpus.load(bid);z=runtime.predict_logits(x);all_z.append(z);comparison=parity(old[bid],z,atol=1e-5,rtol=1e-5)
        if comparison['top1_disagreements'] or not comparison['numeric_within_tolerance']:raise ValueError('Frozen reference differs: '+bid)
        rows.append(dict(id=bid,**prep,timings=dict(runtime.last_timings),saved_parity=comparison));print(kind,bid,runtime.last_timings['total_seconds'],flush=True)
    np.savez_compressed(ROOT/f'reference_{kind}_logits.npz',logits=np.stack(all_z))
    result=dict(schema_version='convnext_c11_reference_v1',kind=kind,saved_top1_reproduced=True,rows=rows,**metadata(runtime,imports),
        saved_backend_note='C10 original PyTorch FP32 batch16 threads2; C9 saved CPU batch32 compared explicitly to the unchanged C10 batch16 reference',checkpoint_sha256=CHECKPOINT_SHA)
    write_new(output,result)

def screen(candidate):
    if (ROOT/'screens'/f'{candidate}.json').exists():raise ValueError('Screen already recorded')
    if not read(ROOT/'reference_c10.json')['saved_top1_reproduced']:raise ValueError('Reference not reproduced')
    cfg=read(CONFIG);registry=read(REGISTRY);ids=registry['screening_boards'];corpus=Corpus('development');runtime,imports=prepare_runtime(candidate)
    refs=read(ROOT/'reference_development.json');positions={r['id']:i for i,r in enumerate(refs['rows'])};logits=np.load(ROOT/'reference_development_logits.npz')['logits'];arrays={};rows=[]
    for bid in ids:arrays[bid]=corpus.load(bid)[0]
    for _ in range(cfg['warmup_passes']):runtime.predict_logits(arrays[ids[0]])
    measurements={bid:[] for bid in ids};comparisons={}
    for repeat in range(cfg['screening_rounds']):
        for bid in ids[repeat%len(ids):]+ids[:repeat%len(ids)]:
            x=arrays[bid];before=value_hash(x);z=runtime.predict_logits(x)
            if before!=value_hash(x) or before!=refs['rows'][positions[bid]]['input_sha256']:raise ValueError('Runtime changed input values')
            measurements[bid].append(dict(runtime.last_timings));comparisons[bid]=parity(logits[positions[bid]],z)
    result=dict(schema_version='convnext_c11_screen_v1',candidate=candidate,rows=[dict(id=bid,measurements=measurements[bid],parity=comparisons[bid]) for bid in ids],
        median_forward_seconds=float(np.median([v['model_forward_seconds'] for a in measurements.values() for v in a])),
        median_total_seconds=float(np.median([v['total_seconds'] for a in measurements.values() for v in a])),
        top1_disagreements=sum(v['top1_disagreements'] for v in comparisons.values()),max_abs_logit_error=max(v['max_abs_logit_error'] for v in comparisons.values()),**metadata(runtime,imports))
    write_new(ROOT/'screens'/f'{candidate}.json',result);print(json.dumps({k:result[k] for k in ('candidate','median_forward_seconds','median_total_seconds','top1_disagreements','max_abs_logit_error','peak_rss_bytes')}))

def full(candidate):
    if (ROOT/f'{candidate}_full_parity.json').exists() or (ROOT/f'{candidate}_development_logits.npz').exists():raise ValueError('Full parity already recorded')
    finalists=read(ROOT/'finalists.json');corpus=Corpus('development');runtime,imports=prepare_runtime(candidate);reference=read(ROOT/'reference_development.json');old=np.load(ROOT/'reference_development_logits.npz')['logits'];rows=[];values=[]
    for i,bid in enumerate(corpus.ids):
        x,prep=corpus.load(bid,preallocate=True)
        if prep['input_sha256']!=reference['rows'][i]['input_sha256']:raise ValueError('Parity input changed')
        z=runtime.predict_logits(x);values.append(z);check=parity(old[i],z,finalists['atol'],finalists['rtol']);rows.append(dict(id=bid,parity=check,timings=dict(runtime.last_timings)));print(candidate,i+1,'/',len(corpus.ids),flush=True)
    np.savez_compressed(ROOT/f'{candidate}_development_logits.npz',logits=np.stack(values))
    result=dict(schema_version='convnext_c11_full_parity_v1',candidate=candidate,rows=rows,top1_disagreements=sum(r['parity']['top1_disagreements'] for r in rows),
        board_disagreements=sum(r['parity']['board_changed'] for r in rows),numeric_pass=all(r['parity']['numeric_within_tolerance'] for r in rows),
        max_abs_logit_error=max(r['parity']['max_abs_logit_error'] for r in rows),mean_abs_logit_error=sum(r['parity']['sum_abs_logit_error'] for r in rows)/sum(r['parity']['logit_count'] for r in rows),**metadata(runtime,imports))
    write_new(ROOT/f'{candidate}_full_parity.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','reference-c10','reference-development','screen','full']);p.add_argument('--candidate');a=p.parse_args()
    if a.action=='register':register()
    elif a.action.startswith('reference'):reference('c10' if a.action=='reference-c10' else 'development')
    elif a.action=='screen':screen(a.candidate)
    else:full(a.candidate)
