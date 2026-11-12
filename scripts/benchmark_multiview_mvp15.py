"""Registered real-input CPU benchmark, strict parity and ConvNeXt fusion audit."""
import argparse,hashlib,json,resource,sys,time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
cv2.setNumThreads(1)
from src.training.multidataset import SQUARES,digest,placement_labels
from src.training.manifest_tools import write_new,labels_to_placement
from src.scanning.video_keyframes import select_keyframes
from src.scanning.multiview_inference import STRATEGIES,MultiViewRunner,aggregate_views
from src.scanning.assisted_photo import PhotoService
from src.vision.convnext_runtime import prepare_board_inputs,CHECKPOINT_SHA
ROOT=Path('outputs/evaluation/assisted_mvp15_v1')
REGISTRY=Path('data/metadata/assisted_mvp15_corpus_v1.json')

def read(p):return json.loads(Path(p).read_text())
def value_hash(x):return hashlib.sha256(x.tobytes()).hexdigest()

def register():
    evidence=read('data/metadata/evidence_d2_manifest_v1.json')
    settings=read('configs/experiments/keyframes_d1.json')
    source=read('outputs/scanning/d1_pilot_run1/selection_run2/features.json')
    features={r['frame_id']:r for d in source['datasets'] if d['name']=='primary_3s_high' for s in d['sessions'] for r in s['policies']['marked-board']}
    grouped=defaultdict(list)
    for f in evidence['frames']:
        if f['human_review']!='human_verified':raise ValueError('Reviewed prepared geometry required')
        r=dict(features[f['frame_id']],geometry_state='visually_verified');grouped[f['session_id']].append((f,r))
    sessions=[];frames=[]
    c9=read('outputs/evaluation/convnext_c9/approved_cache_with_reviews_v1.json')
    for sid,pairs in sorted(grouped.items()):
        selection=select_keyframes([r for f,r in pairs],'quality_diverse',3,settings)
        chosen=[(f,r) for f,r in pairs if f['frame_id'] in selection['selected']]
        chosen.sort(key=lambda p:p[0]['timestamp_ms'])
        if len(chosen)!=3:raise ValueError('Expected three reviewed complementary views for offline comparison')
        position=pairs[0][0]['position_id'];labels=next(b for b in c9['boards'] if b['position_id']==position)
        sessions.append(dict(id=sid,position_id=position,frame_ids=[f['frame_id'] for f,r in chosen],placement=labels['placement'],label_version=labels.get('label_version','C9 owner-reviewed placement'),selection=selection))
        for f,r in chosen:
            g=read(f['geometry_reference'])['policies']['marked-board']
            frames.append(dict(id=f['frame_id'],session_id=sid,kind='d2',input_cache=f['input_cache'],input_cache_sha256=f['input_cache_sha256'],square_ids=g['semantic_square_ids'],quality=r['quality_score'],timestamp_ms=f['timestamp_ms'],raw_path=f['d1_metadata']['path'],raw_sha256=f['d1_metadata']['derivative_sha256']))
    edge=read('data/metadata/convnext_c11_corpus_v1.json')['edge_cases']
    hard=[]
    for e in edge:
        if e['board_id'] not in hard:hard.append(e['board_id'])
        if len(hard)==3:break
    for bid in hard:frames.append(dict(id=bid,kind='c9',quality=None))
    main=sessions[0]['frame_ids'];parity=main+sessions[1]['frame_ids']+hard
    registry=dict(schema_version='assisted_mvp15_corpus_v1',frames=frames,sessions=sessions,timing_ids=main,parity_ids=parity,
        checkpoint_sha256=CHECKPOINT_SHA,numeric_tolerance=dict(atol=2.6702880859375e-5,rtol=1e-5),
        strategies=STRATEGIES,warmups=1,rounds=3,minimum_registered_reference_margin=edge[0],
        ground_truth_used_for_selection=False,development_only=True,c10_used=False,final_holdout_access=False,
        fusion_promotion='Strict gain in correct squares; no extra empty FP, no worse occupied errors in any session, no lost exact boards, no new >90% confident wrong occupied square. Otherwise retain best_quality.',
        source_sha256={p:digest(p) for p in ['data/metadata/evidence_d2_manifest_v1.json','data/metadata/evidence_d2_ground_truth_corrections_v1.json','configs/experiments/keyframes_d1.json','outputs/evaluation/convnext_c9/approved_cache_with_reviews_v1.json']})
    write_new(REGISTRY,registry);print('registered',len(frames),'frames;',len(parity),'parity frames;',len(sessions),'fusion sessions')

class Inputs:
    def __init__(self):
        self.registry=read(REGISTRY);self.frames={f['id']:f for f in self.registry['frames']};self.c9=None
    def load(self,bid):
        f=self.frames[bid]
        if f['kind']=='d2':
            if digest(f['input_cache'])!=f['input_cache_sha256']:raise ValueError('Frozen packed input changed')
            packed=np.load(f['input_cache'])['packed'];lookup=dict(zip(f['square_ids'],packed));packed=np.stack([lookup[s] for s in SQUARES])
        else:
            if self.c9 is None:
                from src.training.dataset_c9 import C9Squares
                self.c9=C9Squares(read('outputs/evaluation/convnext_c9/approved_cache_with_reviews_v1.json'),'val');self.indices={}
                for i,(b,o) in enumerate(self.c9.rows):self.indices.setdefault(b['id'],{})[o['square']]=i
            packed=np.stack([self.c9.packed(self.indices[bid][s]) for s in SQUARES])
        return prepare_board_inputs(packed)

def reference():
    if (ROOT/'reference.json').exists():raise ValueError('Reference exists')
    data=Inputs();runtime=PhotoService().runtime();values=[];rows=[]
    for f in data.registry['frames']:
        x=data.load(f['id']);z=runtime.predict_logits(x);values.append(z);rows.append(dict(id=f['id'],input_sha256=value_hash(x)))
        print('reference',f['id'],flush=True)
    np.savez_compressed(ROOT/'reference_logits.npz',logits=np.stack(values))
    write_new(ROOT/'reference.json',dict(rows=rows,checkpoint_sha256=CHECKPOINT_SHA,source='unchanged C11 optimized runtime',square_order=list(SQUARES)))

def candidate(name):
    if (ROOT/f'benchmark_{name}.json').exists():raise ValueError('Benchmark exists')
    data=Inputs();r=data.registry;ref=read(ROOT/'reference.json');old=np.load(ROOT/'reference_logits.npz')['logits'];indices={row['id']:i for i,row in enumerate(ref['rows'])}
    service=PhotoService();runtime=service.runtime();runner=MultiViewRunner(runtime)
    arrays=[data.load(bid) for bid in r['timing_ids']]
    runner.predict(arrays,name)
    samples=[];timed_values=None
    for repeat in range(r['rounds']):
        before=resource.getrusage(resource.RUSAGE_SELF);start=time.perf_counter();timed_values=runner.predict(arrays,name);wall=time.perf_counter()-start;after=resource.getrusage(resource.RUSAGE_SELF)
        original=np.stack([old[indices[bid]] for bid in r['timing_ids']])
        samples.append(dict(seconds=wall,samples_per_second=192/wall,cpu_percent=100*((after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime))/wall,
                            top1_changes=int((timed_values.argmax(2)!=original.argmax(2)).sum()),numeric_pass=bool(np.allclose(timed_values,original,**r['numeric_tolerance']))))
    checks=[];allvalues=[]
    for i,bid in enumerate(r['parity_ids']):
        x=data.load(bid)
        if value_hash(x)!=ref['rows'][indices[bid]]['input_sha256']:raise ValueError('Input changed')
        # First timing triplet is reused; remaining views are evaluated in one
        # three-view call so the concurrent path itself is checked throughout.
        if i<3:z=timed_values[i]
        else:z=None
        allvalues.append(z)
    for lo in range(3,len(r['parity_ids']),3):
        ids=r['parity_ids'][lo:lo+3];values=runner.predict([data.load(bid) for bid in ids],name)
        allvalues[lo:lo+len(ids)]=list(values)
    for bid,z in zip(r['parity_ids'],allvalues):
        original=old[indices[bid]];delta=np.abs(z.astype(np.float64)-original);a=original.argmax(1);b=z.argmax(1)
        checks.append(dict(id=bid,max_abs=float(delta.max()),mean_abs=float(delta.mean()),top1_changes=int((a!=b).sum()),board_changed=not np.array_equal(a,b),numeric_pass=bool(np.allclose(z,original,**r['numeric_tolerance'])),minimum_margin=float(np.diff(np.sort(original,axis=1)[:,-2:],axis=1).min())))
    write_new(ROOT/f'benchmark_{name}.json',dict(strategy=name,config=STRATEGIES[name],samples=samples,median_seconds=float(np.median([s['seconds'] for s in samples])),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,checks=checks,parity_pass=all(not c['top1_changes'] and c['numeric_pass'] for c in checks) and all(not s['top1_changes'] and s['numeric_pass'] for s in samples),model_instances=1))
    print(name,'median',np.median([s['seconds'] for s in samples]),'parity',sum(c['top1_changes'] for c in checks),flush=True)

def choose():
    results={name:read(ROOT/f'benchmark_{name}.json') for name in STRATEGIES}
    good=[n for n,r in results.items() if r['parity_pass']]
    best=min(good,key=lambda n:results[n]['median_seconds']) if good else 'sequential'
    write_new(ROOT/'inference_selection_v1.json',dict(strategy=best,results=results,selection='fastest median among strict categorical/numerical parity passes; development only',no_c10=True))

def fusion():
    from src.scanning.square_evidence import classification_metrics
    data=Inputs();ref=read(ROOT/'reference.json');values=np.load(ROOT/'reference_logits.npz')['logits'];by={r['id']:z for r,z in zip(ref['rows'],values)};rows=[]
    for session in data.registry['sessions']:
        z=np.stack([by[i] for i in session['frame_ids']]);q=[data.frames[i]['quality'] for i in session['frame_ids']];truth=np.array(placement_labels(session['placement']));methods={}
        baseline=aggregate_views(z,q,'best_quality')['predictions']
        for method in ('best_quality','mean_probabilities','mean_logits'):
            result=aggregate_views(z,q,method);pred=result['predictions'];metrics=classification_metrics(truth,pred)
            metrics.update(exact=bool(np.array_equal(truth,pred)),corrections=int((truth!=pred).sum()),confident_wrong_occupied=int(((truth!=0)&(truth!=pred)&(result['scores'].max(1)>.9)).sum()),corrected=int(((baseline!=truth)&(pred==truth)).sum()),introduced=int(((baseline==truth)&(pred!=truth)).sum()))
            metrics['confident_wrong_occupied_squares']=[SQUARES[i] for i in np.flatnonzero((truth!=0)&(truth!=pred)&(result['scores'].max(1)>.9))]
            methods[method]=dict(metrics=metrics,predictions=pred.tolist(),scores=result['scores'].tolist(),placement=labels_to_placement(pred.tolist()))
        rows.append(dict(session_id=session['id'],position_id=session['position_id'],frame_ids=session['frame_ids'],label_version=session['label_version'],truth=truth.tolist(),methods=methods))
    eligible=[]
    for method in ('mean_probabilities','mean_logits'):
        if sum(r['methods'][method]['metrics']['corrections'] for r in rows)>=sum(r['methods']['best_quality']['metrics']['corrections'] for r in rows):continue
        if all(r['methods'][method]['metrics']['empty_false_positives']<=r['methods']['best_quality']['metrics']['empty_false_positives'] and
               r['methods'][method]['metrics']['occupied_correct']>=r['methods']['best_quality']['metrics']['occupied_correct'] and
               (not r['methods']['best_quality']['metrics']['exact'] or r['methods'][method]['metrics']['exact']) and
               set(r['methods'][method]['metrics']['confident_wrong_occupied_squares'])<=set(r['methods']['best_quality']['metrics']['confident_wrong_occupied_squares']) for r in rows):eligible.append(method)
    selected=min(eligible,key=lambda m:sum(r['methods'][m]['metrics']['corrections'] for r in rows)) if eligible else 'best_quality'
    write_new(ROOT/'fusion_v1.json',dict(rows=rows,selected_policy=selected,development_only=True,independent_generalization=False,correlated_frames=True,model_training_exposure=True,rule=data.registry['fusion_promotion']))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','reference','candidate','choose','fusion']);p.add_argument('--strategy',choices=STRATEGIES);p.add_argument('--output-root',type=Path,default=ROOT);p.add_argument('--registry',type=Path,default=REGISTRY);a=p.parse_args();ROOT=a.output_root;REGISTRY=a.registry;ROOT.mkdir(parents=True,exist_ok=True)
    if a.action=='candidate':candidate(a.strategy)
    else:globals()[a.action]()
