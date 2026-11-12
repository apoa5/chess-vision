"""Register and compare three fresh photographs using frozen CPU classifiers only."""
import argparse,copy,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from PIL import Image
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import digest,position_key,placement_labels,SQUARES
from src.training.manifest_tools import write_new
from src.vision.independent_c10 import (FENS,CHECKPOINTS,independence,validate_manifest,verify_checkpoint,canonical_inputs,score,cohort_metrics)
ROOT=Path('outputs/evaluation/convnext_c10_v1')
MANIFEST=Path('data/metadata/convnext_independent_c10_manifest_v1.json')
CONFIG=Path('configs/evaluation/convnext_independent_c10_v1.json')

def register():
    from src.utils.config import load_config
    from scripts.run_square_evidence import make_inputs
    from src.utils.image_io import load_image
    c9_path=Path('outputs/evaluation/convnext_c9/frozen_manifest_v1.json');c9=json.loads(c9_path.read_text())
    historical=[dict(id=b['id'],placement=b['placement'],split=b['split'],source_hashes=[b['source']['sha256'],b.get('original_source',b['source'])['sha256']]) for b in c9['boards']]
    inventory=json.loads(Path('outputs/evaluation/convnext_c9/inventory_v1.json').read_text())
    for p in inventory['photos']:
        if not any(b['id']=='personal:'+p['id'] for b in historical):historical.append(dict(id='historical:'+p['id'],placement=p['fen'],split='historically_excluded',source_hashes=[p['sha256']]))
    ds=json.loads(Path('configs/experiments/evidence_d2.json').read_text())
    crop={k:ds[k] for k in ('board_size','context_margin','initial_context_padding','crop_spec')}
    def image_dhash(path):
        a=cv2.resize(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE),(9,8),interpolation=cv2.INTER_AREA)
        return np.packbits(a[:,1:]>a[:,:-1]).tobytes()
    prior_visual=[(p,image_dhash(p['source_path'])) for p in inventory['photos']]
    current=load_config();geometry={k:current[k] for k in ('board','board_detection','orientation')}
    config=dict(schema_version='convnext_independent_c10_config_v1',models=CHECKPOINTS,class_names=list(CLASS_NAMES),crop=crop,geometry=geometry,
        geometry_policy='legacy/default',settings_path='config/settings.yaml',settings_sha256=digest('config/settings.yaml'),batch_size=16,cpu_threads=2,
        device='cpu',dtype='float32',warmup_passes=2,timing_rounds=5,weights_or_threshold_tuning=False,training_enabled=False,paid_cloud=False,
        secondary_geometry='not_available_without_explicit_owner_corner_review')
    for spec in CHECKPOINTS.values():verify_checkpoint(spec['path'],spec['sha256'])
    write_new(CONFIG,config);photos=[]
    for n,fen in enumerate(FENS,1):
        pid=f'c10_position_{n:02d}';path=Path(f'data/raw/conv_test/convNext_{n:02d}.jpg');image=Image.open(path);source_sha=digest(path)
        gpath=ROOT/pid/'automatic/result.json';g=json.loads(gpath.read_text());g['evaluation_visual_inspection']='assistant_inspected_no_gross_grid_or_orientation_error; fine geometry not owner-verified'
        audit=independence(fen,source_sha,historical)
        fingerprint=image_dhash(path)
        nearest=sorted([dict(filename=p['id'],distance_bits=sum((a^b).bit_count() for a,b in zip(fingerprint,h)),position_id=p['position_id']) for p,h in prior_visual],key=lambda v:(v['distance_bits'],v['filename']))[:5]
        audit['raw_photo_dhash_nearest']=nearest
        audit['perceptual_hash_interpretation']='64-bit raw-image proxy only; similarity is not identical piece placement. Printed-coordinate visual FEN cross-check and canonical placement audit govern independence.'
        photo=dict(id=pid,filename=path.name,path=str(path),sha256=source_sha,resolution=list(image.size),exif_orientation=image.getexif().get(274),
            exif_make=image.getexif().get(271),exif_model=image.getexif().get(272),exif_capture_time=image.getexif().get(36867),fen=fen,
            ground_truth=dict(source='owner_supplied_reference_FEN',association='explicit numbered filename-to-FEN mapping in C10 request',
                photo_agreement='assistant visual cross-check consistent; no labels inferred from classifier outputs',separate_owner_corner_attestation=False),
            physical_orientation=dict(value='white_near_camera',a1_corner='bottom_left_playing_area',basis='visible printed A-H at near edge and 1-8 ranks; assistant visual inspection',owner_corner_review=False),
            capture_session_id=None,capture_conditions=dict(owner_documented=None,observable='indoor; board on patterned bedding; strong piece shadows',room=None,lighting_type=None,camera_elevation=None),
            provenance='Owner uploaded three fresh independent-evaluation photographs; not authorized for training',independence=audit,
            automatic_geometry=g,geometry_path=str(gpath),geometry_sha256=digest(gpath),secondary_verified_geometry=None)
        if g['completed']:
            inputs=canonical_inputs(make_inputs(load_image(path),g,crop));packed=np.stack([i['packed'] for i in inputs]);cache=ROOT/pid/'packed_inputs.npz';np.savez_compressed(cache,packed=packed)
            photo['inputs']=dict(path=str(cache),sha256=digest(cache),squares=list(SQUARES),packed_sha256=[hashlib.sha256(p.tobytes()).hexdigest() for p in packed],
                target_valid_fractions=[i['target_valid_fraction'] for i in inputs],pixel_shape=list(packed.shape),raw_fen_used_for_extraction=False)
        photos.append(photo)
    manifest=dict(schema_version='convnext_independent_c10_manifest_v1',designation='limited_independent_position_evaluation_only',training_eligible=False,
        final_holdout_access=False,class_names=list(CLASS_NAMES),config_path=str(CONFIG),config_sha256=digest(CONFIG),c9_manifest_path=str(c9_path),c9_manifest_sha256=digest(c9_path),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),photos=photos,independence_units='three physical positions; shared capture conditions may be correlated')
    validate_manifest(manifest);write_new(MANIFEST,manifest)
    # Snapshot recorded history and every currently tracked file; never discover a holdout directory.
    expected=json.loads(Path('outputs/evaluation/convnext_c9/protected_inputs_v1.json').read_text())['protected_sha256']
    expected.update(json.loads(Path('data/metadata/convnext_c9_results_v1.json').read_text())['artifacts'])
    for p in subprocess.check_output(['git','ls-files','-z']).decode().split('\0'):
        if p and 'holdout' not in p.lower() and Path(p).is_file():expected[p]=digest(p)
    for p in inventory['photos']:expected[p['source_path']]=p['sha256']
    for s in inventory['sessions']:expected[s['source']['path']]=s['source']['sha256']
    for spec in CHECKPOINTS.values():expected[spec['path']]=spec['sha256']
    for p in ROOT.parent.joinpath('convnext_c9').rglob('*'):
        if p.is_file() and 'holdout' not in str(p).lower():expected[str(p)]=digest(p)
    for p in photos:expected[p['path']]=p['sha256']
    for path,sha in expected.items():
        if 'holdout' in path.lower():raise ValueError('Holdout path prohibited')
        if digest(path)!=sha:raise ValueError('Historical input already differs: '+path)
    write_new(ROOT/'protected_inputs.json',dict(protected_sha256=expected,final_holdout_access=False))
    print(json.dumps({'registered':len(photos),'independent_eligible':sum(p['independence']['independent_eligible'] for p in photos),'automatic_completed':sum(p['automatic_geometry']['completed'] for p in photos)},indent=2))

def timed_preparation(path,config):
    """Time the exact existing raw pipeline via transparent call wrappers."""
    from contextlib import ExitStack
    from unittest.mock import patch
    from src.vision import position_pipeline as pipeline
    times={};images=[]
    def wrap(name,original):
        def measured(*args,**kwargs):
            t=time.perf_counter()
            try:
                result=original(*args,**kwargs)
                if name=='load_image':images.append(result)
                return result
            finally:times[name]=times.get(name,0)+time.perf_counter()-t
        return measured
    start=time.perf_counter()
    with ExitStack() as stack:
        for name in ('load_image','detect_board_corners','rectify_with_config','resolve_orientation','fit_grid'):
            stack.enter_context(patch.object(pipeline,name,wrap(name,getattr(pipeline,name))))
        result=pipeline.prepare_raw_photo(path,config)
    times['raw_preparation_total']=time.perf_counter()-start
    return result,images[0],times

def infer(name):
    import torch
    from src.training.convnext_c9 import make_model
    from src.vision.classifier import create_classifier
    from src.vision.crop_experiments import tensor_from_packed
    from src.vision.dataset import chess_square_at
    from src.vision.geometry_audit import project,context_transform
    from src.vision.position_pipeline import PreparationError
    from scripts.run_square_evidence import make_inputs
    from src.utils.image_io import load_image
    contract=json.loads((ROOT/'execution_contract.json').read_text())
    for path,expected in contract['sha256'].items():
        if digest(path)!=expected:raise ValueError('Frozen C10 execution input changed: '+path)
    m=json.loads(MANIFEST.read_text());validate_manifest(m);cfg=json.loads(CONFIG.read_text())
    if digest(CONFIG)!=m['config_sha256'] or digest(cfg['settings_path'])!=cfg['settings_sha256']:raise ValueError('Frozen evaluation settings changed')
    torch.set_num_threads(cfg['cpu_threads']);torch.set_num_interop_threads(1);cv2.setNumThreads(1);torch.manual_seed(42)
    out=ROOT/(name+'_result.json')
    if out.exists():raise ValueError('Fresh model output required; no silent rerun')
    claim=ROOT/(name+'_inference_claim.json');write_new(claim,dict(manifest_sha256=digest(MANIFEST),model=name,classification_only=False))
    spec=cfg['models'][name];verify_checkpoint(spec['path'],spec['sha256']);t=time.perf_counter()
    state=torch.load(spec['path'],map_location='cpu',weights_only=True);verify_checkpoint(spec['path'],spec['sha256'],state,name)
    if name=='convnext':
        from src.training.trainer_c9 import fingerprint
        c9config=json.loads(Path('configs/training/convnext_c9_authorized_v1.json').read_text())
        if state['plan_hash']!=fingerprint(c9config):raise ValueError('C9 checkpoint configuration mismatch')
    model=make_model() if name=='convnext' else create_classifier(pretrained=False,mode='fine_tune')
    model.load_state_dict(state['model_state_dict'],strict=True);model.eval();model.requires_grad_(False);loading=time.perf_counter()-t
    del state
    def forward(x):
        scores=[];begin=time.perf_counter()
        with torch.inference_mode():
            for batch in x.split(cfg['batch_size']):
                z=model(batch)
                if z.shape!=(len(batch),13) or not torch.isfinite(z).all():raise ValueError('Invalid model outputs')
                scores.extend(z.tolist())
        return scores,time.perf_counter()-begin
    rows=[];tensors={};reference_logits={}
    for photo in m['photos']:
        row=dict(position_id=photo['id'],independent_eligible=photo['independence']['independent_eligible'],completed=False,failure_stage=None)
        begin=time.perf_counter()
        try:
            (rectified,grid,turns,prov),image,times=timed_preparation(Path(photo['path']),cfg['geometry'])
            original=photo['automatic_geometry']
            if not original['completed'] or turns!=original['rotation_quarters_ccw'] or not np.allclose(prov['source_corners'],original['returned_polygon'],atol=1e-5):raise ValueError('Replayed geometry differs from registered automatic output')
            t=time.perf_counter();inputs=canonical_inputs(make_inputs(image,original,cfg['crop']));packed=np.stack([p['packed'] for p in inputs]);times['crop_extraction_including_reference_context_warp']=time.perf_counter()-t
            if digest(photo['inputs']['path'])!=photo['inputs']['sha256']:raise ValueError('Input cache changed')
            if [hashlib.sha256(p.tobytes()).hexdigest() for p in packed]!=photo['inputs']['packed_sha256']:raise ValueError('Replayed crop/mask pixels differ')
            t=time.perf_counter();x=torch.from_numpy(np.stack([tensor_from_packed(p) for p in packed]));times['normalization']=time.perf_counter()-t
            z,seconds=forward(x);times['model_forward']=seconds;t=time.perf_counter();scored=score(placement_labels(photo['fen']),z);times['reconstruction_and_scoring']=time.perf_counter()-t
            # FEN creation is timed separately without ground-truth-driven repairs.
            t=time.perf_counter();placement=scored['predicted_placement'];full=placement+' '+' '.join(photo['fen'].split()[1:]);import chess;fen_valid=chess.Board(full).is_valid();times['fen_serialization_and_validation']=time.perf_counter()-t
            row.update(completed=True,**scored,predicted_full_fen_using_owner_game_state=full,predicted_chess_state_valid=fen_valid,timings=times,
                geometry_completed=True,geometry_owner_verified=False,semantic_orientation=original['predicted_camera_orientation'],packed_inputs_sha256=photo['inputs']['packed_sha256'])
            tensors[photo['id']]=x;reference_logits[photo['id']]=np.array(z)
        except PreparationError as error:row.update(failure_stage=error.stage,error=str(error))
        except (ValueError,OSError,RuntimeError) as error:row.update(failure_stage='input_or_runtime',error=f'{type(error).__name__}: {error}')
        row['end_to_end_after_model_load_seconds']=time.perf_counter()-begin;rows.append(row)
        print(name,photo['id'],row.get('metrics',{}).get('corrections'),row.get('failure_stage'),flush=True)
    # Identical-input repeats are timing only. They cannot reselect predictions.
    warm={pid:[] for pid in tensors}
    if tensors:
        first=next(iter(tensors.values()))
        for _ in range(cfg['warmup_passes']):forward(first)
        ids=list(tensors)
        for repeat in range(cfg['timing_rounds']):
            for pid in ids[repeat%len(ids):]+ids[:repeat%len(ids)]:
                z,seconds=forward(tensors[pid])
                if not np.array_equal(np.asarray(z),reference_logits[pid]):raise ValueError('Timing repeat changed logits')
                warm[pid].append(seconds)
    result=dict(schema_version='convnext_independent_c10_predictions_v1',model=name,checkpoint=spec,manifest_sha256=digest(MANIFEST),config_sha256=digest(CONFIG),
        rows=rows,aggregate=cohort_metrics(rows,3),model_loading_seconds=loading,warm_forward_seconds=warm,warm_median_forward_seconds={k:float(np.median(v)) for k,v in warm.items()},
        cold_first_image_plus_model_seconds=loading+rows[0]['end_to_end_after_model_load_seconds'],
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,torch_version=torch.__version__,opencv_version=cv2.__version__,
        cpu_model=next((v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name')),'unknown'),
        timing_scope='CPU FP32; batch16; two threads; preprocessing repeats use exact legacy functions; forward repeats only; model-loading excludes Python/library imports',
        paid_cloud=False,training_performed=False,secondary_verified_geometry_evaluation=None)
    write_new(out,result);verify_checkpoint(spec['path'],spec['sha256'])

def check():
    m=json.loads(MANIFEST.read_text());validate_manifest(m);expected=json.loads((ROOT/'protected_inputs.json').read_text())['protected_sha256'];mismatch=[]
    expected.update(json.loads((ROOT/'execution_contract.json').read_text())['sha256'])
    for p,sha in expected.items():
        if 'holdout' in p.lower():raise ValueError('Holdout access prohibited')
        if not Path(p).is_file() or digest(p)!=sha:mismatch.append(p)
    result=dict(checked=len(expected),mismatches=mismatch,original_photos_unchanged=True,final_holdout_access=False)
    write_new(ROOT/'integrity.json',result)
    if mismatch:raise ValueError('Historical input changed')
    print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['register','infer','check']);p.add_argument('--model',choices=list(CHECKPOINTS));a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='infer':
        if not a.model:p.error('--model required')
        infer(a.model)
    else:check()
