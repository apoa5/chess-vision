"""C.4 personal/external manifests, immutable policy supplement and smoke bundle."""
import hashlib,json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import PROFILE,SQUARES,position_key,validate_board,split_audit,digest


def read(path):return json.loads(Path(path).read_text())
def rebase_manifest(root):
    root=Path(root);m=read(root/'manifest.json')
    for b in m['boards']:
        b['source']['path']=str(root/b['source']['path'])
        if 'valid_mask_path' in b['geometry']:b['geometry']['valid_mask_path']=str(root/b['geometry']['valid_mask_path'])
    if (root/'checks.json').is_file():m['smoke_checks']=read(root/'checks.json')
    return m

def write_new(path,value):
    path=Path(path)
    if path.exists():raise ValueError('Refuse overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def labels_to_placement(labels):
    symbols='.PNBRQKpnbrqk';ranks=[]
    for offset in range(0,64,8):
        text='';n=0
        for k in labels[offset:offset+8]:
            if k==0:n+=1
            else:
                if n:text+=str(n);n=0
                text+=symbols[k]
        if n:text+=str(n)
        ranks.append(text)
    return '/'.join(ranks)


def personal_manifest():
    from src.vision.dataset import review_approved
    metadata=read('data/classification/crops.json');prepared={b['image_id']:b for b in read('data/classification/prepared_boards.json')}
    reviews=read('data/metadata/day9_board_reviews.json');images=read('data/metadata/images.json');by_source={}
    for r in metadata['results']:by_source.setdefault(r['source_image'],[]).append(r)
    caches={s:np.load('outputs/evaluation/crop_b2_20261018_run1/cache/context1_224/'+s+'.npy',mmap_mode='r') for s in ('train','val')};indices=Counter()
    boards=[]
    for source,rows in by_source.items():
        p=prepared[source]
        if not review_approved(p,reviews.get(source)):raise ValueError('Historical approval changed')
        split=rows[0]['split'];observations=[]
        for r in rows:
            o=dict(square=r['chess_square'],class_id=r['class_id'],crop_path=r['crop_path'],crop_sha256=r['crop_sha256'],
                   target_polygon=r['target_polygon'],crop_id=r['crop_id'],camera_index=r['row']*8+r['col'],visibility='historically_unreviewed')
            if split in caches:o['packed_sha256']=hashlib.sha256(caches[split][indices[split]].tobytes()).hexdigest();indices[split]+=1
            observations.append(o)
        b=dict(id='personal:'+source,dataset='chessvision_photos',source_id=source,position_id=p['position_id'],session_id=images[source].get('capture_session_id'),game_id=None,
               source=dict(path=images[source]['path'],filename=source,sha256=p['source_sha256']),placement=p['fen'].split()[0],original_fen=p['fen'],
               semantic_orientation=images[source]['orientation'],split=split,representation=PROFILE,geometry=dict(status='historical_owner_verified',kind='frozen_context_crop',review_signature=p['review_signature']),
               annotation=dict(status='historical_owner_verified',provenance='unchanged day9 owner board geometry/label approval; individual visibility not universally reviewed'),
               inclusion=dict(status='included',reason='approved_original_split_preserved'),observations=observations)
        boards.append(validate_board(b))
    split_audit(boards)
    return dict(schema_version='multidataset_c4_v1',representation=PROFILE,class_names=list(CLASS_NAMES),final_holdout=False,boards=boards,
                provenance=dict(manifest='data/classification/crops.json',manifest_sha256=digest('data/classification/crops.json')))


def protected_personal():
    p=read('data/metadata/positions.json');assignment=read('data/classification/crops.json')['split_report']['position_assignment']
    return {position_key(v['fen']):dict(position_id=k,split=assignment[k]) for k,v in p.items() if assignment[k] in ('val','test')}


def retire_d3c():
    old=read('outputs/scanning/d3c_interim_v1/registration.json');protected=protected_personal();sessions=[]
    for s in old['sessions']:
        source=s['source']
        if digest(source['path'])!=source['sha256']:raise ValueError('Original D3c video changed')
        overlap=protected.get(position_key(s['fen']))
        sessions.append(dict(session_id=s['session_id'],position_id=s['position_id'],source=source,fen=s['fen'],
            data_use='retired_independent_evaluation_authorized_development_training_pending_verification',
            training_eligible=False,verification='pending_frame_FEN_orientation_geometry_crop_review',protected_overlap=overlap,
            inclusion='quarantine_protected_position' if overlap else 'pending_verification',
            reference_photos=s['reference_photos'],unchanged_position=s['unchanged_position']))
    return dict(schema_version='d3c_data_use_c4_v1',authority='Explicit owner authorization in C.4 assignment',
        historical_report=dict(path='docs/evaluation/fusion_independent_d3c.md',sha256=digest('docs/evaluation/fusion_independent_d3c.md')),
        historical_registry=dict(path='data/metadata/fusion_independent_d3c_cohort_v1.json',sha256=digest('data/metadata/fusion_independent_d3c_cohort_v1.json')),
        original_pilot_excluded=True,original_historical_artifacts_modified=False,independent_evaluation_status='retired',
        future_evaluation='Only genuinely new unseen recording sessions after training/model selection; not required during C4',sessions=sessions)


def smoke_bundle(personal,output):
    """128 approved same-profile observations; no full photos/videos uploaded."""
    from src.training.multidataset import MultiDatasetSquares
    output=Path(output)
    if output.exists():raise ValueError('Fresh smoke bundle required')
    output.mkdir(parents=True);boards=[]
    prepared={b['image_id']:b for b in read('data/classification/prepared_boards.json')}
    for split in ('train','val'):
        b=next(b for b in personal['boards'] if b['split']==split);b=json.loads(json.dumps(b));
        original=prepared[b['source_id']];b['original_source']=dict(b['source'])
        context=Path('contexts')/Path(original['context_path']).name;valid=Path('contexts')/Path(original['mask_path']).name
        (output/'contexts').mkdir(exist_ok=True)
        shutil.copyfile(original['context_path'],output/context);shutil.copyfile(original['mask_path'],output/valid)
        b['source']=dict(path=str(context),filename=context.name,sha256=original['context_sha256'])
        b['geometry']=dict(status='historical_owner_verified',kind='verified_prepared_context',valid_mask_path=str(valid),
                           context_square_polygons=original['grid_geometry']['square_polygons'],review_signature=original['review_signature'])
        boards.append(b)
    manifest=dict(schema_version='multidataset_c4_v1',representation=PROFILE,class_names=list(CLASS_NAMES),final_holdout=False,boards=boards)
    write_new(output/'manifest.json',manifest)
    # Expected normalized values bind NumPy/OpenCV differences in the cloud.
    for b in boards:
        b['source']['path']=str(output/b['source']['path'])
        b['geometry']['valid_mask_path']=str(output/b['geometry']['valid_mask_path'])
    ds=MultiDatasetSquares(dict(manifest,boards=boards),'val')
    first=ds[0][0].numpy();write_new(output/'checks.json',dict(normalized_first_val_sha256=hashlib.sha256(first.tobytes()).hexdigest(),expected_train=64,expected_val=64,
        stage='smoke_only_not_chesscog_pretraining',source='original approved photographs; D3c not verified and not uploaded'))
    initial=Path('models/cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth');dest=output/'torch/hub/checkpoints'/initial.name;dest.parent.mkdir(parents=True);shutil.copyfile(initial,dest)
    write_new(output/'bundle_inventory.json',{str(p.relative_to(output)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()})
    return dict(boards=2,observations=128,bytes=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),raw_videos_uploaded=0)
