"""Re-use reviewed B2 contexts; produce C9-only approved-source cache without relabeling history."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.training.manifest_tools import read,write_new
from src.training.multidataset import digest,PROFILE,SQUARES,placement_labels,regular_observations,position_key
from src.training.video_review import apply_reviews
from src.training.dataset_c9 import validate_board,audit_splits
from src.scanning.square_evidence import verify_reviews
from src.scanning.interim_video import decode_exact,pixel_hash
from src.vision.perspective import warp_board_with_context
from src.vision.geometry_audit import project,context_transform
from scripts.review_convnext_c9 import signature,CHECKS
ROOT=Path('outputs/evaluation/convnext_c9')

def context_board(b,image,g):
    directory=ROOT/'contexts_v1';directory.mkdir(exist_ok=True);key=hashlib.sha256(b['id'].encode()).hexdigest()[:24];ip=directory/(key+'.png');mp=directory/(key+'_valid.png')
    context=warp_board_with_context(image,np.float32(g['provenance']['source_corners']),800,200)
    polygons=project(g['source_square_polygons'],context_transform(g['provenance']['source_corners'],context.board_bounds)).tolist()
    if ip.exists() or mp.exists():raise ValueError('Fresh context paths required')
    cv2.imwrite(str(ip),context.image);cv2.imwrite(str(mp),context.valid_mask)
    b['original_source']=copy.deepcopy(b['source']);b['source']=dict(path=str(ip),filename=ip.name,sha256=digest(ip));b['geometry']=dict(kind='verified_prepared_context',status='human_verified',valid_mask_path=str(mp),context_square_polygons=polygons)
    return b

def canonicalize(b,assignment,photo_training_placements,photo_training_ids):
    b['historical_split']=b.get('split');b['split']=assignment[position_key(b['placement'])];b['representation']=PROFILE
    if b['dataset']=='d3c_retired' and b['inclusion']['status']=='quarantine':b['historical_inclusion']=b['inclusion'];b['inclusion']=dict(status='included',reason='C9 authorized regrouping; explicit C4 review remains valid')
    b['b2_training_placement_exposed']=position_key(b['placement']) in photo_training_placements
    b['b2_exact_training_image_exposed']=b['dataset']=='chessvision_photos' and b['source_id'] in photo_training_ids
    validate_board(b);return b

def from_video(session,row,g,review,expected=None,source_image=None):
    camera={sq:i for i,sq in enumerate(g['semantic_square_ids'])};obs=regular_observations(session['fen'],0)
    for o in obs:
        o['camera_index']=camera[o['square']];o['visibility']='unknown_occlusion_reviewed_label_alignment'
        if expected is not None:o['packed_sha256']=expected[o['square']]
    if session['dataset']=='a3_pilot' and '04_endgame' in session['id'] and row['timestamp_ms']==0:
        next(o for o in obs if o['square']=='h4')['visibility']='ambiguous'
    image=source_image if source_image is not None else decode_exact(session['source']['path'],row)
    if pixel_hash(image)!=row['decoded_pixel_sha256']:raise ValueError('Reviewed source pixels changed')
    b=dict(id=session['dataset']+':'+row['frame_id'],dataset=session['dataset'],source_id=row['frame_id'],position_id=session['position_id'],session_id=session['id'],independence_group_id=session['independence_group_id'],
        source=dict(path=session['source']['path'],sha256=session['source']['sha256'],filename=Path(session['source']['path']).name),placement=session['fen'].split()[0],original_fen=session['fen'],
        semantic_orientation=g['predicted_camera_orientation'],split='unassigned',geometry=dict(status='human_verified'),annotation=dict(status='human_verified',review=review),review_binding_verified=True,
        inclusion=dict(status='included',reason='explicit_attributed_review_and_C9_authorization'),observations=obs,original_frame={k:row[k] for k in ['frame_id','pts_time','timestamp_ms','decoded_pixel_sha256']})
    return context_board(b,image,g)

def main(reviews_path=None):
    output=ROOT/('approved_cache_with_reviews_v1.json' if reviews_path else 'approved_cache_prior_reviews_v1.json')
    if output.exists():raise ValueError('Fresh approved cache required')
    reuse_prior=bool(reviews_path and (ROOT/'approved_cache_prior_reviews_v1.json').exists())
    inventory=read(ROOT/'inventory_v1.json');sessions={s['id']:s for s in inventory['sessions']};assignment=read('data/metadata/convnext_c9_split_v2.json')['assignment_by_placement']
    photos=read('outputs/evaluation/multidataset_c4/personal_manifest_v1.json')['boards'];pmap={b['image_id']:b for b in read('data/classification/prepared_boards.json')};photo_train=[b for b in photos if b['split']=='train'];seen_places={position_key(b['placement']) for b in photo_train};seen_ids={b['source_id'] for b in photo_train}
    c7root=Path('outputs/evaluation/personal_c7/bundle_v1');c7={b['id']:b for b in read(c7root/'manifest.json')['boards']};boards=read(ROOT/'approved_cache_prior_reviews_v1.json')['boards'] if reuse_prior else []
    def add(b):boards.append(canonicalize(b,assignment,seen_places,seen_ids))
    for original in ([] if reuse_prior else photos):
        b=copy.deepcopy(c7.get(original['id'],original));b['historical_split']=original['split']
        if original['id'] in c7:
            b['source']['path']=str(c7root/b['source']['path']);b['geometry']['valid_mask_path']=str(c7root/b['geometry']['valid_mask_path']);b['reuse_remote_root']='personal-c7-bundle-v2'
        else:
            p=pmap[b['source_id']];b['original_source']=copy.deepcopy(b['source']);b['source']=dict(path=p['context_path'],sha256=p['context_sha256'],filename=Path(p['context_path']).name)
            b['geometry']=dict(kind='verified_prepared_context',status='historical_owner_verified',valid_mask_path=p['mask_path'],context_square_polygons=p['grid_geometry']['square_polygons'])
        add(b)
    packet=read('outputs/evaluation/multidataset_c4/video_review_v1/packet.json');reviews=read('data/metadata/c4_video_reviews_v1.json');video=apply_reviews(packet,reviews)
    if video!=read('outputs/evaluation/multidataset_c4/video_manifest_verified_v1.json')['boards']:raise ValueError('Original C4 review replay differs')
    for original in ([] if reuse_prior else video):
        if original['geometry']['status']!='human_verified' or original['annotation']['status']!='human_verified':continue
        sid=original['session_id'];b=copy.deepcopy(c7.get(original['id'],original));b['review_binding_verified']=True
        if original['id'] in c7:
            b['source']['path']=str(c7root/b['source']['path']);b['geometry']['valid_mask_path']=str(c7root/b['geometry']['valid_mask_path']);b['reuse_remote_root']='personal-c7-bundle-v2'
        else:
            g=original['geometry'];row=read(g['candidate_path'])
            if digest(g['candidate_path'])!=g['candidate_sha256']:raise ValueError('C4 reviewed geometry changed')
            image=decode_exact(original['source']['path'],row)
            if pixel_hash(image)!=row['decoded_pixel_sha256']:raise ValueError('Reviewed video pixels changed')
            b=context_board(b,image,row['geometry']);b['original_frame']={k:row[k] for k in ['frame_id','pts_time','timestamp_ms','decoded_pixel_sha256']}
        add(b)
    packet_path=Path('docs/evaluation/evidence_d2_geometry_review_v1/packet.json');packet=read(packet_path);reviews=read('data/metadata/evidence_d2_geometry_reviews_v1.json');statuses=verify_reviews(packet,reviews,digest(packet_path))
    evidence={f['frame_id']:f for f in read('data/metadata/evidence_d2_manifest_v1.json')['frames']}
    for item in ([] if reuse_prior else packet['frames']):
        if statuses[item['review_id']]['human_review']!='human_verified':continue
        f=item['frame'];sid=item['session']['capture_session_id'];g=read(f['geometry_reference'])['policies']['marked-board']
        if digest(f['geometry_reference'])!=f['geometry_reference_sha256'] or item['source_square_polygons']!=g['source_square_polygons']:raise ValueError('D2 reviewed geometry changed')
        if digest(f['path'])!=f['derivative_sha256']:raise ValueError('D2 reviewed frame changed')
        image=cv2.imread(f['path']);e=evidence[f['frame_id']]
        if digest(e['input_cache'])!=e['input_cache_sha256']:raise ValueError('D2 input cache changed')
        packed=np.load(e['input_cache'])['packed'];expected={sq:hashlib.sha256(p.tobytes()).hexdigest() for sq,p in zip(g['semantic_square_ids'],packed)}
        b=from_video(sessions[sid],f,g,dict(path='data/metadata/evidence_d2_geometry_reviews_v1.json',review_id=item['review_id']),expected,image);add(b)
    if reviews_path:
        packet=read(ROOT/'review_v1/packet.json');reviews=read(reviews_path)
        if reviews['packet_sha256']!=digest(ROOT/'review_v1/packet.json') or not reviews.get('reviewer'):raise ValueError('C9 review provenance invalid')
        existing={b['source_id'] for b in boards}
        for item in packet['items']:
            decision=reviews['items'].get(item['id'])
            if decision is None or decision['status']!='verified':continue
            if decision['binding']!=item['binding'] or not all(decision['checks'].get(c) is True for c in CHECKS):raise ValueError('Unverified review checks')
            if item['id'] in existing:continue
            raw=read(item['geometry_reference']);g=raw['geometry']
            if hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest()!=item['geometry_sha256']:raise ValueError('C9 reviewed geometry changed')
            if item['source_kind']=='photo':
                meta=raw['metadata'];image=cv2.imread(meta['source_path']);obs=regular_observations(item['fen'],0);camera={sq:i for i,sq in enumerate(g['semantic_square_ids'])}
                for o in obs:o.update(camera_index=camera[o['square']],packed_sha256=item['packed_sha256'][o['square']])
                b=dict(id=item['id'],dataset='chessvision_photos',source_id=meta['id'],position_id=meta['position_id'],session_id=meta['session_id'],source=dict(path=meta['source_path'],sha256=meta['sha256']),
                  placement=item['fen'].split()[0],original_fen=item['fen'],semantic_orientation=g['predicted_camera_orientation'],split='historically_excluded',annotation=dict(status='human_verified',review_path=str(reviews_path)),
                  inclusion=dict(status='included',reason='C9 explicit review'),observations=obs)
                if pixel_hash(image)!=item['decoded_pixel_sha256']:raise ValueError('Reviewed photo pixels changed')
                b=context_board(b,image,g)
            else:
                b=from_video(sessions[item['session_id']],raw,g,dict(path=str(reviews_path),id=item['id']),item['packed_sha256'])
            for o in b['observations']:
                if o['square'] in decision.get('ambiguous_squares',[]):o['visibility']='ambiguous'
            add(b)
    # Joint post-review exact/near-duplicate guard across old and new video views.
    # Labels and predictions never affect quality ranking or exclusions.
    from src.scanning.video_keyframes import near_duplicate
    settings=read('configs/experiments/keyframes_d1.json');features={}
    for folder in (ROOT/'candidates_v1').iterdir():
        for path in folder.glob('candidate_*.json'):
            row=read(path);features[row['frame_id']]=row
    retained=[];excluded=[];by_session={}
    for b in boards:
        if b['dataset']=='chessvision_photos':retained.append(b)
        else:by_session.setdefault(b['session_id'],[]).append(b)
    for sid,pool in by_session.items():
        chosen=[]
        for b in sorted(pool,key=lambda b:(-features[b['source_id']]['quality_score'],b['source_id'])):
            row=features[b['source_id']]
            duplicate=next((s for s in chosen if near_duplicate(row,features[s['source_id']],settings) or abs(row['timestamp_ms']-features[s['source_id']]['timestamp_ms'])<1500),None)
            if duplicate:excluded.append(dict(id=b['id'],source_id=b['source_id'],status='human_verified_excluded_redundancy',represented_by=duplicate['id'],reason='unchanged D1 appearance+view near-duplicate or less than1.5 seconds temporal spacing'))
            else:chosen.append(b)
        retained.extend(chosen)
    boards=retained
    for b in boards:
        b['physical_position_id']='placement_'+position_key(b['placement'])[:16]
        if b['dataset']=='a3_pilot':
            session=sessions[b['session_id']];b['ground_truth_fen']=session['fen'];b['original_catalog_fen']=session['original_fen'];b['label_version']=session['label_version']
    manifest=dict(schema_version='convnext_c9_approved_context_cache_v1',representation=PROFILE,class_names=read('outputs/evaluation/multidataset_c4/personal_manifest_v1.json')['class_names'],boards=boards,final_holdout=False,
        split_manifest='data/metadata/convnext_c9_split_v2.json',split_manifest_sha256=digest('data/metadata/convnext_c9_split_v2.json'),group_audit=audit_splits(boards),redundancy_exclusions=excluded,new_reviews_path=str(reviews_path) if reviews_path else None)
    # Check every accepted input once; no classifier or test predictions run.
    from src.training.dataset_c9 import C9Squares
    for split in ['train','val','internal_test']:
        ds=C9Squares(manifest,split,allow_internal_test=split=='internal_test')
        for i in range(len(ds)):ds.packed(i)
        print(split,len(ds.boards),'boards',len(ds),'eligible squares',flush=True)
    write_new(output,manifest)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--reviews',type=Path);a=p.parse_args();main(a.reviews)
