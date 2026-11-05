"""Freeze C9 runtime inputs and one checksum-bound transfer archive; no original media."""
import copy,hashlib,json,sys,tarfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.manifest_tools import read,write_new
from src.training.multidataset import digest
from src.training.convnext_c9 import MODEL_NAME,WEIGHT_SHA
from src.chess.labels import CLASS_NAMES
ROOT=Path('outputs/evaluation/convnext_c9')

def main():
    source=ROOT/'approved_cache_with_reviews_v1.json';m=read(source);mapping={};newroot='/data/convnext-c9-data-v1'
    if (ROOT/'frozen_manifest_v1.json').exists():raise ValueError('C9 dataset already frozen')
    for b in m['boards']:
        b['local_source_path']=b['source']['path'];b['local_mask_path']=b['geometry']['valid_mask_path']
        for path,kind in [(b['local_source_path'],'image'),(b['local_mask_path'],'mask')]:
            original=Path(path);sha=digest(original)
            if kind=='image' and sha!=b['source']['sha256']:raise ValueError('Approved context changed')
            if b.get('reuse_remote_root'):
                remote='/data/'+b['reuse_remote_root']+'/contexts/'+original.name
            else:
                rel='contexts/'+sha+'.png';remote=newroot+'/'+rel
                mapping.setdefault(rel,dict(local_path=str(original),sha256=sha,bytes=original.stat().st_size))
            if kind=='image':b['source']['path']=remote
            else:b['geometry']['valid_mask_path']=remote
    m.update(schema_version='convnext_c9_frozen_v1',approved_cache_sha256=digest(source),internal_test_once=True,final_holdout=False,
        authority='data/metadata/convnext_c9_data_use_v1.json',new_reviews='data/metadata/convnext_c9_reviews_v1.json')
    write_new(ROOT/'frozen_manifest_v1.json',m)
    baseline=Path('outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt')
    if digest(baseline)!='dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8':raise ValueError('B2 reference changed')
    mapping['b2_reference.pt']=dict(local_path=str(baseline),sha256=digest(baseline),bytes=baseline.stat().st_size)
    mapping['manifest.json']=dict(local_path=str(ROOT/'frozen_manifest_v1.json'),sha256=digest(ROOT/'frozen_manifest_v1.json'),bytes=(ROOT/'frozen_manifest_v1.json').stat().st_size)
    write_new(ROOT/'file_inventory_v1.json',mapping)
    archive=ROOT/'transfer_v1.tar.gz'
    if archive.exists():raise ValueError('Refuse archive overwrite')
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for rel,item in mapping.items():tar.add(item['local_path'],arcname=rel,recursive=False)
        tar.add(ROOT/'file_inventory_v1.json',arcname='file_inventory.json',recursive=False)
    receipt=dict(archive_sha256=digest(archive),archive_bytes=archive.stat().st_size,unpacked_bytes=sum(i['bytes'] for i in mapping.values()),
        files=len(mapping)+1,manifest_sha256=digest(ROOT/'frozen_manifest_v1.json'),inventory_sha256=digest(ROOT/'file_inventory_v1.json'),
        remote_root=newroot,original_videos_in_archive=0,shared_C7_contexts_reused=True)
    write_new(ROOT/'package_v1.json',receipt)
    config=dict(schema_version='convnext_training_c9_v1',architecture=MODEL_NAME,representation=m['representation'],class_names=list(CLASS_NAMES),seed=42,
        pretrained_sha256=WEIGHT_SHA,license='cc-by-nc-4.0; noncommercial research only',training_authorized=True,budget_usd=5,
        epochs=12,head_epochs=1,max_updates=20000,max_seconds=9000,patience=4,batch_size=32,gradient_accumulation=1,workers=2,
        checkpoint_every_updates=200,precision='bf16',head_stem_lr=dict(backbone=0.,stem=1e-4,head=1e-3),full_lr=dict(backbone=3e-5,stem=1e-4,head=3e-4),weight_decay=.05,
        scheduler='head adaptation then cosine to0.1 over full-finetune epochs',selection='minimum macro-physical-position class-weighted validation CE',
        sampler='full photo and video coverage every cycle; repeat complete photo passes if required so video share<=50%; class/position weighted CE',
        augmentation=read('configs/training/multidataset_c4_v1.json')['augmentation'],dataset_root=newroot,manifest_sha256=receipt['manifest_sha256'],
        split_manifest_sha256=digest('data/metadata/convnext_c9_split_v2.json'),photo_video_review_sha256=digest('data/metadata/convnext_c9_reviews_v1.json'),
        gpu='L4',cpu_cores=2,memory_mib=8192,function_timeout_seconds=10800,startup_timeout_seconds=300,automatic_retries=0,
        baseline_checkpoint_sha256=digest(baseline),final_holdout_access=False)
    write_new('configs/training/convnext_c9_authorized_v1.json',config)
    registry=dict(schema_version='convnext_c9_dataset_registry_v1',manifest=str(ROOT/'frozen_manifest_v1.json'),manifest_sha256=receipt['manifest_sha256'],
        groups=m['group_audit'],split_manifest='data/metadata/convnext_c9_split_v2.json',data_use='data/metadata/convnext_c9_data_use_v1.json',reviews='data/metadata/convnext_c9_reviews_v1.json',
        boards=[{k:b.get(k) for k in ['id','source_id','position_id','physical_position_id','session_id','dataset','original_fen','ground_truth_fen','label_version','semantic_orientation','split','original_source','original_frame','b2_training_placement_exposed','b2_exact_training_image_exposed']} for b in m['boards']],
        redundancy_exclusions=m['redundancy_exclusions'],unresolved_square='A3 endgame at0s h4 retained as ambiguous; not used for training/evaluation',independent_generalization=False)
    write_new('data/metadata/convnext_c9_dataset_v1.json',registry);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
