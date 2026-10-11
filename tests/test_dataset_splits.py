import json
import numpy as np
import pytest
from src.vision.dataset_splits import split_dataset
from src.vision.dataset_loader import ContextSquareDataset
from src.utils.image_io import save_image


def manifest():
    rows=[]
    for p in range(9):
        for cls in range(13):
            rows.append(dict(crop_id=f'{p}_{cls}',source_image=f'{p}.jpg',position_id=f'p{p}',class_id=cls,
                             geometry_review_required=False,quality_flags=[]))
    return {'results':rows,'preview_only':False}


def test_position_splits_are_reproducible_and_cover_classes():
    m=manifest();a,report=split_dataset(m,trials=30);b,_=split_dataset(m,trials=30)
    assert a==b
    assert all(not missing for missing in report['missing_classes'].values())
    for key in ['position_id','source_image']:
        groups={s:{r[key] for r in a['results'] if r['split']==s} for s in ('train','val','test')}
        assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    assert a['training_ready']


def test_duplicate_positions_and_photos_share_a_split():
    m=manifest();fens={f'p{i}':f'placement{i} w - - 0 1' for i in range(9)};fens['p1']=fens['p0']
    hashes={f'{i}.jpg':str(i) for i in range(9)};hashes['2.jpg']=hashes['1.jpg']
    _,r=split_dataset(m,position_fens=fens,source_hashes=hashes,trials=30)
    assert r['position_assignment']['p0']==r['position_assignment']['p1']==r['position_assignment']['p2']


def test_training_weights_use_only_training_counts():
    m=manifest();a,r=split_dataset(m,trials=30)
    count=np.array(list(r['splits']['train']['class_counts'].values()),float)
    expected=np.sqrt(count.sum()/count);expected/=expected.mean()
    assert np.allclose(list(r['class_weights'].values()),expected)
    assert m['results'][0].get('split') is None


@pytest.mark.parametrize('problem',['preview','unapproved','flags','empty'])
def test_unsafe_inputs_rejected(problem):
    m=manifest()
    if problem=='preview':m['preview_only']=True
    elif problem=='unapproved':m['results'][0]['geometry_review_required']=True
    elif problem=='flags':m['results'][0]['quality_flags']=['bad']
    else:m['results']=[]
    with pytest.raises(ValueError):split_dataset(m,trials=5)


def loader_manifest(tmp_path):
    image=np.full((80,80,3),128,np.uint8);image[20:40,20:40]=[30,60,90]
    path=save_image(tmp_path/'image.png',image)
    rows=[dict(crop_id=s,crop_path=str(path),position_id=s,source_image=s,class_id=2,
               target_polygon=[[20,20],[39,20],[39,39],[20,39]],split=s,geometry_review_required=False,quality_flags=[])
          for s in ('train','val','test')]
    manifest_path=tmp_path/'manifest.json';manifest_path.write_text(json.dumps({'results':rows,'split_status':'position_grouped','preview_only':False}))
    return manifest_path


def test_train_augmentation_changes_rgb_but_preserves_target_mask(tmp_path):
    path=loader_manifest(tmp_path)
    train=ContextSquareDataset(path,'train',input_size=80,tensor=False)
    first,label=train[0];assert first.shape==(4,80,80) and first.dtype==np.float32 and label==2
    assert np.array_equal(first,train[0][0])
    train.set_epoch(1);second,_=train[0]
    assert not np.array_equal(first[:3],second[:3])
    assert np.array_equal(first[3],second[3])
    assert first[3,20:40,20:40].sum()==400 and first[3].sum()==400
    val=ContextSquareDataset(path,'val',input_size=80,tensor=False);a,_=val[0];val.set_epoch(8)
    assert np.array_equal(a,val[0][0])
    assert set(np.unique(a[3]))=={0,1}


def test_real_tensor_batch_contract(tmp_path):
    torch=pytest.importorskip('torch')
    path=loader_manifest(tmp_path);dataset=ContextSquareDataset(path,'test',input_size=32)
    batch,labels=next(iter(torch.utils.data.DataLoader(dataset,batch_size=1)))
    assert batch.shape==(1,4,32,32) and batch.dtype==torch.float32
    assert labels.tolist()==[2]


def test_loader_rejects_position_leakage(tmp_path):
    path=loader_manifest(tmp_path);m=json.loads(path.read_text());m['results'][1]['position_id']='train';path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='leaks'):ContextSquareDataset(path,'train')


def test_calibration_positions_are_kept_in_training():
    _,report=split_dataset(manifest(),trials=50,train_only_positions=['p0'])
    assert report['position_assignment']['p0']=='train'
    assert report['train_only_positions']==['p0']


def test_dataset_fingerprint_changes_when_crop_pixels_change():
    m=manifest();_,a=split_dataset(m,trials=10)
    m['results'][0]['crop_sha256']='new-pixels';_,b=split_dataset(m,trials=10)
    assert a['dataset_fingerprint']!=b['dataset_fingerprint']
    assert a['position_assignment']==b['position_assignment']
