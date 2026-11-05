"""C7 direct-parent, equal-budget, source exclusion and provenance contracts."""
import copy
import json
from pathlib import Path
import pytest
from src.chess.labels import CLASS_NAMES
from src.training.personal_c7 import branch_plan, branch_manifest, validate_parent, PARENT_SHA
from src.training.staged import check_transfer

@pytest.fixture
def plan():
    return json.loads(Path('configs/training/personal_c7_authorized_v1.json').read_text())

def test_branches_matched_except_treatment_and_names(plan):
    a=branch_plan(plan,'B'); b=branch_plan(plan,'C')
    assert a['parent_sha256']==b['parent_sha256']==PARENT_SHA
    assert a['experiment']==b['experiment']=='B'
    for p in (a,b):
        p.pop('c7_branch'); p['stages']['personal'].pop('video_fraction');p['stages']['personal'].pop('best_name');p['stages']['personal'].pop('required_dataset')
    assert a==b

@pytest.mark.parametrize('key,value',[('max_updates',673),('epochs',7),('batch_size',32),('max_seconds',1801),('validation_every_updates',56)])
def test_budget_mutation_rejected(plan,key,value):
    plan['stages']['personal'][key]=value
    with pytest.raises(ValueError):branch_plan(plan,'C')

def test_authorization_parent_and_representation(plan):
    for key,value in [('full_training_authorization',False),('parent_sha256','0'*64),('representation','other')]:
        p=copy.deepcopy(plan);p[key]=value
        with pytest.raises(ValueError):branch_plan(p,'B')

def test_old_chessred_contract_not_weakened():
    state=dict(class_names=list(CLASS_NAMES),input_channels=4,stage='chesscog',smoke_only=False)
    check_transfer(state,'personal','B')
    for experiment in ('D','E','C'):
        with pytest.raises(ValueError):check_transfer(state,'personal',experiment)

def test_exact_parent_hash_before_model_acceptance(monkeypatch,tmp_path):
    state=dict(class_names=list(CLASS_NAMES),input_channels=4,input_size=224,architecture='mobilenet_v3_small',stage='chesscog',smoke_only=False)
    monkeypatch.setattr('src.training.personal_c7.digest',lambda _:PARENT_SHA)
    validate_parent(tmp_path,state)
    with pytest.raises(ValueError):validate_parent(tmp_path,dict(state,stage='chessred'))
    monkeypatch.setattr('src.training.personal_c7.digest',lambda _:'0'*64)
    with pytest.raises(ValueError):validate_parent(tmp_path,state)

def test_real_manifest_counts_and_video_quarantine():
    root=Path('outputs/evaluation/multidataset_c4')
    if not (root/'video_manifest_verified_v1.json').exists():pytest.skip('Private approved data unavailable')
    from src.training.multidataset import board_ready
    m=json.loads((root/'personal_manifest_v1.json').read_text());v=json.loads((root/'video_manifest_verified_v1.json').read_text())
    m['boards']=[b for b in m['boards']+v['boards'] if board_ready(b) and b['split'] in ('train','val')]
    assert len(branch_manifest(m,'B')['boards'])==137
    assert len(branch_manifest(m,'C')['boards'])==214
    bad=copy.deepcopy(m); bad['boards'][0]['split']='test'
    with pytest.raises(ValueError):branch_manifest(bad,'B')
    video=next(b for b in m['boards'] if b['dataset']=='d3c_retired');video['session_id']='d3c_pos_03'
    with pytest.raises(ValueError):branch_manifest(m,'C')


def test_small_upload_batches_publish_inventory_last(tmp_path):
    from src.training.personal_c7 import bundle_batches
    for name in ['manifest.json','bundle_inventory.json']+[str(i)+'.png' for i in range(9)]:
        (tmp_path/name).write_text('fixture')
    batches=bundle_batches(tmp_path)
    assert [len(b) for b in batches]==[4,4,1,2]
    assert [p.name for p in batches[-1]]==['manifest.json','bundle_inventory.json']
    (tmp_path/'original.MOV').write_text('not a video')
    with pytest.raises(ValueError):bundle_batches(tmp_path)

def test_planned_exposure_is_deterministic_and_budget_matched(plan):
    from src.training.personal_c7 import exposure_report
    root=Path('outputs/evaluation/multidataset_c4')
    if not (root/'video_manifest_verified_v1.json').exists():pytest.skip('Private approved data unavailable')
    from src.training.multidataset import board_ready
    m=json.loads((root/'personal_manifest_v1.json').read_text());v=json.loads((root/'video_manifest_verified_v1.json').read_text())
    m['boards']=[b for b in m['boards']+v['boards'] if board_ready(b) and b['split'] in ('train','val')]
    a=exposure_report(branch_manifest(m,'B'),branch_plan(plan,'B'))
    b=exposure_report(branch_manifest(m,'C'),branch_plan(plan,'C'))
    assert a==exposure_report(branch_manifest(m,'B'),branch_plan(plan,'B'))
    assert sum(a['planned_class_exposure'])==sum(b['planned_class_exposure'])==672*64
    assert a['planned_observations_by_source']=={'chessvision_photos':672*64}
    assert .4 < b['planned_observations_by_source']['d3c_retired']/(672*64) < .6
    assert not any('pos_03' in key for key in b['planned_observations_by_board'])
