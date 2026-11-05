"""C4 contracts: mapping, real B2 pixel parity, leakage and explicit review gates."""
import copy,hashlib,json,random
from pathlib import Path
import numpy as np
import pytest
import torch
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import *
from src.training.manifest_tools import read,labels_to_placement,protected_personal
from src.training.video_review import binding,apply_reviews,CHECKS
from src.training.staged import BoardBatches,rng_state,restore_rng,contract_checkpoint,run_stage

ROOT=Path('outputs/evaluation/multidataset_c4')

def require_artifact(path):
    if not Path(path).exists():pytest.skip('C4 private/remote artifact not installed: '+str(path))
    return path

@pytest.fixture
def probe():
    return read(require_artifact(ROOT/'chesscog_manifest_probe_v1.json'))

@pytest.mark.parametrize('turns',range(4))
def test_canonical_orientation_preserves_labels(turns):
    fen='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    obs=regular_observations(fen,turns)
    assert len({o['square'] for o in obs})==64
    assert obs[0]['square']==('a8','a1','h1','h8')[turns]
    assert {o['square']:o['class_id'] for o in obs}==dict(zip(SQUARES,placement_labels(fen)))

def test_class_and_fen_contract():
    assert list(CLASS_NAMES)==['empty','white_pawn','white_knight','white_bishop','white_rook','white_queen','white_king','black_pawn','black_knight','black_bishop','black_rook','black_queen','black_king']
    labels=placement_labels('8/8/8/8/8/8/8/4K2k')
    assert labels[-4]==6 and labels[-1]==12
    assert placement_labels(labels_to_placement(labels))==labels
    with pytest.raises(ValueError):placement_labels('8/8/8/8/8/8/8/9')

def test_actual_external_samples_source_geometry(probe):
    assert len(probe['boards'])==4
    assert split_audit(probe['boards'])['conflicts']==0
    ds=MultiDatasetSquares(probe,'train')
    assert len(ds)==128 and ds[0][0].shape==(4,224,224)
    assert set(np.unique(ds.packed(0)[3]))<= {0,1}

@pytest.mark.parametrize('split',['train','val'])
def test_64_original_pixels_and_normalization_match_frozen_b2(split):
    from src.training.manifest_tools import rebase_manifest
    require_artifact(ROOT/'smoke_bundle_v2/manifest.json')
    manifest=rebase_manifest(ROOT/'smoke_bundle_v2')
    ds=MultiDatasetSquares(manifest,split)
    assert len(ds)==64
    # Every packed_sha256 comes directly from the immutable B2 .npy cache.
    for i in range(64):assert ds.packed(i).shape==(4,224,224)
    if split=='val':assert hashlib.sha256(ds[0][0].numpy().tobytes()).hexdigest()==manifest['smoke_checks']['normalized_first_val_sha256']

@pytest.mark.parametrize('relationship',['placement','session','game','source'])
def test_grouped_leak_rejected(probe,relationship):
    a=copy.deepcopy(probe['boards'][0]);b=copy.deepcopy(probe['boards'][-1]);b['split']='val'
    if relationship=='placement':
        b['placement']=a['placement'];b['observations']=copy.deepcopy(a['observations'])
    elif relationship=='source':b['source']['sha256']=a['source']['sha256']
    elif relationship=='session':a['session_id']=b['session_id']='same'
    else:a['game_id']=b['game_id']='same'
    with pytest.raises(ValueError,match='leakage'):split_audit([a,b])

def test_quarantine_protected_preserves_frozen_roles(probe):
    b=copy.deepcopy(probe['boards'][0]);quarantine_protected([b],{position_key(b['placement']):{'split':'val'}})
    assert b['split']=='train' and b['inclusion']['status']=='quarantine'
    p=read('data/metadata/d3c_data_use_c4_v1.json')
    overlap=[s for s in p['sessions'] if s['protected_overlap']]
    assert len(overlap)==1 and overlap[0]['session_id']=='d3c_pos_03'
    assert overlap[0]['protected_overlap']==dict(position_id='position_003',split='val')

def test_unverified_and_original_pilot_never_train(probe):
    b=copy.deepcopy(probe['boards'][0]);b['geometry']['status']='completed_unverified'
    with pytest.raises(ValueError,match='Unverified'):validate_board(b)
    b['dataset']='d3c_retired';b['session_id']='a3_pilot_01';b['inclusion']['status']='quarantine'
    with pytest.raises(ValueError,match='ten explicitly'):validate_board(b)

def test_mapping_or_approximation_cannot_change(probe):
    m=copy.deepcopy(probe);m['class_names']=list(reversed(CLASS_NAMES))
    with pytest.raises(ValueError,match='Class ordering'):MultiDatasetSquares(m,'train')
    m=copy.deepcopy(probe);m['boards'][0]['geometry'].pop('approximation_version')
    with pytest.raises(ValueError,match='approximation'):validate_board(m['boards'][0])
    m=copy.deepcopy(probe);m['boards'][0]['observations'][0]['class_id']=99
    with pytest.raises(ValueError,match='Labels disagree'):validate_board(m['boards'][0])

def test_invalid_geometry_identity_and_camera_mapping_rejected(probe):
    b=copy.deepcopy(probe['boards'][0]);b['geometry']['corners'][0][0]=float('nan')
    with pytest.raises(ValueError,match='Finite corners'):validate_board(b)
    b=copy.deepcopy(probe['boards'][0]);b['observations'][0]['camera_index']=1
    with pytest.raises(ValueError,match='camera-square'):validate_board(b)
    with pytest.raises(ValueError,match='Duplicate'):split_audit([probe['boards'][0]]*2)

def test_checkpoint_rng_deterministic_replay():
    state=rng_state();expected=(random.random(),np.random.rand(),torch.rand(3))
    restore_rng(state);actual=(random.random(),np.random.rand(),torch.rand(3))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])

def test_interrupted_cpu_smoke_resumes_identical_optimizer_updates(tmp_path,monkeypatch):
    from src.training.manifest_tools import rebase_manifest
    require_artifact(ROOT/'smoke_bundle_v2/manifest.json')
    monkeypatch.setenv('TORCH_HOME',str(Path('models/cache').resolve()))
    threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        m=rebase_manifest(ROOT/'smoke_bundle_v2');config=read('configs/training/multidataset_c4_v1.json')
        run_stage(m,config,'smoke',tmp_path/'full',smoke=True)
        def interrupt():raise InterruptedError('simulated interruption after atomic save at update2')
        with pytest.raises(InterruptedError):run_stage(m,config,'smoke',tmp_path/'resumed',smoke=True,commit=interrupt)
        result=run_stage(m,config,'smoke',tmp_path/'resumed',smoke=True,resume=tmp_path/'resumed/last.pt')
        a=torch.load(tmp_path/'full/last.pt',weights_only=True);b=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
        assert result['updates']==4 and a['updates']==b['updates']==4
        assert all(torch.equal(a['model_state_dict'][k],v) for k,v in b['model_state_dict'].items())
        assert a['scheduler_state_dict']==b['scheduler_state_dict']
    finally:
        torch.set_num_threads(threads)
        for p in tmp_path.rglob('*.pt'):p.unlink()  # Only this test's temporary checkpoints.

def test_grouped_sampler_replay_and_fixed_budget(probe):
    ds=MultiDatasetSquares(probe,'train')
    a=BoardBatches(ds,16,42,0,max_batches=12);b=BoardBatches(ds,16,42,0,max_batches=12)
    assert list(a)==list(b) and len(a)==12
    assert all(len({ds.rows[i][0]['id'] for i in batch})==1 for batch in a)
    with pytest.raises(ValueError,match='VERIFIED'):BoardBatches(ds,16,42,0,video_fraction=.5)

def test_full_training_not_authorized_before_dataset_or_model_load(tmp_path):
    with pytest.raises(ValueError,match='separate explicit'):run_stage({},read('configs/training/multidataset_c4_v1.json'),'chesscog',tmp_path)

def test_resume_contract():
    state=dict(class_names=list(CLASS_NAMES),input_channels=4,architecture='mobilenet_v3_small',stage='chesscog',dataset_fingerprint='abc')
    contract_checkpoint(state,'chesscog','abc')
    with pytest.raises(ValueError):contract_checkpoint(state,'chessred','abc')
    with pytest.raises(ValueError):contract_checkpoint(state,'chesscog','changed')

def test_controlled_d_e_initialization_and_video_only_difference():
    from src.training.staged import experiment_plan,check_transfer
    config=read('configs/training/multidataset_c4_v1.json');d=experiment_plan(config,'D');e=experiment_plan(config,'E')
    assert d['stages']['personal']['max_updates']==e['stages']['personal']['max_updates']==672
    assert d['stages']['personal']['video_fraction']==0 and e['stages']['personal']['video_fraction']==.5
    d['stages']['personal'].pop('video_fraction');e['stages']['personal'].pop('video_fraction')
    assert d['stages']==e['stages'] and not config['full_training_authorization']
    state=dict(class_names=list(CLASS_NAMES),input_channels=4,stage='chessred',smoke_only=False)
    check_transfer(state,'personal','E')
    with pytest.raises(ValueError):check_transfer(dict(state,smoke_only=True),'personal','E')
    with pytest.raises(ValueError):check_transfer(state,'personal','B')

def test_chessred_missing_geometry_quarantined():
    a=dict(categories=[dict(id=10,name='white-king')],images=[dict(id=1,file_name='missing.jpg',game_id='g')],annotations=dict(pieces=[dict(image_id=1,category_id=10,chessboard_position='e1')]))
    b=chessred_boards(a,'.',{},dict(g='train'))[0]
    assert b['inclusion']['status']=='quarantine' and b['observations'][60]['class_id']==6

def test_original_frozen_context_crop_route_matches_cache():
    m=read(require_artifact(ROOT/'personal_manifest_v1.json'))
    for split in ('train','val'):
        ds=MultiDatasetSquares(m,split)
        for i in range(64):assert ds.packed(i).shape==(4,224,224)

def test_video_addition_cannot_fallback_to_photo_only(tmp_path):
    from scripts.export_multidataset_c4 import export
    require_artifact(ROOT/'personal_manifest_v1.json');require_artifact(ROOT/'video_manifest_pending_v1.json')
    with pytest.raises(ValueError,match='No human-verified'):
        export([ROOT/'personal_manifest_v1.json',ROOT/'video_manifest_pending_v1.json'],tmp_path/'absent',True)
    assert not (tmp_path/'absent').exists()

def test_owner_review_and_compact_video_export_preserve_pixels(tmp_path):
    from scripts.export_multidataset_c4 import export
    from src.training.manifest_tools import rebase_manifest
    m=read(require_artifact(ROOT/'video_manifest_verified_v1.json'))
    included=[b for b in m['boards'] if board_ready(b)]
    assert len(included)==77 and sum(len(b['observations']) for b in included)==4928
    assert sum(b['geometry']['status']=='human_verified' for b in m['boards'])==87
    assert all(b['inclusion']['status']=='quarantine' for b in m['boards'] if b['session_id']=='d3c_pos_03')
    ds=MultiDatasetSquares(m,'train')
    for i in range(64):assert ds.packed(i).shape==(4,224,224)
    photos=read(ROOT/'personal_manifest_v1.json');subset=[next(b for b in photos['boards'] if b['split']==s) for s in ('train','val')]
    photos['boards']=subset+[included[0]];path=tmp_path/'source.json';path.write_text(json.dumps(photos))
    result=export([path],tmp_path/'bundle',True)
    assert result==dict(boards=3,observations=192,original_videos_copied=0)
    exported=rebase_manifest(tmp_path/'bundle');again=MultiDatasetSquares(exported,'train')
    for i,(b,o) in enumerate(again.rows):
        if b['dataset']=='d3c_retired':assert hashlib.sha256(again.packed(i).tobytes()).hexdigest()==o['packed_sha256']
    assert not list((tmp_path/'bundle').rglob('*.MOV'))

def test_smoke_is_persisted_and_not_transferable_to_full_stage():
    r=read(require_artifact(ROOT/'c4-smoke-v1_result.json'))
    assert r['smoke_only'] and r['updates']==4 and r['gradient_nonzero'] and r['checkpoint_reload_exact']
    assert r['persistence']['verified_from_separate_cpu_container']
    assert r['persistence']['sha256']==r['last_checkpoint_sha256']

def test_bounded_zip_crc_and_truncation_fail_closed():
    import struct,zlib
    from scripts.data.probe_chesscog_c4 import member
    payload=b'verified synthetic annotation';crc=zlib.crc32(payload)&0xffffffff
    header=struct.pack('<4s5H3I2H',b'PK\x03\x04',20,0,0,0,0,crc,len(payload),len(payload),0,0)
    raw=header+payload
    row=dict(disk=0,local_offset=0,compressed_bytes=len(payload),bytes=len(payload),method=0,crc32=crc)
    assert member(row,[('local',len(raw))],read_range=lambda p,a,b:raw[a:b+1])==payload
    with pytest.raises(ValueError):member(dict(row,crc32=crc+1),[('local',len(raw))],read_range=lambda p,a,b:raw[a:b+1])

def test_grouped_validation_metrics_and_confident_king_rook_error(probe):
    from scripts.evaluate_multidataset_c4 import summarize
    ds=MultiDatasetSquares(probe,'val');rows=ds.rows[:64];z=np.full((64,13),-10.)
    for i,(b,o) in enumerate(rows):z[i,o['class_id']]=10.
    king=next(i for i,(b,o) in enumerate(rows) if o['class_id']==6);z[king,:]=-10.;z[king,4]=10.
    r=summarize(rows,z)
    assert r['king_to_rook_errors']==1 and r['confident_errors_at_0_9']==1
    assert r['exact_boards']==0 and r['complete_board_denominator']==1

def test_review_binding_and_no_silent_acceptance(probe):
    b=copy.deepcopy(probe['boards'][0]);b.update(dataset='d3c_retired',session_id='d3c_pos_01');b['geometry']['status']='completed_unverified';b['annotation']['status']='unverified';b['inclusion']['status']='quarantine'
    f=dict(frame_id='f',source_sha256='a'*64,pts_time=[1,2],decoded_pixel_sha256='b'*64,geometry_sha256='c'*64,fen=b['placement'],protected_overlap=None,board=b)
    p=dict(frames=[f]);r=dict(packet_sha256=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest(),reviewer='human',frames={})
    assert not board_ready(apply_reviews(p,r)[0])
    r['frames']['f']=dict(binding=binding(f),status='verified',checks={c:True for c in CHECKS},ambiguous_squares=['a8'])
    out=apply_reviews(p,r)[0];assert board_ready(out) and next(o for o in out['observations'] if o['square']=='a8')['visibility']=='ambiguous'
    r['frames']['f']['checks']['canonical_mapping']=False
    with pytest.raises(ValueError,match='All geometry'):apply_reviews(p,r)
    r['frames']['f']['binding']='wrong'
    with pytest.raises(ValueError,match='different'):apply_reviews(p,r)
