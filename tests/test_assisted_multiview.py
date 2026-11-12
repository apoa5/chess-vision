"""Bounded scan contracts; real-input parity and HTTP replay are separate artifacts."""
from types import SimpleNamespace
import hashlib,threading,time
import numpy as np
import pytest
from src.training.multidataset import SQUARES
from src.scanning.assisted_photo import PhotoError
from src.scanning.assisted_multiview import ScanManager,validate_geometry,MAX_ATTEMPTS,MAX_CANDIDATES,MAX_CANDIDATE_BYTES
from src.scanning.multiview_inference import STRATEGIES,MultiViewRunner,ScanCancelled,aggregate_views


def geometry(turns=0):
    p=[[[c*20,r*20],[(c+1)*20,r*20],[(c+1)*20,(r+1)*20],[c*20,(r+1)*20]] for r in range(8) for c in range(8)]
    return dict(completed=True,rotation_quarters_ccw=turns,source_square_polygons=p,semantic_square_ids=list(SQUARES),
                predicted_camera_orientation='white_near_camera',provenance={'source_corners':[[0,0],[160,0],[160,160],[0,160]],'orientation_source':'border_markings'})


def fake_prepared(raw,*args):
    n=raw[0] if raw else 0
    features=dict(quality_score=.8,quality_rejection_reasons=[],appearance_dhash=f'{n:016x}',view_descriptor=[n*.3,0,0,0])
    return dict(source={'width':200,'height':200},geometry=geometry(n%4),features=features,geometry_seconds=.1,decoded_pixel_sha256=hashlib.sha256(raw).hexdigest())


@pytest.fixture
def manager(monkeypatch):
    monkeypatch.setattr('src.scanning.assisted_multiview.prepare_candidate',fake_prepared)
    photo=SimpleNamespace(profile={},runtime=lambda:pytest.fail('Candidate collection must not classify'))
    manager=ScanManager(photo,{'strategy':'sequential','fusion_policy':'best_quality'})
    yield manager
    for sid in list(manager.sessions):manager.cancel(sid)


def create(manager):return manager.create({'unchanged_position':True})['session_id']


@pytest.mark.parametrize('turns',range(4))
def test_all_resolved_orientations_use_canonical_ids(turns):
    validate_geometry(geometry(turns),200,200)


@pytest.mark.parametrize('bad',['unresolved','shape','ids','outside','degenerate','nonfinite'])
def test_invalid_geometry_is_never_fused(bad):
    g=geometry()
    if bad=='unresolved':g['rotation_quarters_ccw']=None
    elif bad=='shape':g['source_square_polygons']=[]
    elif bad=='ids':g['semantic_square_ids'][1]='a8'
    elif bad=='outside':g['source_square_polygons'][0][0]=[-1,0]
    elif bad=='degenerate':g['source_square_polygons'][0]=[[0,0]]*4
    else:g['source_square_polygons'][0][0][0]=float('nan')
    with pytest.raises(PhotoError):validate_geometry(g,200,200)


def test_collection_is_bounded_and_prediction_blind(manager):
    sid=create(manager)
    for i in range(MAX_ATTEMPTS):
        manager.candidate(sid,str(i),i*3000,bytes([i]),'image/jpeg')
        assert len(manager.sessions[sid]['frames'])<=MAX_CANDIDATES
    assert manager.status(sid)['attempts']==MAX_ATTEMPTS
    with pytest.raises(PhotoError,match='limit reached'):manager.candidate(sid,'extra',74000,b'x','image/jpeg')


def test_identical_pixels_and_stale_candidates_are_not_new_evidence(manager):
    sid=create(manager);manager.candidate(sid,'a',0,b'view','image/jpeg')
    second=manager.candidate(sid,'b',5000,b'view','image/jpeg')
    assert second['status']=='redundant' and second['retained_ids']==['a']
    with pytest.raises(PhotoError,match='stale'):manager.candidate(sid,'a',6000,b'new','image/jpeg')
    with pytest.raises(PhotoError,match='stale'):manager.candidate(sid,'c',1000,b'new','image/jpeg')


def test_limits_attestation_expiry_and_cancel_release_candidates(manager):
    for bad in ({},{'unchanged_position':False},{'unchanged_position':1}):
        with pytest.raises(PhotoError):manager.create(bad)
    sid=create(manager)
    with pytest.raises(PhotoError) as error:manager.candidate(sid,'big',0,b'x'*(MAX_CANDIDATE_BYTES+1),'image/jpeg')
    assert error.value.status==413
    manager.candidate(sid,'one',0,b'x','image/jpeg');manager.cancel(sid)
    assert sid not in manager.sessions
    with pytest.raises(PhotoError):manager.status(sid)
    expired=create(manager);manager.sessions[expired]['created']-=601
    with pytest.raises(PhotoError):manager.status(expired)
    assert expired not in manager.sessions


def test_rejected_geometry_is_recorded_and_no_usable_finish_fails(manager,monkeypatch):
    def fail(*args):raise PhotoError('orientation','Unresolved orientation')
    monkeypatch.setattr('src.scanning.assisted_multiview.prepare_candidate',fail)
    sid=create(manager);result=manager.candidate(sid,'a',0,b'x','image/jpeg')
    assert result['status']=='rejected' and result['reason']=='orientation'
    assert manager.status(sid)['diagnostics'][0]['status']=='rejected'
    with pytest.raises(PhotoError,match='No usable frames'):manager.claim_finish(sid)


def test_finish_is_once_only_and_source_binding_cannot_cross_sessions(manager):
    sid=create(manager);manager.candidate(sid,'a',0,b'x','image/jpeg');manager.claim_finish(sid)
    with pytest.raises(PhotoError,match='already'):manager.claim_finish(sid)
    manager.sessions[sid]['frames']['a']['binding']['session_id']='another'
    with pytest.raises(PhotoError,match='not bound'):manager.process(sid)


@pytest.mark.parametrize('name',STRATEGIES)
def test_shared_model_batches_keep_view_square_order_and_fp32(name):
    import torch
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.active=0;self.maximum=0;self.guard=threading.Lock();self.batches=[]
        def forward(self,x):
            assert x.dtype==torch.float32 and x.is_contiguous(memory_format=torch.channels_last)
            with self.guard:self.active+=1;self.maximum=max(self.maximum,self.active);self.batches.append(len(x))
            time.sleep(.002)
            out=x[:,0,0,0,None].expand(-1,13).contiguous()
            with self.guard:self.active-=1
            return out
    model=Model().eval();runtime=SimpleNamespace(model=model,config={'backend':'pytorch_channels_last'},last_timings={'sentinel':1})
    runtime.predict_logits=lambda x:np.repeat(x[:,0,0,0,None],13,axis=1)
    arrays=[np.zeros((64,4,224,224),np.float32) for _ in range(3)]
    for i,x in enumerate(arrays):x[:,0,0,0]=np.arange(64)+100*i
    progress=[];result=MultiViewRunner(runtime).predict(arrays,name,progress=lambda a,b:progress.append((a,b)))
    assert result.shape==(3,64,13) and result.dtype==np.float32
    for i in range(3):np.testing.assert_array_equal(result[i,:,0],np.arange(64)+100*i)
    if name!='sequential':
        assert max(model.batches)<=STRATEGIES[name]['batch_size']
        assert model.maximum<=STRATEGIES[name]['workers']
        if STRATEGIES[name]['workers']>1:assert model.maximum>1
        assert runtime.last_timings=={'sentinel':1}
    assert progress[-1]==(192,192)


def test_cancelled_inference_never_runs_model_or_returns_draft():
    runtime=SimpleNamespace(config={'backend':'pytorch_channels_last'},predict_logits=lambda x:pytest.fail('Cancelled call'))
    with pytest.raises(ScanCancelled):MultiViewRunner(runtime).predict([np.zeros((64,4,224,224),np.float32)],'sequential',cancelled=lambda:True)


def test_fusion_definitions_and_quality_choice_are_not_oracle():
    from src.scanning.offline_fusion import softmax
    z=np.zeros((3,64,13),np.float32);z[0,:,1]=8;z[1,:,2]=4;z[2,:,2]=4
    single=aggregate_views(z,[.9,.8,.7],'best_quality');assert (single['predictions']==1).all()
    mean=aggregate_views(z,[.9,.8,.7],'mean_logits');assert (mean['predictions']==1).all() # 8/3 ties; lowest class
    probability=aggregate_views(z,[.9,.8,.7],'mean_probabilities')
    np.testing.assert_allclose(probability['scores'][0],np.mean([softmax(v[0]) for v in z],axis=0))
    np.testing.assert_allclose(mean['scores'][0],softmax(z[:,0].mean(0)))
    assert probability['disagreement'].all()
    with pytest.raises(ValueError):aggregate_views(z[:,:,:12],[1,1,1])

@pytest.mark.parametrize('budget',[0,4,True,'3',None])
def test_invalid_view_budget_does_not_claim_session(manager,budget):
    sid=create(manager)
    with pytest.raises(PhotoError):manager.claim_finish(sid,budget)
    assert manager.status(sid)['phase']=='capturing'


def test_empty_scan_cannot_create_draft(manager):
    sid=create(manager)
    with pytest.raises(PhotoError,match='No usable frames'):manager.claim_finish(sid)
    assert manager.status(sid)['phase']=='capturing'


def test_one_view_budget_is_frozen_before_worker_start(manager):
    sid=create(manager);manager.candidate(sid,'frame_001',0,b'\x01','image/jpeg')
    manager.claim_finish(sid,1)
    assert manager.sessions[sid]['view_budget']==1
    assert manager.status(sid)['phase']=='queued'
    with pytest.raises(PhotoError):manager.claim_finish(sid,3)


def test_video_source_extends_only_its_own_timestamp_range(manager):
    source=dict(kind='uploaded_video',filename='long.mp4',size_bytes=10000000,duration_ms=600000,width=1920,height=1080)
    sid=manager.create(dict(unchanged_position=True,source=source))['session_id']
    manager.candidate(sid,'video_midpoint',300000,b'\x01','image/jpeg')
    frame=manager.sessions[sid]['frames']['video_midpoint']
    assert frame['binding']['source']==source and frame['timestamp_ms']==300000
    with pytest.raises(PhotoError):manager.candidate(sid,'outside_source',600001,b'\x02','image/jpeg')
    live=create(manager)
    with pytest.raises(PhotoError):manager.candidate(live,'outside_live',300000,b'\x02','image/jpeg')


@pytest.mark.parametrize('field,value',[('kind','remote_url'),('size_bytes',2**31),('duration_ms',float('nan')),('duration_ms',1800001),('filename',''),('height',0),('width',True)])
def test_video_source_is_bounded_and_validated_before_session_creation(manager,field,value):
    source=dict(kind='uploaded_video',filename='x.mp4',size_bytes=1,duration_ms=10000,width=1920,height=1080)
    source[field]=value
    with pytest.raises(PhotoError):manager.create(dict(unchanged_position=True,source=source))
    assert not manager.sessions
