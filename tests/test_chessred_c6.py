"""C6 transport/adapter contracts. Fixtures are not real ChessReD annotations."""
import hashlib,json,urllib.error
from pathlib import Path
import cv2,numpy as np,pytest,torch
from torch.utils.data import DataLoader
from src.chess.labels import CLASS_NAMES
from src.training import chessred_access_c6 as access
from src.training.multidataset import PROFILE,chessred_boards,MultiDatasetSquares,board_ready

class Response:
    status=200
    def __init__(self,body=b'',headers=None):self.body=body;self.headers=headers or {};self.read_calls=0
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def read(self,n):
        self.read_calls+=1;b=self.body[:n];self.body=self.body[n:];return b

def test_archive_head_never_reads_body(tmp_path):
    response=Response(b'not-read',{'Content-Length':str(100_000_000_000)})
    r=access.bounded_fetch('image_archive_metadata',access.IMAGES_URL,'HEAD',0,tmp_path,opener=lambda *a,**k:response)
    assert r['success'] and response.read_calls==0 and not list(tmp_path.iterdir())

@pytest.mark.parametrize('query',['','?download=1'])
def test_image_archive_get_forbidden_even_with_query(tmp_path,query):
    with pytest.raises(ValueError,match='forbidden'):access.bounded_fetch('images',access.IMAGES_URL+query,'GET',96*1024*1024,tmp_path)

def test_advertised_oversize_refused_before_body(tmp_path):
    response=Response(b'never-read',{'Content-Length':'100'})
    r=access.bounded_fetch('metadata',access.ANNOTATION_URL,'GET',10,tmp_path,opener=lambda *a,**k:response)
    assert not r['success'] and response.read_calls==0 and len(r['attempts'])==1

def test_streaming_oversize_refused_without_retry(tmp_path):
    r=access.bounded_fetch('metadata',access.ANNOTATION_URL,'GET',3,tmp_path,opener=lambda *a,**k:Response(b'too long'))
    assert not r['success'] and len(r['attempts'])==1 and not (tmp_path/'metadata.response').exists()

def test_transport_attempts_bounded_and_http404_not_retried(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(access.time,'sleep',lambda _:None)
    def timeout(*a,**k):calls.append(1);raise urllib.error.URLError(TimeoutError('timeout'))
    r=access.bounded_fetch('metadata',access.ANNOTATION_URL,'GET',10,tmp_path,opener=timeout)
    assert not r['success'] and len(calls)==len(r['attempts'])==2
    calls.clear()
    def absent(*a,**k):calls.append(1);raise urllib.error.HTTPError(access.ANNOTATION_URL,404,'not found',{},None)
    r=access.bounded_fetch('metadata',access.ANNOTATION_URL,'GET',10,tmp_path,opener=absent)
    assert not r['success'] and len(calls)==1

def test_annotation_checksum_verified_or_quarantined(tmp_path,monkeypatch):
    body=json.dumps(dict(images=[],categories=[],annotations={},splits={})).encode()
    monkeypatch.setattr(access,'ANNOTATION_MD5',hashlib.md5(body).hexdigest())
    r=access.bounded_fetch('annotations',access.ANNOTATION_URL,'GET',1000,tmp_path,opener=lambda *a,**k:Response(body))
    assert r['success'] and r['attempts'][0]['published_md5_verified']
    other=tmp_path/'bad';other.mkdir();monkeypatch.setattr(access,'ANNOTATION_MD5','0'*32)
    r=access.bounded_fetch('annotations',access.ANNOTATION_URL,'GET',1000,other,opener=lambda *a,**k:Response(body))
    assert not r['success'] and not r['attempts'][0]['published_md5_verified']

def normalized_fixture(tmp_path):
    # Deliberately synthetic RGB array, not a photograph or verified real datum.
    cv2.imwrite(str(tmp_path/'fixture.png'),np.full((64,64,3),120,np.uint8))
    a=dict(images=[dict(id=7,path='fixture.png',game_id='g1')],categories=[dict(id=101,name='white-king'),dict(id=0,name='black-rook'),dict(id=12,name='empty')],
           annotations=dict(pieces=[dict(image_id=7,category_id=101,chessboard_position='e1'),dict(image_id=7,category_id=0,chessboard_position='h8')]),splits=dict(val=dict(image_ids=[7],n_samples=1)))
    geometry={'7':dict(status='source_ground_truth',kind='raw_regular_grid',corners=[[4,4],[60,4],[60,60],[4,60]],turns=0,orientation='white_near_camera',approximation_version='b2-annotated-corners-regular-grid-v1')}
    return a,geometry

def test_documented_boundary_maps_names_not_raw_category_indices(tmp_path):
    a,g=normalized_fixture(tmp_path);b=chessred_boards(a,tmp_path,g,{'g1':'val'})[0]
    labels={o['square']:o['class_id'] for o in b['observations']}
    assert board_ready(b) and labels['e1']==6 and labels['h8']==10 and labels['a1']==0
    m=dict(representation=PROFILE,class_names=list(CLASS_NAMES),boards=[b],final_holdout=False)
    ds=MultiDatasetSquares(m,'val');x,y=next(iter(DataLoader(ds,batch_size=4)))
    assert len(ds)==64 and x.shape==(4,4,224,224) and torch.isfinite(x).all()
    assert set(np.unique(ds.packed(0)[3]))<={0,1} and ds.packed(0)[3].any()

@pytest.mark.parametrize('missing',['geometry','game','role','verified_status'])
def test_incomplete_metadata_never_becomes_training_eligible(tmp_path,missing):
    a,g=normalized_fixture(tmp_path);games={'g1':'val'}
    if missing=='geometry':g={}
    elif missing=='game':a['images'][0].pop('game_id')
    elif missing=='role':games={'g1':'train'}
    else:g['7']['status']='completed_unverified'
    b=chessred_boards(a,tmp_path,g,games)[0]
    assert not board_ready(b) and b['inclusion']['status']=='quarantine'
    if missing=='geometry':assert 'corners' not in b['geometry']

def test_duplicate_or_unknown_square_annotation_rejected(tmp_path):
    a,g=normalized_fixture(tmp_path);a['annotations']['pieces'].append(dict(a['annotations']['pieces'][0]))
    with pytest.raises(ValueError,match='duplicate'):chessred_boards(a,tmp_path,g,{'g1':'val'})
