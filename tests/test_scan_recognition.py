import hashlib
import cv2
import numpy as np
import pytest
from src.scanning.recognition import prepare_keyframe,SessionRegistry,recognize_keyframe,VERSION
from src.vision.perspective import order_corners,warp_board_with_context
from src.vision.inference import PieceClassifier,predict_rectified
from src.scanning.bootstrap import semantic_mapping


def fixture(turns=0):
    image=np.random.default_rng(3).integers(0,256,(1000,1000,3),dtype=np.uint8)
    camera=np.float32([[250,250],[750,250],[750,750],[250,750]])
    corners=camera[[[3,2,1,0],[0,3,2,1],[1,0,3,2],[2,1,0,3]][turns]]
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),corners)
    polygons={}
    for r in range(8):
        for f in range(8):
            poly=np.float32([[f*4,r*4],[(f+1)*4,r*4],[(f+1)*4,(r+1)*4],[f*4,(r+1)*4]])
            polygons[f'{chr(97+f)}{r+1}']=cv2.perspectiveTransform(poly[:,None,:],h).reshape(4,2).tolist()
    return image,{'source':'tracking','version':1,'corners':corners.tolist(),'h_board_to_frame':h.ravel().tolist(),'square_polygons':polygons}

@pytest.mark.parametrize('turns',range(4))
def test_live_crops_are_identical_to_baseline_camera_relative_crops(turns):
    image,geometry=fixture(turns);crops=prepare_keyframe(image,geometry)
    rect=warp_board_with_context(image,order_corners(geometry['corners']),800,200)
    transform=cv2.getPerspectiveTransform(order_corners(geometry['corners']),np.float32([[200,200],[999,200],[999,999],[200,999]]))
    ids=semantic_mapping(turns);camera_polys=[None]*64
    for crop in crops:
        square=crop['chess_square'];canonical_row=8-int(square[1]);col=ord(square[0])-97
        camera_polys[int(ids[canonical_row,col])]=cv2.perspectiveTransform(np.float32(geometry['square_polygons'][square])[:,None,:],transform).reshape(4,2)
    class Reference:
        checkpoint_epoch=16
        def predict(self,reference):
            from src.vision.classification import SquarePrediction
            lookup={c['chess_square']:c for c in crops}
            from src.vision.dataset import chess_square_at
            for c in reference:
                actual=lookup[chess_square_at(c['row'],c['col'],turns)]
                np.testing.assert_array_equal(c['image'],actual['image'])
                np.testing.assert_allclose(c['target_polygon'],actual['target_polygon'],atol=0)
            return [SquarePrediction.from_probabilities(c['row'],c['col'],[1.]+[0.]*12) for c in reference]
    predict_rectified(Reference(),rect,{'fit_status':'fitted','square_polygons':camera_polys},turns)


def test_sessions_reject_stale_epoch_timestamp_and_geometry_version(monkeypatch):
    monkeypatch.setattr('src.scanning.recognition.model_metadata',lambda:{'test':True})
    registry=SessionRegistry();session=registry.create()
    meta={'schema_version':VERSION,'session_id':session['session_id'],'frame_id':'frame-1','board_epoch':1,'geometry_version':1,'orientation_version':1,'width':1000,'height':1000,'timestamp_ms':10.}
    registry.accept(meta)
    for changed in [{'board_epoch':0,'timestamp_ms':20,'geometry_version':2},{'timestamp_ms':10,'geometry_version':2},{'timestamp_ms':20,'geometry_version':1},{'timestamp_ms':float('nan')},{'session_id':'missing'}]:
        with pytest.raises(ValueError):registry.validate(meta|changed)


@pytest.mark.skipif(not __import__('pathlib').Path('models/checkpoints/piece_classifier_best.pt').exists(),reason='Local trained checkpoint required')
def test_real_checkpoint_keeps_all_logits_and_probability_parity():
    image,geometry=fixture();crops=prepare_keyframe(image,geometry);model=PieceClassifier('models/checkpoints/piece_classifier_best.pt')
    baseline=model.predict(crops);predictions,logits=model.predict_with_logits(crops)
    np.testing.assert_allclose([p.probabilities for p in baseline],[p.probabilities for p in predictions],atol=0)
    probabilities=np.exp(np.array(logits)-np.max(logits,axis=1,keepdims=True));probabilities/=probabilities.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(probabilities,[p.probabilities for p in predictions],atol=1e-7)
    _,encoded=cv2.imencode('.jpg',image);raw=encoded.tobytes()
    meta={'session_id':'test','frame_id':'one','board_epoch':0,'orientation_version':0,'geometry_version':1,'timestamp_ms':0,'source_sha256':hashlib.sha256(raw).hexdigest(),'width':1000,'height':1000,'geometry':geometry}
    result=recognize_keyframe(raw,meta,model)
    assert len(result['squares'])==64 and result['resolved_squares']==0 and not result['completion_enabled']
    with pytest.raises(ValueError,match='hash'):recognize_keyframe(raw,meta|{'source_sha256':'0'*64},model)
    with pytest.raises(ValueError,match='dimensions'):recognize_keyframe(raw,meta|{'width':999},model)


def test_http_routes_validate_session_metadata_and_reject_stale_frames(monkeypatch):
    from io import BytesIO
    import json
    import scripts.serve_scan_web as server
    monkeypatch.setattr('src.scanning.recognition.model_metadata',lambda:{'test':True})
    monkeypatch.setattr(server,'RECOGNITION_SESSIONS',SessionRegistry())
    monkeypatch.setattr(server,'recognize_keyframe',lambda raw,meta:{'session_id':meta['session_id'],'frame_id':meta['frame_id']})
    def post(path,raw,metadata=None,origin=None):
        handler=server.DevelopmentHandler.__new__(server.DevelopmentHandler)
        handler.path=path;handler.headers={'Content-Length':str(len(raw)),'Content-Type':'application/json' if path.endswith('session') else 'image/jpeg','Host':'local'}
        if metadata is not None:handler.headers['X-Scan-Observation']=json.dumps(metadata)
        if origin:handler.headers['Origin']=origin
        handler.rfile=BytesIO(raw);handler.wfile=BytesIO();codes=[]
        handler.send_response=lambda code:codes.append(code);handler.send_error=lambda code,*args:codes.append(code)
        handler.send_header=lambda *args:None;handler.end_headers=lambda:None
        handler.do_POST()
        return codes[-1],json.loads(handler.wfile.getvalue()) if handler.wfile.getvalue() else None
    code,session=post('/api/scan/session',b'{}');assert code==200
    metadata={'schema_version':VERSION,'session_id':session['session_id'],'frame_id':'one','board_epoch':0,'orientation_version':0,'geometry_version':1,'timestamp_ms':10,'width':100,'height':100}
    assert post('/api/scan/keyframe',b'JPEG',metadata)[0]==200
    assert post('/api/scan/keyframe',b'JPEG',metadata)[0]==422
    assert post('/api/scan/session',b'{}',origin='https://another-host')[0]==403


def test_session_capacity_recycles_oldest_without_blocking_new_scans(monkeypatch):
    monkeypatch.setattr('src.scanning.recognition.model_metadata', lambda: {'test': True})
    registry = SessionRegistry()
    first = registry.create()['session_id']
    registry.sessions[first]['touched'] -= 1
    for _ in range(32):
        newest = registry.create()['session_id']
    assert len(registry.sessions) == 32
    assert first not in registry.sessions
    assert newest in registry.sessions


def test_sessions_remain_available_during_analysis_and_board_requests_wait(monkeypatch):
    from io import BytesIO
    import json
    import threading
    import scripts.serve_scan_web as server
    monkeypatch.setattr('src.scanning.recognition.model_metadata', lambda: {'test': True})
    monkeypatch.setattr(server, 'RECOGNITION_SESSIONS', SessionRegistry())
    monkeypatch.setattr(server, 'BOOTSTRAP_LOCK', threading.Lock())
    monkeypatch.setattr(server, 'INFERENCE_SLOTS', threading.BoundedSemaphore(2))
    monkeypatch.setattr('src.scanning.bootstrap.bootstrap_image', lambda *args: {'status': 'candidate'})
    def handler(path, raw):
        h = server.DevelopmentHandler.__new__(server.DevelopmentHandler)
        h.path = path
        h.headers = {'Content-Length': str(len(raw)), 'Content-Type': 'application/json' if path.endswith('session') else 'image/jpeg', 'Host': 'local'}
        h.rfile = BytesIO(raw)
        h.wfile = BytesIO()
        h.codes = []
        h.send_response = lambda code: h.codes.append(code)
        h.send_error = lambda code, *args: h.codes.append(code)
        h.send_header = lambda *args: None
        h.end_headers = lambda: None
        return h
    server.BOOTSTRAP_LOCK.acquire()
    session = handler('/api/scan/session', b'{}')
    session.do_POST()
    assert session.codes == [200], 'Creating a session must not compete with board inference'
    jpeg = cv2.imencode('.jpg', np.zeros((32, 32, 3), np.uint8))[1].tobytes()
    board = handler('/api/bootstrap', jpeg)
    uploaded = threading.Event()
    class Upload(BytesIO):
        def read(self, count):
            uploaded.set()
            return super().read(count)
    board.rfile = Upload(jpeg)
    worker = threading.Thread(target=board.do_POST)
    worker.start()
    try:
        assert uploaded.wait(2), 'Upload must be read before waiting for inference'
        assert not board.codes, 'A busy analysis must queue rather than immediately reject'
    finally:
        server.BOOTSTRAP_LOCK.release()
        worker.join(3)
    assert not worker.is_alive()
    assert board.codes == [200]
    assert json.loads(board.wfile.getvalue())['status'] == 'candidate'
    assert not server.BOOTSTRAP_LOCK.locked()
