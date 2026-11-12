"""MVP photo/confirmation contracts; actual model replay is recorded separately."""
from io import BytesIO
from pathlib import Path
import json
import threading
import uuid
import cv2
import numpy as np
import pytest
from PIL import Image,ImageOps
from src.scanning.assisted_photo import PhotoError,PhotoService,decode_photo,manual_geometry,confirm_position
from src.chess.fen import board_from_fen,full_fen,GameState

FEN='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

def payload(fen=FEN):
    board=board_from_fen(fen)
    return dict(user_confirmed=True,revision=3,recognition_id=str(uuid.uuid4()),
                board=[s for row in board.rows for s in row],
                game_state=dict(side_to_move='w',castling='KQkq',en_passant='-',halfmove_clock=0,fullmove_number=1))

def encoded(fmt='JPEG',orientation=1):
    a=np.zeros((80,112,3),np.uint8);a[:40,:56]=(200,30,10);a[40:,:56]=(10,210,40);a[:,56:]=(30,70,220)
    out=BytesIO();exif=Image.Exif();exif[274]=orientation
    Image.fromarray(a).save(out,format=fmt,exif=exif)
    return out.getvalue()

@pytest.mark.parametrize('orientation',range(1,9))
@pytest.mark.parametrize('fmt,mime',[('JPEG','image/jpeg'),('PNG','image/png')])
def test_decoding_orientation_matches_independent_pillow_display(orientation,fmt,mime):
    raw=encoded(fmt,orientation);image,info=decode_photo(raw,mime)
    with Image.open(BytesIO(raw)) as source:
        expected=np.asarray(ImageOps.exif_transpose(source).convert('RGB'))[:,:,::-1]
    np.testing.assert_array_equal(image,expected)
    assert [info['height'],info['width']]==list(image.shape[:2])
    assert info['exif_orientation']==orientation

@pytest.mark.parametrize('raw,mime',[(b'','image/jpeg'),(b'garbage','image/jpeg'),(b'<svg/>','image/svg+xml'),(encoded(),'image/png')])
def test_invalid_image_inputs_fail_closed(raw,mime):
    with pytest.raises(PhotoError):decode_photo(raw,mime)

def test_pixel_and_byte_limits_are_checked_before_cv_decode(monkeypatch):
    raw=encoded();monkeypatch.setattr('src.scanning.assisted_photo.MAX_PIXELS',100)
    monkeypatch.setattr(cv2,'imdecode',lambda *args:pytest.fail('Oversized image must not be decoded'))
    with pytest.raises(PhotoError,match='32 megapixels'):decode_photo(raw,'image/jpeg')
    monkeypatch.setattr('src.scanning.assisted_photo.MAX_UPLOAD',2)
    with pytest.raises(PhotoError) as caught:decode_photo(raw,'image/jpeg')
    assert caught.value.status==413

def test_manual_geometry_separates_image_corners_and_semantic_side():
    points=[[0,0],[100,0],[100,100],[0,100]]
    p,side=manual_geometry({'corners':points,'white_side':'left'},120,120)
    assert side=='left' and p.tolist()==points
    for bad in ({'corners':points[:3]},{'white_side':'guess'},{'fen':FEN},[],{'corners':[[0,0],[0,0],[1,1],[2,2]]}):
        with pytest.raises(PhotoError):manual_geometry(bad,120,120)

def test_runtime_is_explicit_cached_and_never_falls_back(monkeypatch):
    instances=[]
    def fake(config):instances.append(config);return object()
    monkeypatch.setattr('src.scanning.assisted_photo.load_runtime',fake)
    service=PhotoService();a=service.runtime();assert service.runtime() is a
    assert len(instances)==1
    assert instances[0]['backend']=='pytorch_channels_last' and instances[0]['batch_size']==8 and instances[0]['threads']==4
    assert service.info()['calibration']=='uncalibrated'
    diagnostic=PhotoService(reference=True);diagnostic.runtime();assert instances[-1]['backend']=='pytorch_reference'
    def failed(config):raise RuntimeError('bad checkpoint')
    monkeypatch.setattr('src.scanning.assisted_photo.load_runtime',failed)
    with pytest.raises(RuntimeError,match='bad checkpoint'):PhotoService().runtime()

def test_concurrent_initialization_loads_only_one_runtime(monkeypatch):
    import time
    calls=[]
    def load(config):calls.append(1);time.sleep(.02);return object()
    monkeypatch.setattr('src.scanning.assisted_photo.load_runtime',load)
    service=PhotoService();results=[];threads=[threading.Thread(target=lambda:results.append(service.runtime())) for _ in range(4)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert len(calls)==1 and all(r is results[0] for r in results)

def test_geometry_failure_never_calls_classifier_and_temp_is_removed(monkeypatch):
    from src.vision.position_pipeline import PreparationError
    paths=[]
    def failed(path,*args,**kwargs):paths.append(path);assert path.exists();raise PreparationError('orientation','uncertain')
    monkeypatch.setattr('src.scanning.assisted_photo.prepare_raw_photo',failed)
    service=PhotoService();monkeypatch.setattr(service,'runtime',lambda:pytest.fail('No geometry, no classification'))
    with pytest.raises(PhotoError) as caught:service.recognize(encoded(),'image/jpeg')
    assert caught.value.code=='orientation' and not paths[0].exists()
    assert caught.value.details['geometry_status']=='unresolved'

def test_confirmation_is_explicit_deterministic_and_roundtrips():
    p=payload();result=confirm_position(p)
    assert result['full_fen']==FEN and result['revision']==3 and result['user_confirmed']
    assert result['schema_version']=='confirmed_position_v1'
    assert full_fen(board_from_fen(result['full_fen']),GameState(**p['game_state']))==FEN
    assert result['game_state_inferred_from_image'] is False
    p['user_confirmed']=False
    with pytest.raises(PhotoError,match='explicitly confirm'):confirm_position(p)

@pytest.mark.parametrize('change', ['king','pawn_rank','excess','symbol','castling','ep','clock','missing_state'])
def test_confirmation_rejects_structural_or_game_state_errors(change):
    p=payload()
    if change=='king':p['board'][60]=None
    elif change=='pawn_rank':p['board'][32]=None;p['board'][0]='P'
    elif change=='excess':p['board'][32]='Q'
    elif change=='symbol':p['board'][32]='unrecognized'
    elif change=='castling':p['board'][63]=None
    elif change=='ep':p['game_state']['en_passant']='a3'
    elif change=='clock':p['game_state']['fullmove_number']=0
    else:del p['game_state']['side_to_move']
    with pytest.raises(PhotoError):confirm_position(p)

def test_promoted_extra_queen_is_not_blanket_rejected():
    p=payload('4k3/8/8/8/8/8/QQ6/4K3 w - - 0 1');p['game_state']['castling']='-'
    assert confirm_position(p)['validation']['standard_chess_valid']

def test_photo_endpoint_disabled_by_default_and_bounded_queue(monkeypatch):
    import scripts.serve_scan_web as server
    from types import SimpleNamespace
    def handler(service=None,raw=b'x',origin=None):
        h=server.DevelopmentHandler.__new__(server.DevelopmentHandler);h.path='/api/photo/recognize'
        h.server=SimpleNamespace(photo_service=service);h.headers={'Host':'localhost','Content-Length':str(len(raw)),'Content-Type':'image/jpeg'}
        if origin:h.headers['Origin']=origin
        h.rfile=BytesIO(raw);h.connection=SimpleNamespace(settimeout=lambda n:None);h.results=[]
        h.send_json=lambda body,status=200:h.results.append((status,body));return h
    h=handler();h.do_POST();assert h.results[0][0]==503
    h=handler(object(),origin='https://elsewhere');h.do_POST();assert h.results[0][0]==403
    monkeypatch.setattr(server,'INFERENCE_SLOTS',threading.BoundedSemaphore(0))
    h=handler(object());h.do_POST();assert h.results[0][0]==429 and h.rfile.tell()==0

def test_photo_endpoint_does_not_retain_uploaded_body_or_return_success_on_error(monkeypatch):
    import scripts.serve_scan_web as server
    from types import SimpleNamespace
    class Service:
        def recognize(self,*args):raise PhotoError('localization','Board not found')
    monkeypatch.setattr(server,'INFERENCE_SLOTS',threading.BoundedSemaphore(2))
    h=server.DevelopmentHandler.__new__(server.DevelopmentHandler);h.path='/api/photo/recognize'
    h.server=SimpleNamespace(photo_service=Service());h.headers={'Content-Type':'image/jpeg','Content-Length':'1','Host':'localhost'}
    h.rfile=BytesIO(b'x');h.connection=SimpleNamespace(settimeout=lambda n:None);h.results=[];h.send_json=lambda body,status=200:h.results.append((status,body))
    h.do_POST();assert h.results==[(422,{'status':'error','code':'localization','message':'Board not found'})]
    assert server.INFERENCE_SLOTS.acquire(blocking=False)
    assert not hasattr(h.server.photo_service,'images')

@pytest.mark.parametrize('orientation',[1,6,8])
def test_mpo_jpeg_primary_is_accepted_without_reencoding(orientation):
    output=BytesIO();exif=Image.Exif();exif[274]=orientation
    primary=Image.new('RGB',(112,80),(180,30,20));secondary=Image.new('RGB',(112,80),(20,180,30))
    primary.save(output,format='MPO',save_all=True,append_images=[secondary],exif=exif)
    raw=output.getvalue()
    with Image.open(BytesIO(raw)) as header:
        assert header.format=='MPO' and header.n_frames==2
        expected=np.asarray(ImageOps.exif_transpose(header).convert('RGB'))[:,:,::-1]
    actual,info=decode_photo(raw,'image/jpeg')
    np.testing.assert_array_equal(actual,expected)
    assert info['exif_orientation']==orientation
    with pytest.raises(PhotoError):decode_photo(raw,'image/png')
