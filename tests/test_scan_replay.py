import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
import pytest
from src.scanning.replay import replay_session


def archive(tmp_path, *, duplicate=False, geometry=True, tamper=False, path='frames/frame_0001.jpg'):
    image=np.zeros((400,400,3),np.uint8)
    cv2.rectangle(image,(40,40),(360,360),(170,190,220),-1)
    raw=cv2.imencode('.jpg',image)[1].tobytes()
    frame=dict(frame_id='frame_0001',path=path,timestamp_ms=100,board_epoch=0,width=400,height=400,source_sha256=hashlib.sha256(raw).hexdigest(),view_group='overhead',source='camera',visibility='unknown',geometry={'source':'manual','version':1,'corners':[[40,360],[360,360],[360,40],[40,40]],'h_board_to_frame':[10,0,40,0,-10,360,0,0,1]} if geometry else None)
    frames=[frame]
    if duplicate:frames.append(frame|{'frame_id':'frame_0002','path':'frames/frame_0002.jpg','timestamp_ms':200})
    manifest=dict(schema_version='guided_capture_v1',session_id='scan_fixture',board={'square_cm':4,'border_cm':3.5},metadata={'fen':'8/8/8/8/8/8/8/8 w - - 0 1'},frames=frames,events=[])
    target=tmp_path/'scan.zip'
    with ZipFile(target,'w') as z:
        z.writestr('manifest.json',json.dumps(manifest))
        for f in frames:z.writestr(f['path'],raw+b'tampered' if tamper else raw)
    return target


def test_replay_is_deterministic_and_preserves_all_square_ids(tmp_path):
    source=archive(tmp_path)
    one=replay_session(source,tmp_path/'output')
    assert one==replay_session(source)
    squares=one['frames'][0]['square_polygons']
    assert len(squares)==64
    assert squares['a1']==[[40.,360.],[80.,360.],[80.,320.],[40.,320.]]
    assert squares['h8']==[[320.,80.],[360.,80.],[360.,40.],[320.,40.]]
    assert one['resolved_squares']==0 and not one['automatic_completion']
    assert (tmp_path/'output/frame_0001_overlay.jpg').exists()


def test_duplicate_image_does_not_become_independent_evidence(tmp_path):
    result=replay_session(archive(tmp_path,duplicate=True))
    assert 'duplicate_source_in_epoch' in result['frames'][1]['reasons']
    assert not any(f['eligible_for_recognition_evidence'] for f in result['frames'])


def test_unaligned_frames_retained_without_stale_geometry(tmp_path):
    frame=replay_session(archive(tmp_path,geometry=False))['frames'][0]
    assert frame['square_polygons'] is None
    assert 'geometry_uninitialized' in frame['reasons']


def test_tampered_pixels_rejected(tmp_path):
    with pytest.raises(ValueError,match='Hash mismatch'):replay_session(archive(tmp_path,tamper=True))


def test_unsafe_archive_path_rejected(tmp_path):
    with pytest.raises(ValueError,match='Unsafe'):replay_session(archive(tmp_path,path='../escape.jpg'))


def test_browser_zip_can_be_read_by_python(tmp_path):
    import subprocess
    node=Path('tools/node/bin/node')
    if not node.exists():pytest.skip('Local Node runtime not installed')
    script="import {sessionZip} from './web/src/recording.ts'; const zip=sessionZip([{name:'manifest.json',data:new TextEncoder().encode('hello')}]);process.stdout.write(Buffer.from(await zip.arrayBuffer()));"
    result=subprocess.run([str(node),'--experimental-strip-types','--input-type=module','-e',script],capture_output=True,check=True)
    with ZipFile(io.BytesIO(result.stdout)) as zipped:
        assert zipped.read('manifest.json')==b'hello'
        assert zipped.testzip() is None
