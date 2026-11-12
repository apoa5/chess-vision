"""One uploaded-video session API replay from a hash-checked approved frame.

This is a frame/API test, not a claim of browser decoding this particular MOV.
"""
import argparse,json,hashlib,time,sys
from pathlib import Path
from urllib.request import Request,urlopen
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path('outputs/evaluation/assisted_mvp16_v1')
def rpc(path,data=None,headers=None):
 if isinstance(data,dict):data=json.dumps(data).encode();headers={'Content-Type':'application/json'}
 with urlopen(Request('http://127.0.0.1:8765/api/photo/scan/'+path,data=data,headers=headers or {}),timeout=60) as r:return json.load(r)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'single_video_api_v1.json');output=parser.parse_args().output
 if output.exists():raise ValueError('No overwrite')
 previous=json.loads(Path('outputs/evaluation/assisted_mvp15_v1/application_replay_v1_budget1.json').read_text())
 source=json.loads(Path('outputs/evaluation/assisted_mvp15_v1/replay_source_v1.json').read_text());fid=previous['result']['multiview']['reference_frame_id'];f=next(f for f in source['frames'] if f['frame_id']==fid)
 raw=Path(f['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==f['derivative_sha256'];image=cv2.imread(f['path']);ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
 metadata=dict(kind='uploaded_video',filename='new_position_001_hd.MOV',size_bytes=83441135,duration_ms=75410,width=1080,height=1920)
 sid=rpc('session',dict(unchanged_position=True,source=metadata))['session_id']
 try:
  c=rpc('candidate',jpeg.tobytes(),{'Content-Type':'image/jpeg','X-Scan-Session':sid,'X-Scan-Frame':fid,'X-Scan-Timestamp':str(f['relative_timestamp_ms'])});assert c['status']=='retained'
  start=time.perf_counter();rpc('finish',{'session_id':sid})
  while True:
   x=rpc('status?session_id='+sid)
   if x['phase']=='failed':raise RuntimeError(x['error'])
   if x['phase']=='review_ready':break
   if time.perf_counter()-start>120:raise TimeoutError('Bounded fallback replay')
   time.sleep(.25)
  r=x['result'];assert r['multiview']['accepted_view_count']==1;assert r['multiview']['selected_strategy']=='single_photo_fallback';assert r['multiview']['source']==metadata
  original=previous['result'];changes=[s for s in r['squares'] if r['squares'][s]['symbol']!=original['squares'][s]['symbol']];assert not changes
  output.write_text(json.dumps(dict(kind='API frame replay, not browser video decoding',source_frame=f,candidate=c,result=r,post_scan_wall_seconds=time.perf_counter()-start,changed_squares=changes),indent=2));print('PASS actual one-view uploaded-video session; unchanged photo fallback; zero changed squares')
 finally:rpc('cancel',{'session_id':sid})
if __name__=='__main__':main()
