"""Real HTTP replay from bounded pre-registered historical video samples."""
import argparse,json,hashlib,time,sys,resource
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.training.multidataset import SQUARES,placement_labels
ROOT=Path('outputs/evaluation/assisted_mvp15_v1')

def request(base,path,payload=None,*,raw=None,headers=None):
    body=raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    h={'Content-Type':'image/jpeg' if raw is not None else 'application/json'};h.update(headers or {})
    start=time.perf_counter()
    try:
        with urlopen(Request(base+'/api/photo/scan/'+path,data=body,headers=h),timeout=60) as response:
            return json.load(response),time.perf_counter()-start
    except HTTPError as error:
        raise RuntimeError(error.read().decode()) from error

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',default='http://127.0.0.1:8765');p.add_argument('--output',type=Path,default=ROOT/'application_replay_v1.json');args=p.parse_args()
    if args.output.exists():raise ValueError('Use a fresh versioned output')
    source=json.loads((ROOT/'replay_source_v1.json').read_text());runs=[]
    truth=placement_labels('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR')
    for budget in (1,2,3):
        session,_=request(args.url,'session',{'unchanged_position':True});sid=session['session_id'];records=[]
        try:
            for index,frame in enumerate(source['frames']):
                path=Path(frame['path'])
                if hashlib.sha256(path.read_bytes()).hexdigest()!=frame['derivative_sha256']:raise ValueError('Historical frame changed')
                image=cv2.imread(str(path));ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,95])
                if not ok:raise ValueError('Frame encoding failed')
                response,seconds=request(args.url,'candidate',raw=encoded.tobytes(),headers={'X-Scan-Session':sid,'X-Scan-Frame':frame['frame_id'],'X-Scan-Timestamp':str(frame['relative_timestamp_ms'])})
                records.append(dict(frame_id=frame['frame_id'],source_timestamp_ms=frame['relative_timestamp_ms'],status=response['status'],reason=response.get('reason'),retained_ids=response['retained_ids'],request_seconds=seconds))
            begun=time.perf_counter();request(args.url,'finish',{'session_id':sid,'view_budget':budget});timeline=[]
            while True:
                status,_=request(args.url,'status?session_id='+sid);timeline.append({k:status[k] for k in ('phase','selected_count','inputs_prepared','completed_inputs','total_inputs')})
                if status['phase']=='failed':raise RuntimeError(status['error'])
                if status['phase']=='review_ready':break
                if time.perf_counter()-begun>150:raise TimeoutError('Post-scan timeout')
                time.sleep(.25)
            result=status['result'];multi=result['multiview'];prediction=[result['squares'][s]['symbol'] for s in SQUARES]
            from src.chess.labels import SYMBOL_TO_ID
            ids=[SYMBOL_TO_ID[v] for v in prediction];corrections=sum(a!=b for a,b in zip(ids,truth))
            runs.append(dict(requested_views=budget,actual_views=multi['accepted_view_count'],post_scan_wall_seconds=time.perf_counter()-begun,
                candidate_replay_seconds=sum(r['request_seconds'] for r in records),candidates=records,progress_timeline=timeline,result=result,
                corrections=corrections,exact_board=corrections==0,classification_scope='training-exposed approved pilot position, not independent validation'))
            from src.training.manifest_tools import write_new
            write_new(args.output.with_name(args.output.stem+f'_budget{budget}.json'),runs[-1])
            print('budget',budget,'actual',multi['accepted_view_count'],'post-scan',runs[-1]['post_scan_wall_seconds'],'errors',corrections,flush=True)
        finally:
            try:request(args.url,'cancel',{'session_id':sid})
            except RuntimeError:pass
    from src.training.manifest_tools import write_new
    write_new(args.output,dict(schema_version='assisted_mvp15_application_replay_v1',source=source,runs=runs,client_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,actual_gpu=False,frames_independent=False))

if __name__=='__main__':main()
