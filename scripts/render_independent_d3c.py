"""Bounded D.3c review packet and one multi-view crop case per recording."""
import argparse
import hashlib
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write
from scripts.report_independent_d3c import gzread
from scripts.run_square_evidence import make_inputs
from src.scanning.interim_video import decode_exact,pixel_hash
from src.scanning.native_pilot import sha256_file
from src.chess.labels import CLASS_NAMES

SHORT=['--','wP','wN','wB','wR','wQ','wK','bP','bN','bB','bR','bQ','bK']


def boardsheet(boards):
    sheet=np.full((310,4*280,3),250,np.uint8)
    for j,name in enumerate(['supplied FEN']+[b['method'] for b in boards]):
        cv2.putText(sheet,name,(j*280+5,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(10,10,10),1)
        for i in range(64):
            s=boards[0]['squares'][i] if j==0 else boards[j-1]['squares'][i]
            k=s['true_class_id'] if j==0 else s['predicted_class_id'];r,c=divmod(i,8);x=j*280+c*33;y=34+r*33
            color=(205,225,235) if (r+c)%2==0 else (145,170,190)
            if j and not s['correct']:color=(100,120,245)
            cv2.rectangle(sheet,(x,y),(x+32,y+32),color,-1);cv2.putText(sheet,SHORT[k],(x+3,y+22),cv2.FONT_HERSHEY_SIMPLEX,.45,(10,10,10),1)
    return sheet


def render(run,evaluation,output):
    if output.exists():raise ValueError('Fresh diagnostic directory required')
    result=gzread(evaluation/'results.json.gz');reg=read(run/'registration.json');obs=gzread(run/'recognition/observations.json.gz')['observations'];cfg=read('configs/experiments/evidence_d2.json')
    output.mkdir();pages=[];cases=[]
    for session in reg['sessions']:
        sid=session['session_id'];folder=run/sid;g=read(folder/'session.json');rows={}
        for p in folder.glob('candidate_*.json'):
            f=read(p);rows[f['frame_id']]=f
        selected=[rows[f] for f in g['selection']['selected']];boards=[b for b in result['boards'] if b['session_id']==sid]
        if not boards:pages.append(f'<h2>{sid}</h2><p>No complete five-frame comparison. See manifest failures.</p>');continue
        target=next(b for b in boards if b['method']=='mean_logits');wrong=[s for s in target['squares'] if not s['correct']]
        case=max(wrong,key=lambda s:s['maximum_score']) if wrong else max(target['squares'],key=lambda s:s['variation_ratio'])
        square=case['canonical_square'];cells=[];contact=[]
        for i,f in enumerate(selected):
            image=decode_exact(session['source']['path'],f)
            if pixel_hash(image)!=f['decoded_pixel_sha256']:raise ValueError('Original pixels changed')
            inputs=make_inputs(image,f['geometry'],cfg);inp=next(x for x in inputs if x['canonical_square']==square)
            o=next(o for o in obs if o['session_id']==sid and o['frame_id']==f['frame_id'] and o['canonical_square']==square)
            if hashlib.sha256(inp['packed'].tobytes()).hexdigest()!=o['input_packed_sha256']:raise ValueError('Actual crop replay mismatch')
            cell=np.full((580,300,3),250,np.uint8);bgr=cv2.cvtColor(inp['packed'][:3].transpose(1,2,0),cv2.COLOR_RGB2BGR)
            contours,_=cv2.findContours(inp['packed'][3].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(bgr,contours,-1,(0,255,0),2);cell[50:274,38:262]=bgr
            texts=[f'{square} truth {SHORT[case["true_class_id"]]} {o["timestamp_ms"]/1000:.3f}s',f'pred {SHORT[o["predicted_class_id"]]} max {o["maximum_probability"]:.3f}']
            for n,t in enumerate(texts):cv2.putText(cell,t,(6,18+n*20),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
            for k,p in enumerate(o['probabilities']):
                y=300+20*k;cv2.putText(cell,CLASS_NAMES[k],(4,y),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1);cv2.rectangle(cell,(110,y-9),(110+int(130*p),y),(170,100,20),-1);cv2.putText(cell,f'{p:.3f}',(247,y),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,0),1)
            cells.append(cell)
            overlay=cv2.imread(g['reviews'][i]['overlay_path']);overlay=cv2.resize(overlay,(270,480));cv2.putText(overlay,f'V{i+1} {o["timestamp_ms"]/1000:.2f}s',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1);contact.append(overlay)
        casepath=output/(sid+'_case.jpg');cv2.imwrite(str(casepath),cv2.hconcat(cells),[cv2.IMWRITE_JPEG_QUALITY,90])
        cv2.imwrite(str(output/(sid+'_views.jpg')),cv2.hconcat(contact),[cv2.IMWRITE_JPEG_QUALITY,88])
        cv2.imwrite(str(output/(sid+'_boards.png')),boardsheet(boards))
        notes=[f'{b["method"]}: {64-b["metrics"]["correct"]} errors; occupied {b["metrics"]["occupied_correct"]}/{b["metrics"]["occupied_count"]}; mean score {b["mean_probability"]:.3f}; {b["reliability"]["abstained_count"]} withheld' for b in boards]
        cases.append(dict(session_id=sid,canonical_square=square,truth=case['true_class_id'],mean_logit_prediction=case['predicted_class_id'],mean_logit_score=case['maximum_score'],mean_logit_abstained=case['abstained'],
            classification_attribution='conditional_on_unverified_geometry_and_supplied_FEN',path=str(casepath)))
        sections=f'<h2>{sid}</h2><p>'+html.escape(' | '.join(notes))+'</p>'
        sections+=f'<img style="max-width:100%" src="{sid}_views.jpg"><p>Source grid overlays, provisional completion only.</p><img style="max-width:100%" src="{sid}_boards.png"><p>Left: supplied FEN. Then single, mean probabilities, mean logits. Red: conditional error.</p><img style="max-width:100%" src="{sid}_case.jpg">'
        pages.append(sections)
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>D.3c limited interim diagnostics</h1><p>Original videos are unchanged. Missing independent reference photos and human geometry verification: predictions/metrics are provisional. No automatic completion.</p>'+''.join(pages))
    write(output/'cases.json',dict(schema_version='independent_fusion_diagnostics_d3c_v1',cases=cases,original_pixel_crop_replays=len(cases)*5,source_sha256=sha256_file(Path(__file__))))
    print('Rendered one compact multi-view case, view contact and board comparison per session.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();render(a.run,a.evaluation,a.output_dir)
