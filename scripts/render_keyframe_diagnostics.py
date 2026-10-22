"""Standalone offline engineering diagnostics; no browser integration or completion UI."""
import html
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.ingest_native_pilot import contact_sheet
from scripts.select_video_keyframes import read
from src.scanning.video_keyframes import STRATEGIES


def sheet(items, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if items:contact_sheet(items,path)
    else:
        image=np.full((140,920,3),245,np.uint8)
        cv2.putText(image,'No examples in this category; no synthetic ground truth invented.',(12,50),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1)
        cv2.imwrite(str(path),image)


def text(x,y,value,size=12):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-family="sans-serif">{html.escape(str(value))}</text>'


def timeline(session, selections, path):
    width,height=1240,360;left,right=180,1210;end=session['duration_seconds']
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',text(10,22,session['position_id']+' — original PTS timeline',16),
         text(10,44,'Blue: completed unverified; red: unresolved; orange: quality gate; selection ticks: black. No tracking/recognition accuracy.')]
    for k,policy in enumerate(['legacy','marked-board']):
        y=80+k*35;svg.append(text(10,y+4,policy+' geometry'))
        for r in session['policies'][policy]:
            color='#e05242' if r['geometry_state'] in ('unresolved','invalid_geometry_binding') else ('#276cb4' if r['eligible'] else '#e59e25')
            if r['geometry_state']=='rejected_incorrect':color='#222'
            if r['geometry_state']=='visually_verified':color='#238b45'
            x=left+r['relative_timestamp_ms']/1000/end*(right-left)
            svg.append(f'<circle cx="{x:.3f}" cy="{y}" r="4" fill="{color}"><title>{html.escape(r["frame_id"]+": "+r["geometry_state"]+" "+str(r["unusable_reasons"]))}</title></circle>')
    rows=session['policies']['marked-board']
    for k,strategy in enumerate(STRATEGIES):
        y=160+k*36;svg.append(text(10,y+4,strategy+' @5'))
        ids=selections[strategy]
        for r in rows:
            x=left+r['relative_timestamp_ms']/1000/end*(right-left)
            svg.append(f'<circle cx="{x:.3f}" cy="{y}" r="2" fill="#ccc"/>')
            if r['frame_id'] in ids:svg.append(f'<line x1="{x:.3f}" x2="{x:.3f}" y1="{y-9}" y2="{y+9}" stroke="#111" stroke-width="3"/>')
    svg.append(f'<line x1="{left}" x2="{right}" y1="320" y2="320" stroke="#555"/>')
    for t in np.arange(0,end+1,10):
        x=left+t/end*(right-left);svg.append(text(round(x),342,f'{t:.0f}s'))
    svg.append('</svg>');Path(path).write_text('\n'.join(svg)+'\n')


def render(output,datasets,metrics,settings):
    output=Path(output);primary=datasets[0];contacts=output/'contacts';contacts.mkdir();timelines=output/'timelines';timelines.mkdir();rejects=output/'rejections';rejects.mkdir()
    pages=[]
    for session in primary['sessions']:
        sid=session['capture_session_id'];selected_for_timeline={}
        for policy in ['legacy','marked-board']:
            rows=session['policies'][policy];lookup={r['frame_id']:r for r in rows}
            for strategy in STRATEGIES:
                path=output/'selected'/primary['name']/policy/strategy/'budget_5'/'selected.json'
                selected=next(s for s in read(path)['sessions'] if s['capture_session_id']==sid)['selected']
                ids=[r['frame_id'] for r in selected]
                destination=contacts/f'{sid}_{policy}_{strategy}_5.png'
                sheet([(lookup[f]['path'],f"{lookup[f]['relative_timestamp_ms']/1000:.3f}s Q={lookup[f]['quality_score']:.3f} UNVERIFIED") for f in ids],destination)
                if policy=='marked-board':selected_for_timeline[strategy]=ids
            unusable=[r for r in rows if not r['eligible']]
            sheet([(r['path'],f"{r['relative_timestamp_ms']/1000:.3f}s {r['geometry_state']}") for r in unusable[:8]],rejects/f'{sid}_{policy}_unusable.png')
        rows=session['policies']['marked-board']
        default=read(output/'selected'/primary['name']/'marked-board'/'quality_diverse'/'budget_5'/'selected.json')
        decisions=next(s for s in default['sessions'] if s['capture_session_id']==sid)['candidate_decisions'];lookup={r['frame_id']:r for r in rows}
        redundant=[lookup[d['frame_id']] for d in decisions if 'redundant_with_selected' in d['reasons']]
        sheet([(r['path'],f"{r['relative_timestamp_ms']/1000:.3f}s rejected as REDUNDANT") for r in redundant[:8]],rejects/f'{sid}_redundant.png')
        badquality=[r for r in rows if r['quality_rejection_reasons']]
        if badquality:
            items=[(r['path'],f"{r['relative_timestamp_ms']/1000:.3f}s quality gate: {r['quality_rejection_reasons'][0]}") for r in badquality[:8]]
        else:
            lowest=sorted([r for r in rows if r['quality_score'] is not None],key=lambda r:r['board_sharpness'])[:4]
            items=[(r['path'],f"{r['relative_timestamp_ms']/1000:.3f}s low sharpness; PASS") for r in lowest]
        sheet(items,rejects/f'{sid}_quality.png')
        timeline(session,selected_for_timeline,timelines/f'{sid}.svg')
        pages.append(f'<h2>{html.escape(session["position_id"])}</h2><p>Estimated geometry remains unverified. Occlusion and elevation are unknown.</p><img width="1100" src="timelines/{sid}.svg"/><p><a href="contacts/{sid}_marked-board_quality_diverse_5.png">Diverse @5</a> · <a href="contacts/{sid}_marked-board_uniform_5.png">Uniform @5</a> · <a href="rejections/{sid}_marked-board_unusable.png">Unusable</a> · <a href="rejections/{sid}_redundant.png">Redundant</a> · <a href="rejections/{sid}_quality.png">Quality diagnostics</a></p>')
    heads=['session','policy','strategy','budget','eligible','selected','mean quality','predicted sectors','span seconds','similar pairs']
    table=['<table border="1"><tr>'+''.join('<th>'+h+'</th>' for h in heads)+'</tr>']
    for m in metrics:
        if m['dataset']!=primary['name']:continue
        values=[m['session_id'],m['policy'],m['strategy'],m['budget'],m['geometry_eligible'],m['selected_count'],
                round(m['mean_quality'],3) if m['mean_quality'] is not None else 'n/a',m['selected_predicted_orientations'],
                round(m['temporal_span_seconds'],1),m['soft_similar_pairs']]
        table.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    table.append('</table>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>D.1 offline diagnostics</title><h1>D.1 offline keyframe audit</h1><p>Development-only, five correlated sessions. No classifier, fusion or completion evaluation. Image-plane sectors are predicted proxies, not calibrated camera pose.</p>'+''.join(pages)+''.join(table)+'\n')
