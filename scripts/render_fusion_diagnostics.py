"""D.3 post-evaluation visuals and occupied-only reliability; never changes predictions."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write
from scripts.characterize_square_evidence import draw_crop_distributions
from src.chess.labels import CLASS_NAMES,CLASSES
from src.scanning.fusion_metrics import reliability
from src.scanning.native_pilot import sha256_file


def score_strip(boards,square):
    columns=[]
    for b in boards:
        s=next(x for x in b['squares'] if x['canonical_square']==square);im=np.full((360,320,3),250,np.uint8)
        lines=[b['method'],f"pred={s['predicted_class']} {'CORRECT' if s['correct'] else 'WRONG'}",f"score={s['maximum_score']:.3f} abstain={s['abstained']}"]
        for i,t in enumerate(lines):cv2.putText(im,t,(4,17+i*18),cv2.FONT_HERSHEY_SIMPLEX,.35,(10,10,10),1,cv2.LINE_AA)
        for k,p in enumerate(s['score_distribution']):
            y=85+k*20;cv2.putText(im,CLASS_NAMES[k],(3,y),cv2.FONT_HERSHEY_SIMPLEX,.32,(20,20,20),1)
            cv2.rectangle(im,(120,y-10),(120+round(p*150),y),(60,140,90) if s['correct'] else (60,90,220),-1)
            cv2.putText(im,f'{p:.3f}',(275,y),cv2.FONT_HERSHEY_SIMPLEX,.3,(20,20,20),1)
        columns.append(im)
    return cv2.hconcat(columns)


def board_sheet(boards):
    tiles=[]
    for b in boards:
        im=np.full((650,600,3),250,np.uint8)
        cv2.putText(im,b['method'],(8,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(15,15,15),1)
        cv2.putText(im,f"errors={64-b['metrics']['correct']} withheld={b['reliability']['abstained_count']}",(8,45),cv2.FONT_HERSHEY_SIMPLEX,.55,(15,15,15),1)
        for i,s in enumerate(b['squares']):
            x=25+(i%8)*68;y=70+(i//8)*68;base=(225,230,225) if (i//8+i%8)%2==0 else (160,180,170)
            cv2.rectangle(im,(x,y),(x+66,y+66),base,-1);color=(30,150,30) if s['correct'] else (20,20,230)
            cv2.rectangle(im,(x+1,y+1),(x+65,y+65),color,3)
            cv2.putText(im,s['canonical_square'],(x+3,y+14),cv2.FONT_HERSHEY_SIMPLEX,.35,(20,20,20),1)
            symbol=CLASSES[s['predicted_class_id']].symbol or '.';target=CLASSES[s['true_class_id']].symbol or '.'
            cv2.putText(im,symbol,(x+23,y+39),cv2.FONT_HERSHEY_SIMPLEX,.8,(10,10,10),2)
            cv2.putText(im,f"GT:{target}{' ?' if s['abstained'] else ''}",(x+3,y+57),cv2.FONT_HERSHEY_SIMPLEX,.32,(20,20,20),1)
        tiles.append(im)
    return cv2.hconcat(tiles)


def run(evaluation_path,output):
    if output.exists():raise ValueError('Fresh diagnostic directory required')
    start=time.perf_counter();r=read(evaluation_path);config=r['config'];e=read(config['evidence_manifest']);obs=[json.loads(x) for x in Path(e['observations_path']).read_text().splitlines()];by_id={o['observation_id']:o for o in obs}
    if sha256_file(e['observations_path'])!=e['observations_sha256']:raise ValueError('D.2 observations changed')
    output.mkdir(parents=True,exist_ok=False);examples=[]
    for condition in config['conditions']:
        for session in e['sessions']:
            sid=session['capture_session_id'];boards=[b for b in r['boards'] if b['session_id']==sid and b['condition']==condition]
            cv2.imwrite(str(output/(sid+'_'+condition+'_boards.jpg')),board_sheet(boards))
            table=['<tr><th>square</th><th>truth</th>'+''.join('<th>'+html.escape(b['method'])+'</th>' for b in boards)+'</tr>']
            for i in range(64):
                cells=[]
                for b in boards:
                    s=b['squares'][i];text=f"{s['predicted_class']} score {s['maximum_score']:.3f}; withheld={s['abstained']}; conflict={s['strong_conflict']}"
                    cells.append(f'<td style="background:{"#dbf4dc" if s["correct"] else "#ffd8d8"}">{html.escape(text)}</td>')
                s=boards[0]['squares'][i];table.append('<tr><td>'+s['canonical_square']+'</td><td>'+CLASS_NAMES[s['true_class_id']]+'</td>'+''.join(cells)+'</tr>')
            (output/(sid+'_'+condition+'.html')).write_text('<!doctype html><meta charset="utf-8"><h1>'+html.escape(sid+' '+condition)+'</h1><p>Tentative argmaxes and diagnostic withholding only; no scan-completion gate.</p><table border="1">'+''.join(table)+'</table>')
    cases=[('starting_g1_rescue','uniform_5','starting','g1'),('opening_c7_rescue','uniform_5','opening','c7'),
           ('puzzle_d7_regression','uniform_5','puzzle','d7'),('endgame_h4_confident_miss','uniform_5','endgame','h4'),
           ('opening_b1_repeated_bias','diversity_5','opening','b1'),('opening_d1_minority','uniform_5','opening','d1')]
    for name,condition,position,square in cases:
        boards=[b for b in r['boards'] if b['condition']==condition and b['position_id'].endswith(position)]
        s=next(s for s in boards[0]['squares'] if s['canonical_square']==square);selected=[by_id[oid] for oid in s['observation_ids']]
        raw=output/(name+'_inputs.png');draw_crop_distributions(selected,f'{position} {square} truth={CLASS_NAMES[s["true_class_id"]]} — actual B.2 inputs',raw)
        strip=score_strip(boards,square);im=cv2.imread(str(raw))
        if strip.shape[1]!=im.shape[1]:strip=cv2.resize(strip,(im.shape[1],strip.shape[0]))
        path=output/(name+'.png');cv2.imwrite(str(path),cv2.vconcat([im,strip]))
        examples.append(dict(name=name,condition=condition,session_id=boards[0]['session_id'],canonical_square=square,true_class_id=s['true_class_id'],
                             path=str(path),observation_ids=s['observation_ids'],input_packed_sha256=[o['input_packed_sha256'] for o in selected]))
    occupied=[]
    for a in r['aggregates']:
        boards=[b for b in r['boards'] if b['condition']==a['condition'] and b['method']==a['method']]
        squares=[s for b in boards for s in b['squares'] if s['true_class_id']!=0]
        m=reliability([s['true_class_id'] for s in squares],[s['score_distribution'] for s in squares],[s['predicted_class_id'] for s in squares],[s['abstained'] for s in squares],config)
        occupied.append(dict(condition=a['condition'],method=a['method'],occupied_reliability=m))
    result=dict(schema_version='offline_fusion_diagnostics_d3_v1',evaluation_sha256=sha256_file(evaluation_path),examples=examples,
        occupied_only_reliability=occupied,rendering_seconds=time.perf_counter()-start,source_sha256=sha256_file(Path(__file__)))
    write(output/'diagnostics.json',result)
    pages=sorted(output.glob('*.html'));(output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>D.3 offline diagnostics</h1><p>No scan-completion or production integration.</p><ul>'+''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in pages)+'</ul>'+''.join(f'<p><a href="{Path(x["path"]).name}">{x["name"]}</a></p>' for x in examples))
    print('Rendered 15 board comparisons, 6 crop/distribution cases, occupied reliability.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.evaluation,a.output_dir)
