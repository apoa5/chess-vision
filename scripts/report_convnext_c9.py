"""Replay metrics/figures from frozen C9 logits, without another test inference."""
import csv,html,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.chess.labels import CLASS_NAMES
from src.training.dataset_c9 import C9Squares
from src.training.manifest_tools import write_new
from src.training.multidataset import digest
from src.scanning.square_evidence import classification_metrics
ROOT=Path('outputs/evaluation/convnext_c9')
GLYPHS=['·','♙','♘','♗','♖','♕','♔','♟','♞','♝','♜','♛','♚']

def aligned(a,b):
    key=lambda r:[(o['board_id'],o['square'],o['truth']) for o in r['observations']]
    if key(a)!=key(b):raise ValueError('Compared coordinates/labels differ')
    return True

def digital(values,truth):
    return '<div class="board">'+''.join('<span class="'+('dark ' if (i//8+i%8)%2 else '')+('wrong' if p!=t else '')+'">'+GLYPHS[p]+'</span>' for i,(p,t) in enumerate(zip(values,truth)))+'</div>'

def curve(history,target):
    width,height=700,300;left,right,top,bottom=55,20,25,45
    maximum=max(max(h['training_loss'],h['validation_loss']) for h in history)*1.1
    x=lambda i:left+i*(width-left-right)/max(1,len(history)-1)
    y=lambda value:height-bottom-value/maximum*(height-top-bottom)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',f'<path d="M{left},{top} V{height-bottom} H{width-right}" fill="none" stroke="black"/>']
    for color,key in [('blue','training_loss'),('red','validation_loss')]:
        points=' '.join(f'{x(i):.1f},{y(h[key]):.1f}' for i,h in enumerate(history))
        svg.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="2"/>')
    svg+=['<text x="70" y="20" font-size="13">Blue: weighted train CE; red: macro-position validation CE</text>','<text x="240" y="290" font-size="13">Completed coverage-cycle validation events</text></svg>']
    target.write_text('\n'.join(svg))

def main():
    remote=ROOT/'export_v1';training=json.loads((remote/'c9_result.json').read_text());lock=json.loads((remote/'selection_lock.json').read_text())
    if training['selection_lock']!=lock:raise ValueError('Selection lock mismatch')
    manifest=json.loads((ROOT/'approved_cache_with_reviews_v1.json').read_text());records={}
    for split in ('val','internal_test'):
        for model in ('b2','convnext'):
            records[model+'_'+split]=json.loads((remote/(model+'_'+split+'.json')).read_text())
        aligned(records['b2_'+split],records['convnext_'+split])
    comparison=dict(schema_version='convnext_c9_comparison_v1',independent_generalization=False,selection_lock=lock,conditions={},paired_changes={})
    for split in ('val','internal_test'):
        a,b=records['b2_'+split],records['convnext_'+split];y=np.array([o['truth'] for o in a['observations']]);ap=np.argmax([o['logits'] for o in a['observations']],axis=1);bp=np.argmax([o['logits'] for o in b['observations']],axis=1)
        comparison['paired_changes'][split]=dict(corrected=int(((ap!=y)&(bp==y)).sum()),regressed=int(((ap==y)&(bp!=y)).sum()),both_wrong=int(((ap!=y)&(bp!=y)).sum()))
    mlookup={b['id']:b for b in manifest['boards']}
    photo_metadata=json.loads(Path('data/metadata/images.json').read_text())
    for name,r in records.items():
        comparison['conditions'][name]={k:v for k,v in r.items() if k!='observations'}
        positions=r['metrics']['by_position_id'];comparison['conditions'][name]['macro_position_occupied_accuracy']=float(np.mean([v['occupied_accuracy'] for v in positions.values()]))
        observations=r['observations'];truth=np.array([o['truth'] for o in observations]);pred=np.argmax([o['logits'] for o in observations],1);strata={}
        for field in ('dataset','declared_photo_angle','declared_lighting'):
            values=[]
            for o in observations:
                b=mlookup[o['board_id']];meta=photo_metadata.get(b['source_id'],{}) if b['dataset']=='chessvision_photos' else {}
                value=b['dataset'] if field=='dataset' else meta.get('camera_angle_degrees','unknown_video_pose') if field=='declared_photo_angle' else meta.get('lighting','nighttime_artificial' if b['dataset']=='d3c_retired' else 'unknown')
                values.append(str(value))
            strata[field]={v:classification_metrics(truth[np.array(values)==v],pred[np.array(values)==v]) for v in sorted(set(values))}
        comparison['conditions'][name]['condition_breakdowns']=strata
    comparison['source_sha256']={p.name:digest(p) for p in remote.glob('*.json')}
    write_new(ROOT/'comparison_v1.json',comparison)
    with Path('docs/evaluation/convnext_c9_classes_v1.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['condition','class_name','class_id','support','predicted_count','precision','recall']);w.writeheader()
        for name,r in records.items():
            for row in r['metrics']['per_class']:w.writerow(dict(condition=name,class_name=CLASS_NAMES[row['class_id']],**row))
    figures=Path('docs/evaluation/convnext_c9_figures');figures.mkdir(exist_ok=True);curve(training['history'],figures/'learning_curves_v1.svg')
    out=ROOT/'visual_v1';out.mkdir(exist_ok=False)
    doc=['<!doctype html><meta charset="utf-8"><title>C9 locked comparison</title><style>body{font:16px system-ui;margin:24px}.row{display:flex;gap:22px;flex-wrap:wrap}.board{display:grid;grid-template-columns:repeat(8,28px)}.board span{font-size:25px;background:#ecd8bb;text-align:center}.board .dark{background:#ad8056}.board .wrong{outline:2px solid red;outline-offset:-2px}section{border-top:1px solid #aaa;padding:18px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style><h1>C.9 retrospective comparison</h1><p>Frozen predictions and existing reviewed geometry; no fresh test inference or label edits. Red cells are errors. These examples illustrate outcomes, not causal explanations. B.2 exposure differs; read the report.</p>']
    # At most four board examples per split, ranked only to illustrate outcomes.
    for split in ('val','internal_test'):
        a,b=records['b2_'+split],records['convnext_'+split];base=a['observations'];truth=np.array([o['truth'] for o in base]);preds={k:np.argmax([o['logits'] for o in records[k+'_'+split]['observations']],axis=1) for k in ('b2','convnext')}
        rows=b['metrics']['per_board'];ranked=sorted(rows,key=lambda r:r['errors']);chosen=[]
        for row in ranked[:2]+ranked[-2:]:
            if row['board_id'] not in chosen:chosen.append(row['board_id'])
        for bid in chosen:
            ix=sorted([i for i,o in enumerate(base) if o['board_id']==bid],key=lambda i:(-int(base[i]['square'][1]),base[i]['square'][0]));board=mlookup[bid];image=cv2.imread(board['source']['path']);h,w=image.shape[:2]
            image=cv2.resize(image,(400,400),interpolation=cv2.INTER_AREA)
            for o in board['observations']:
                points=np.rint(np.asarray(board['geometry']['context_square_polygons'][o['camera_index']])*[400/w,400/h]).astype(np.int32);cv2.polylines(image,[points],True,(0,180,0),1);cv2.putText(image,o['square'],tuple(points.mean(0).astype(int)-5),cv2.FONT_HERSHEY_SIMPLEX,.25,(0,0,255),1)
            filename=f'board_{len(doc)}.jpg';cv2.imwrite(str(out/filename),image,[cv2.IMWRITE_JPEG_QUALITY,82])
            y=truth[ix].tolist();doc.append(f'<section><h2>{split}: {html.escape(bid)}</h2><img width="300" src="{filename}"><div class="row"><div>Ground truth{digital(y,y)}</div>')
            for k in preds:doc.append(f'<div>{k}: {int((preds[k][ix]!=truth[ix]).sum())} errors{digital(preds[k][ix].tolist(),y)}</div>')
            doc.append('</div></section>')
    # Show actual validation crops only, avoiding another read/inference of test inputs.
    ds=C9Squares(manifest,'val');lookup={(b['id'],o['square']):i for i,(b,o) in enumerate(ds.rows)};base=records['b2_val']['observations'];y=np.array([o['truth'] for o in base]);pa=np.argmax([o['logits'] for o in base],1);pb=np.argmax([o['logits'] for o in records['convnext_val']['observations']],1);used=set()
    cases=[('corrected',lambda i:pa[i]!=y[i] and pb[i]==y[i]),('regressed',lambda i:pa[i]==y[i] and pb[i]!=y[i]),('king or rook',lambda i:y[i] in (4,6,10,12) and (pb[i]!=y[i] or pa[i]!=y[i])),('empty false positive',lambda i:y[i]==0 and pb[i]!=0),('persistent failure',lambda i:pa[i]!=y[i] and pb[i]!=y[i])]
    for label,condition in cases:
        for i in [i for i in range(len(y)) if i not in used and condition(i)][:2]:
            used.add(i);o=base[i];packed=ds.packed(lookup[(o['board_id'],o['square'])]);rgb=packed[:3].transpose(1,2,0).copy();contours,_=cv2.findContours(packed[3].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,(255,0,0),1)
            filename=f'crop_{i}.jpg';cv2.imwrite(str(out/filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90]);doc.append(f'<section><h2>{label}: {html.escape(o["board_id"])} {o["square"]}</h2><img src="{filename}"><p>Ground truth: {CLASS_NAMES[y[i]]}. Target mask outlined in red.</p><table><tr><th>Model</th><th>Prediction</th><th>Uncalibrated probability</th></tr>')
            for k in ('b2','convnext'):
                z=np.array(records[k+'_val']['observations'][i]['logits']);p=np.exp(z-z.max());p/=p.sum();doc.append(f'<tr><td>{k}</td><td>{CLASS_NAMES[p.argmax()]}</td><td>{p.max():.4f}</td></tr>')
            doc.append('</table></section>')
    (out/'index.html').write_text('\n'.join(doc));print(json.dumps(comparison['paired_changes'],indent=2))

if __name__=='__main__':main()
