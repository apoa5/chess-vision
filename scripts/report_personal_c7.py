"""Compact fixed-validation comparison; no classifier inference or tuning."""
import html
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.training.manifest_tools import rebase_manifest, write_new
from src.training.multidataset import MultiDatasetSquares, digest

ROOT=Path('outputs/evaluation/personal_c7')
NAMES=['b2','chesscog','B','C']
GLYPHS=['·','♙','♘','♗','♖','♕','♔','♟','♞','♝','♜','♛','♚']

def board_html(values,truth):
    return '<div class="board">'+''.join('<div class="tile '+('dark ' if (i//8+i%8)%2 else '')+('wrong' if k!=truth[i] else '')+'" title="'+f'{"abcdefgh"[i%8]}{8-i//8}: {CLASS_NAMES[k]}'+'">'+GLYPHS[k]+'</div>' for i,k in enumerate(values))+'</div>'

def main():
    records={name:json.loads((ROOT/f'validation_{name}_v1.json').read_text()) for name in NAMES}
    base=records['b2']['observations'];ids=[(o['board_id'],o['square'],o['truth']) for o in base]
    for r in records.values():
        if [(o['board_id'],o['square'],o['truth']) for o in r['observations']]!=ids:raise ValueError('Evaluation coordinates/labels differ')
    scores={k:np.array([o['logits'] for o in r['observations']]) for k,r in records.items()}
    pred={k:z.argmax(1) for k,z in scores.items()};y=np.array([o['truth'] for o in base])
    report={'schema_version':'personal_c7_comparison_v1','development_only':True,'physical_validation_positions':3,'validation_boards':25,'labels_and_order_identical':True,'models':{k:{a:v for a,v in r.items() if a!='observations'} for k,r in records.items()},'paired_changes':{}}
    for a,b in [('b2','chesscog'),('chesscog','B'),('chesscog','C'),('B','C'),('b2','B'),('b2','C')]:
        fixes=np.flatnonzero((pred[a]!=y)&(pred[b]==y));regressions=np.flatnonzero((pred[a]==y)&(pred[b]!=y))
        report['paired_changes'][a+'->'+b]={'corrected':len(fixes),'regressed':len(regressions),'corrected_squares':[ids[i][:2] for i in fixes],'regressed_squares':[ids[i][:2] for i in regressions]}
    output=ROOT/'visual_v1'
    if output.exists():raise ValueError('Fresh visual output required')
    output.mkdir()
    style='<style>body{font:16px system-ui;margin:24px;background:#fafafa}section{border-top:1px solid #aaa;padding:20px 0}.row{display:flex;gap:20px;flex-wrap:wrap}.board{display:grid;grid-template-columns:repeat(8,28px)}.tile{height:28px;text-align:center;background:#efdcc0;font-size:24px}.dark{background:#b98e68}.wrong{outline:2px solid red;outline-offset:-2px}.crop{width:224px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>'
    doc=['<!doctype html><meta charset="utf-8"><title>C7 validation comparison</title>',style,'<h1>C.7 fixed development validation</h1><p>25 photographs, three physical positions; repeatedly exposed validation, not independent generalization. Red borders mark prediction errors against existing approved labels. No geometry was rerun.</p>']
    m=rebase_manifest(ROOT/'bundle_v1');ds=MultiDatasetSquares(m,'val');lookup={(b['id'],o['square']):i for i,(b,o) in enumerate(ds.rows)}
    chosen=[]
    # Bounded examples, prespecified categories. Selection affects illustration only.
    cases=[('king/rook',lambda i:y[i] in (4,6,10,12) and any(pred[k][i]!=y[i] for k in NAMES)),
           ('empty false positive',lambda i:y[i]==0 and any(pred[k][i]!=0 for k in NAMES)),
           ('video fixes photo arm',lambda i:pred['B'][i]!=y[i] and pred['C'][i]==y[i]),
           ('video regresses photo arm',lambda i:pred['B'][i]==y[i] and pred['C'][i]!=y[i]),
           ('adaptation correction',lambda i:pred['chesscog'][i]!=y[i] and pred['B'][i]==y[i]),
           ('persistent failure',lambda i:pred['B'][i]!=y[i] and pred['C'][i]!=y[i])]
    for kind,condition in cases:
        picks=[i for i in range(len(y)) if condition(i) and i not in chosen][:2];chosen.extend(picks)
        for i in picks:
            board,square,_=ids[i];packed=ds.packed(lookup[(board,square)]);rgb=packed[:3].transpose(1,2,0).copy();mask=packed[3].astype(np.uint8)
            contour,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contour,-1,(255,0,0),1)
            name=f'crop_{i}.jpg';cv2.imwrite(str(output/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
            doc.append(f'<section><h2>{html.escape(kind)} — {html.escape(board)} {square}</h2><img class="crop" src="{name}"><p>Ground truth: {CLASS_NAMES[y[i]]}; red outline is target footprint. Neighboring-piece interference or silhouette truncation may remain; predictions alone do not establish cause.</p><table><tr><th>Model</th><th>Prediction</th><th>Raw max probability</th></tr>')
            for k in NAMES:
                z=scores[k][i];p=np.exp(z-z.max());p/=p.sum();doc.append(f'<tr><td>{k}</td><td>{CLASS_NAMES[pred[k][i]]}</td><td>{p.max():.4f}</td></tr>')
            doc.append('</table></section>')
    for board in sorted({o['board_id'] for o in base}):
        ix=[i for i,o in enumerate(base) if o['board_id']==board]
        ordered=sorted(ix,key=lambda i:(-int(base[i]['square'][1]),base[i]['square'][0]));truth=y[ordered].tolist()
        b=next(b for b in m['boards'] if b['id']==board);source=cv2.imread(b['source']['path']);height,width=source.shape[:2]
        source=cv2.resize(source,(360,360),interpolation=cv2.INTER_AREA)
        for o in b['observations']:
            polygon=np.asarray(b['geometry']['context_square_polygons'][o['camera_index']])*np.array([360/width,360/height]);points=np.rint(polygon).astype(np.int32)
            cv2.polylines(source,[points],True,(0,180,0),1);centre=np.mean(points,axis=0).astype(int)
            cv2.putText(source,o['square'],tuple(centre-5),cv2.FONT_HERSHEY_SIMPLEX,.25,(0,0,255),1)
        name='context_'+str(len(doc))+'.jpg';cv2.imwrite(str(output/name),source,[cv2.IMWRITE_JPEG_QUALITY,80])
        doc.append('<section><h2>'+html.escape(board)+'</h2><img width="240" src="'+name+'"><div class="row"><div>Ground truth'+board_html(truth,truth)+'</div>')
        for k in NAMES:
            values=pred[k][ordered].tolist();doc.append('<div>'+k+f' ({sum(a!=b for a,b in zip(values,truth))} errors)'+board_html(values,truth)+'</div>')
        doc.append('</div></section>')
    (output/'index.html').write_text('\n'.join(doc))
    report['diagnostic_square_count']=len(chosen);report['source_artifacts_sha256']={k:digest(ROOT/f'validation_{k}_v1.json') for k in NAMES}
    write_new(ROOT/'comparison_v1.json',report)
    print(json.dumps({k:{f:r['metrics'][f] for f in ['overall_accuracy','occupied_accuracy','empty_false_positives','exact_boards']} for k,r in records.items()},indent=2))

if __name__=='__main__':main()
