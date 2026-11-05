"""Three post-test illustrations from saved predictions; no model inference or selection."""
import html,json,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.chess.labels import CLASS_NAMES
from src.training.dataset_c9 import C9Squares
ROOT=Path('outputs/evaluation/convnext_c9')

def main():
    a,b=[json.loads((ROOT/'export_v1'/f'{name}_internal_test.json').read_text()) for name in ('b2','convnext')]
    if not (ROOT/'export_v1/c9_result.json').exists():raise ValueError('Completed locked evaluation required')
    assert [(o['board_id'],o['square'],o['truth']) for o in a['observations']]==[(o['board_id'],o['square'],o['truth']) for o in b['observations']]
    ds=C9Squares(json.loads((ROOT/'approved_cache_with_reviews_v1.json').read_text()),'internal_test',allow_internal_test=True);lookup={(v['id'],o['square']):i for i,(v,o) in enumerate(ds.rows)}
    cases=[]
    for label,kind in [('black queen at board edge',11),('black king residual error',12),('confident empty false positive',0)]:
        for index,o in enumerate(b['observations']):
            p=np.exp(np.array(o['logits'])-max(o['logits']));p/=p.sum()
            if o['truth']==kind and p.argmax()!=kind and (kind!=0 or p.max()>=.9):cases.append((label,index));break
    out=ROOT/'visual_v1';target=out/'test_edge_cases_v1.html'
    if target.exists():raise ValueError('Edge diagnostics already created')
    doc=['<!doctype html><meta charset="utf-8"><title>C9 post-test edge cases</title><style>body{font:16px system-ui;margin:24px}td,th{padding:8px;border:1px solid #aaa}table{border-collapse:collapse}section{padding:20px;border-top:1px solid #aaa}</style><h1>C.9 post-test illustrations</h1><p>Read-only diagnostics after locked evaluation. No new inference, checkpoint selection, geometry edits or label changes. The red line is the target footprint; pixels are the actual B.2-compatible RGB crop.</p>']
    for label,index in cases:
        o=b['observations'][index];packed=ds.packed(lookup[(o['board_id'],o['square'])]);rgb=packed[:3].transpose(1,2,0).copy();contours,_=cv2.findContours(packed[3].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,(255,0,0),1)
        filename=f'test_edge_{index}.jpg';cv2.imwrite(str(out/filename),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92]);doc.append(f'<section><h2>{label}: {html.escape(o["board_id"])} {o["square"]}</h2><img src="{filename}"><p>Ground truth: {CLASS_NAMES[o["truth"]]}</p><table><tr><th>Model</th><th>Top three uncalibrated probabilities</th></tr>')
        for name,record in [('b2',a),('convnext',b)]:
            z=np.array(record['observations'][index]['logits']);p=np.exp(z-z.max());p/=p.sum();top=np.argsort(-p)[:3];doc.append('<tr><td>'+name+'</td><td>'+', '.join(f'{CLASS_NAMES[k]}: {p[k]:.4f}' for k in top)+'</td></tr>')
        doc.append('</table></section>')
    target.write_text('\n'.join(doc));print(json.dumps([dict(category=c,observation_index=i,board_id=b['observations'][i]['board_id'],square=b['observations'][i]['square']) for c,i in cases],indent=2))

if __name__=='__main__':main()
