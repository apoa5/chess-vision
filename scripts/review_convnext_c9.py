"""Prepare C9 grid/mask packets; only explicit attributed reviews admit new frames."""
import argparse,copy,hashlib,html,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.training.manifest_tools import read,write_new
from src.training.multidataset import digest,PROFILE,SQUARES,placement_labels
from src.scanning.interim_video import decode_exact,pixel_hash
from scripts.run_square_evidence import make_inputs
ROOT=Path('outputs/evaluation/convnext_c9')
CHECKS=('fen_agreement','unchanged_position','semantic_orientation','corners','canonical_mapping','crop_alignment','all_crops_reviewed')

def signature(item):return hashlib.sha256(json.dumps({k:item[k] for k in ['id','source_sha256','decoded_pixel_sha256','geometry_sha256','fen','pts_time']},sort_keys=True).encode()).hexdigest()

def render(item,image,g,dest):
    config=read('configs/experiments/evidence_d2.json');inputs=make_inputs(image,g,config);labels=dict(zip(SQUARES,placement_labels(item['fen'])));preview=image.copy();mosaic=np.zeros((8*96,8*96,3),np.uint8)
    for sq,poly in zip(g['semantic_square_ids'],g['source_square_polygons']):
        pts=np.int32(poly);cv2.polylines(preview,[pts],True,(0,220,30),2);cv2.putText(preview,sq,tuple(pts.mean(0).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2)
    item['packed_sha256']={}
    for row in inputs:
        sq=row['canonical_square'];packed=row['packed'];tile=cv2.resize(cv2.cvtColor(packed[:3].transpose(1,2,0),cv2.COLOR_RGB2BGR),(96,96));mask=cv2.resize(packed[3],(96,96),interpolation=cv2.INTER_NEAREST)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(tile,contours,-1,(255,0,255),1);cv2.putText(tile,f'{sq}:{labels[sq]}',(2,13),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
        r,c=8-int(sq[1]),'abcdefgh'.index(sq[0]);mosaic[r*96:(r+1)*96,c*96:(c+1)*96]=tile;item['packed_sha256'][sq]=hashlib.sha256(packed.tobytes()).hexdigest()
    scale=min(1.,720/max(preview.shape[:2]));preview=cv2.resize(preview,None,fx=scale,fy=scale)
    dest.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(dest/'grid.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,84]);cv2.imwrite(str(dest/'crops.jpg'),mosaic,[cv2.IMWRITE_JPEG_QUALITY,82])
    item['binding']=signature(item);return item


def photo_candidates():
    from scripts.evaluate_independent_d3c import geometry_record
    from src.utils.config import load_config
    cv2.setNumThreads(1);out=ROOT/'photo_candidates_v1';out.mkdir(exist_ok=False);items=[]
    for p in read(ROOT/'inventory_v1.json')['photos']:
        if p['status']!='quarantine_geometry_unverified':continue
        image=cv2.imread(p['source_path']);g=geometry_record(image,{},load_config());item=dict(id='photo:'+p['id'],source=p['source_path'],source_sha256=p['sha256'],decoded_pixel_sha256=pixel_hash(image),fen=p['fen'],
            pts_time=None,geometry=g,geometry_sha256=hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest(),status='unverified' if g.get('completed') else 'unresolved',source_kind='photo',metadata=p)
        if g.get('completed'):
            try:render(item,image,g,out/p['id'])
            except ValueError as e:item.update(status='unresolved',reason=str(e))
        write_new(out/(p['id']+'.json'),item);items.append(item)
        print(p['id'],item['status'],flush=True)
    write_new(out/'summary.json',items)


def packet():
    output=ROOT/'review_v1';output.mkdir(exist_ok=False);inventory=read(ROOT/'inventory_v1.json');known=set()
    for b in read('outputs/evaluation/multidataset_c4/video_manifest_verified_v1.json')['boards']:known.add(b['source_id'])
    for f in read('docs/evaluation/evidence_d2_geometry_review_v1/packet.json')['frames']:known.add(f['frame']['frame_id'])
    items=[]
    for session in inventory['sessions']:
        folder=ROOT/'candidates_v1'/session['id'];summary=read(folder/'summary.json');rows=[read(p) for p in sorted(folder.glob('candidate_*.json'))];selected=set(summary['adaptive_selected'])
        for row in rows:
            if row['frame_id'] not in selected or row['frame_id'] in known:continue
            image=decode_exact(session['source']['path'],row)
            if pixel_hash(image)!=row['decoded_pixel_sha256']:raise ValueError('Candidate pixels changed')
            g=row['geometry'];item=dict(id=row['frame_id'],session_id=session['id'],source_kind=session['dataset'],source=session['source']['path'],source_sha256=session['source']['sha256'],
                fen=session['fen'],pts_time=row['pts_time'],decoded_pixel_sha256=row['decoded_pixel_sha256'],geometry_sha256=hashlib.sha256(json.dumps(g,sort_keys=True).encode()).hexdigest(),
                geometry_reference=str(next(p for p in folder.glob('candidate_*.json') if read(p)['frame_id']==row['frame_id'])),status='pending')
            try:render(item,image,g,output/'frames'/row['frame_id']);items.append(item)
            except ValueError as e:print('Crop rejection',row['frame_id'],str(e),flush=True)
        print('Rendered',session['id'],flush=True)
    for item in read(ROOT/'photo_candidates_v1/summary.json'):
        if item['status']=='unverified':
            copyitem={k:v for k,v in item.items() if k!='geometry'};copyitem['geometry_reference']=str(ROOT/'photo_candidates_v1'/(item['metadata']['id']+'.json'));copyitem['grid_path']=str(ROOT/'photo_candidates_v1'/item['metadata']['id']/'grid.jpg');copyitem['crops_path']=str(ROOT/'photo_candidates_v1'/item['metadata']['id']/'crops.jpg');items.append(copyitem)
    write_new(output/'packet.json',dict(schema_version='convnext_c9_review_packet_v1',items=items,checks=list(CHECKS),authority='No automatic labels or geometry verification; predictions never shown'))
    write_new(output/'review_template.json',dict(schema_version='convnext_c9_reviews_v1',reviewer='',packet_sha256=digest(output/'packet.json'),items={i['id']:dict(binding=i['binding'],status='pending',checks={c:False for c in CHECKS},ambiguous_squares=[]) for i in items}))
    cards=[]
    symbols='.PNBRQKpnbrqk'
    for i in items:
        if i['source_kind']=='photo':
            from os.path import relpath
            grid=relpath(i['grid_path'],output);crops=relpath(i['crops_path'],output)
        else:grid=f"frames/{i['id']}/grid.jpg";crops=f"frames/{i['id']}/crops.jpg"
        labels=placement_labels(i['fen']);board='<table>'+''.join('<tr>'+''.join('<td style="width:26px;height:26px;background:'+('#d2aa7c' if (r+c)%2 else '#f2dfbf')+'">'+symbols[labels[r*8+c]]+'</td>' for c in range(8))+'</tr>' for r in range(8))+'</table>'
        cards.append('<section><h2>'+html.escape(i['id'])+'</h2><code>'+html.escape(i['fen'])+'</code>'+board+f'<img height="650" src="{grid}"><img width="768" src="{crops}"></section>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>C9 review</title><h1>C9 newly selected training candidates — pending verification</h1><p>Check FEN agreement, unchanged arrangement, White orientation, true corners, canonical labels and all64 crop masks. Purple outlines are target footprints. Mark ambiguous or severely occluded squares separately. Existing approved frames are reused without requesting duplicate review. Nothing here is automatically accepted.</p>'+''.join(cards))
    print('Review packet',len(items),'new candidates',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['photos','packet']);a=p.parse_args()
 photo_candidates() if a.action=='photos' else packet()
