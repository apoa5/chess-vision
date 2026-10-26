"""D.2 geometry gate: deduplicated, hash-bound human-review packet; no inference."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write,check_hashes
from scripts.validate_keyframe_manifest import validate_manifest
from src.scanning.native_pilot import sha256_file
from src.vision.geometry_audit import raw_grid_overlay


def prepare(config_path, output):
    if output.exists(): raise ValueError('Fresh review directory required')
    start=time.perf_counter();config=read(config_path);cache={};frames={};conditions=[]
    for condition in config['conditions']:
        m=read(condition['manifest']);validate_manifest(m,cache)
        if m['dataset']!='primary_3s_high' or m['policy']!='marked-board':raise ValueError('Registered primary HD opt-in inputs only')
        conditions.append(dict(condition,manifest_sha256=sha256_file(condition['manifest'])))
        for s in m['sessions']:
            for f in s['selected']:
                key=(s['capture_session_id'],f['frame_id'])
                if key not in frames:
                    g=read(f['geometry_reference'])['policies'][m['policy']]
                    binding={k:f[k] for k in ('frame_id','pts','time_base','source_video_sha256','derivative_sha256','geometry_reference_sha256')}
                    binding['session_id']=s['capture_session_id']
                    frames[key]=dict(frame=f,session={k:s[k] for k in ('capture_session_id','position_id','position_group_id','independence_group_id','source_video')},
                        binding=binding,geometry=g,conditions=[])
                elif frames[key]['frame']!=f:raise ValueError('Same frame has inconsistent D.1 metadata')
                frames[key]['conditions'].append(condition['name'])
    if sha256_file(config['checkpoint'])!=config['checkpoint_sha256']:raise ValueError('B.2 checkpoint mismatch')
    old=read('outputs/scanning/d1_pilot_run1/candidates.json');protected=dict(old['protected_sha256'])
    paths=[]
    for dirname in ['src','scripts','tests','web','configs','config','data/metadata','docs/evaluation','outputs/scanning/d1_pilot_run1']:
        paths.extend(p for p in Path(dirname).rglob('*') if p.is_file() and not any(t in p.parts for t in ('__pycache__','node_modules','.pytest_cache')))
    paths+=list(Path('outputs/evaluation').glob('*/training/**/*.pt'))
    paths += [Path('VISION_RESEARCH.md'),Path('IMPLEMENTATION_PLAN.md')]
    protected.update({str(p):sha256_file(p) for p in paths})
    check_hashes(protected)
    output.mkdir(parents=True,exist_ok=False);rows=[];pages={};reviews=[]
    for n,((sid,fid),item) in enumerate(sorted(frames.items()),1):
        f=item['frame'];g=item['geometry'];image=cv2.imread(f['path']);polys=np.float32(g['source_square_polygons'])
        overlay=raw_grid_overlay(image,polys,g['semantic_square_ids'])
        q=np.float32([polys[0,0],polys[7,1],polys[63,2],polys[56,3]])
        for k,(x,y) in enumerate(np.rint(q).astype(int)):
            cv2.circle(overlay,(x,y),7,(255,100,0),2)
            cv2.putText(overlay,'C'+str(k),(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,60,0),2,cv2.LINE_AA)
        filename=f'{n:03d}_{fid}.jpg';cv2.imwrite(str(output/filename),overlay,[cv2.IMWRITE_JPEG_QUALITY,95])
        original=f'{n:03d}_original.jpg';cv2.imwrite(str(output/original),image,[cv2.IMWRITE_JPEG_QUALITY,95])
        row=dict(review_id=f'R{n:03d}',binding=item['binding'],frame=f,session=item['session'],conditions=item['conditions'],
                 geometry_completed=True,geometry_status='completed_unverified',human_review_status='pending',
                 proposed_playing_corners_px=q.tolist(),rectification_seed_corners_px=g['provenance']['source_corners'],
                 proposed_semantic_orientation=g['predicted_camera_orientation'],rotation_quarters_ccw=g['rotation_quarters_ccw'],
                 canonical_square_ids=g['semantic_square_ids'],source_square_polygons=g['source_square_polygons'],
                 overlay_path=str(output/filename),overlay_sha256=sha256_file(output/filename))
        rows.append(row)
        reviews.append(dict(review_id=row['review_id'],binding=item['binding'],decision='pending',reviewer=None,reviewer_kind='human',
                            checks=dict(true_board_corners=False,semantic_orientation=False,canonical_square_mapping=False,sufficient_for_crops=False),notes=''))
        text=f"{row['review_id']} — {f['timestamp_ms']/1000:.3f}s — {', '.join(item['conditions'])} — {g['predicted_camera_orientation']}"
        pages.setdefault(sid,[]).append(f'<section id="{row["review_id"]}"><h2>{html.escape(text)}</h2><p>Full image: <a href="{filename}">labelled grid</a> · <a href="{original}">unlabelled original</a>. Blue C0–C3 = fitted playing-area corners (image order); green IDs = proposed canonical cells.</p><img style="max-width:100%;width:1080px" src="{filename}"><p>Check all four true playing-area corners, printed semantic orientation, a1–h8 mapping and target-cell alignment. Reject if any is wrong; unresolved if unsure. Fitted corners are reviewed proposals, not new independent point annotations.</p></section>')
    title='<h1>D.2 geometry review — no classifier predictions</h1><p>Review the full-resolution overlays, not just thumbnails. A human approval attests that the fitted board corners and all canonical cells are sufficiently accurate for crop evaluation. No subpixel corner accuracy is implied. Geometry completion alone is insufficient.</p>'
    links=[]
    for sid,sections in pages.items():
        name=sid+'.html';(output/name).write_text('<!doctype html><meta charset="utf-8">'+title+''.join(sections));links.append(f'<li><a href="{name}">{sid}</a> ({len(sections)} unique frames)</li>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8">'+title+'<p>40 primary-union frames; 15 sensitivity-only additions. Shared frames reviewed once. All 55 start pending. Report R-number exceptions, or explicitly confirm all after review.</p><ul>'+''.join(links)+'</ul>')
    packet=dict(schema_version='offline_evidence_geometry_packet_d2_v1',designation='development_only',conditions=conditions,
                config_path=str(config_path),config_sha256=sha256_file(config_path),review_frame_count=len(rows),
                primary_union_count=sum(any(x in r['conditions'] for x in ('diversity_5','uniform_5')) for r in rows),frames=rows,
                protected_sha256=protected,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                preparation_seconds=time.perf_counter()-start)
    write(output/'packet.json',packet)
    write(output/'reviews_template.json',dict(schema_version='offline_evidence_geometry_reviews_d2_v1',packet_sha256=sha256_file(output/'packet.json'),reviews=reviews))
    print(dict(unique_frames=len(rows),primary_union=packet['primary_union_count'],packet=str(output/'packet.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/experiments/evidence_d2.json'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();prepare(a.config,a.output_dir)
