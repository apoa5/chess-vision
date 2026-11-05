"""Future compact approved-data export. No original videos are copied.

Run only after explicit video review, and upload this derived directory with
`modal volume put`. Excluded/quarantined and test sources are metadata-only.
"""
import argparse,copy,hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,numpy as np
from src.training.manifest_tools import read,write_new
from src.training.multidataset import PROFILE,board_ready,digest,split_audit,MultiDatasetSquares
from src.chess.labels import CLASS_NAMES
from src.scanning.interim_video import decode_exact,pixel_hash
from src.vision.perspective import warp_board_with_context
from src.vision.geometry_audit import context_transform,project

def export(paths,output,require_video=False):
    boards=[b for path in paths for b in read(path)['boards']];audit=split_audit(boards)
    eligible=[b for b in boards if board_ready(b) and b['split'] in ('train','val')]
    if require_video and not any(b['dataset']=='d3c_retired' for b in eligible):raise ValueError('No human-verified non-overlap video frames; E is blocked')
    if output.exists():raise ValueError('Fresh export directory required')
    output.mkdir(parents=True);prepared={b['image_id']:b for b in read('data/classification/prepared_boards.json')};exported=[]
    for original in eligible:
        b=copy.deepcopy(original);key=hashlib.sha256(b['id'].encode()).hexdigest()[:20]
        if b['geometry']['kind']=='frozen_context_crop':
            p=prepared[b['source_id']];image=cv2.imread(p['context_path']);mask=cv2.imread(p['mask_path'],0);polygons=p['grid_geometry']['square_polygons']
        elif b['geometry']['kind']=='video_original_pts':
            g=b['geometry']
            if digest(g['candidate_path'])!=g['candidate_sha256']:raise ValueError('Reviewed geometry changed')
            row=read(g['candidate_path']);raw=decode_exact(b['source']['path'],row)
            if pixel_hash(raw)!=row['decoded_pixel_sha256']:raise ValueError('Reviewed pixels changed')
            g=row['geometry'];corners=g['provenance']['source_corners'];context=warp_board_with_context(raw,np.float32(corners),800,200)
            image,mask=context.image,context.valid_mask;polygons=project(g['source_square_polygons'],context_transform(corners,context.board_bounds)).tolist()
            b['original_frame']=dict(pts_time=row['pts_time'],timestamp_ms=row['timestamp_ms'],decoded_pixel_sha256=row['decoded_pixel_sha256'])
        else:raise ValueError('This exporter accepts approved personal photos/video only')
        folder=output/'contexts';folder.mkdir(exist_ok=True);ip=folder/(key+'.png');mp=folder/(key+'_valid.png')
        if not cv2.imwrite(str(ip),image) or not cv2.imwrite(str(mp),mask):raise ValueError('Export write failed')
        b['original_source']=dict(b['source']);b['original_geometry']=copy.deepcopy(b['geometry']);b['source']=dict(path=str(ip),filename=ip.name,sha256=digest(ip))
        b['geometry'].update(kind='verified_prepared_context',valid_mask_path=str(mp),context_square_polygons=polygons)
        exported.append(b)
    local=dict(schema_version='multidataset_c4_v1',representation=PROFILE,class_names=list(CLASS_NAMES),final_holdout=False,boards=exported)
    # Verify ALL packed pixels against frozen B2 / human-reviewed crop hashes.
    for split in ('train','val'):
        ds=MultiDatasetSquares(local,split)
        for i in range(len(ds)):ds.packed(i)
    for b in exported:
        b['source']['path']=str(Path(b['source']['path']).relative_to(output));b['geometry']['valid_mask_path']=str(Path(b['geometry']['valid_mask_path']).relative_to(output))
    local['source_split_audit']=audit;write_new(output/'manifest.json',local)
    init=Path('models/cache/hub/checkpoints/mobilenet_v3_small-047dcff4.pth');dest=output/'torch/hub/checkpoints'/init.name;dest.parent.mkdir(parents=True);shutil.copyfile(init,dest)
    write_new(output/'bundle_inventory.json',{str(p.relative_to(output)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()})
    return dict(boards=len(exported),observations=sum(len(b['observations']) for b in exported),original_videos_copied=0)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--require-video',action='store_true');a=p.parse_args();print(export(a.manifest,a.output,a.require_video))
