"""Export frozen C9 source/class/group counts; no recognition or internal-test inference."""
import csv,json,sys
from collections import Counter,defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.dataset_c9 import audit_splits
from src.training.multidataset import digest,position_key
from src.training.manifest_tools import write_new
from src.chess.labels import CLASS_NAMES
ROOT=Path('outputs/evaluation/convnext_c9')

def main():
    m=json.loads((ROOT/'frozen_manifest_v1.json').read_text());inv=json.loads((ROOT/'inventory_v1.json').read_text())
    candidates=json.loads((ROOT/'candidate_summary_v1.json').read_text())
    result=dict(schema_version='convnext_c9_data_summary_v1',manifest_sha256=digest(ROOT/'frozen_manifest_v1.json'),group_audit=audit_splits(m['boards']),splits={})
    for split in ('train','val','internal_test'):
        boards=[b for b in m['boards'] if b['split']==split];usable=[(b,o) for b in boards for o in b['observations'] if o.get('visibility') not in ('ambiguous','severely_occluded')]
        classes=Counter(o['class_id'] for b,o in usable)
        result['splits'][split]=dict(boards=len(boards),physical_positions=len({position_key(b['placement']) for b in boards}),eligible_squares=len(usable),sources=dict(Counter(b['dataset'] for b in boards)),
            class_counts={name:classes[i] for i,name in enumerate(CLASS_NAMES)},by_position={p:dict(boards=sum(position_key(b['placement'])==p for b in boards),source_position_ids=sorted({b['position_id'] for b in boards if position_key(b['placement'])==p})) for p in sorted({position_key(b['placement']) for b in boards})})
    sessions=[]
    for s in inv['sessions']:
        b=[b for b in m['boards'] if b['session_id']==s['id']];c=next(c for c in candidates if c['session_id']==s['id'])
        sessions.append(dict(session_id=s['id'],source=s['source'],candidates=c['candidates'],quality_eligible=c['eligible'],retained_frames=len(b),split=b[0]['split'] if b else None,
            timestamps_ms=[v['original_frame']['timestamp_ms'] for v in b],orientations=dict(Counter(v['semantic_orientation'] for v in b))))
    result['video_sessions']=sessions;result['candidate_frames']=sum(s['candidates'] for s in sessions);result['quality_eligible_candidates']=sum(s['quality_eligible'] for s in sessions)
    result['photographs_inventoried']=len(inv['photos']);result['photographs_included']=sum(b['dataset']=='chessvision_photos' for b in m['boards']);result['video_frames_included']=sum(b['dataset']!='chessvision_photos' for b in m['boards'])
    result['quarantined_photos']=[p['id'] for p in inv['photos'] if not any(b['dataset']=='chessvision_photos' and b['source_id']==p['id'] for b in m['boards'])]
    result['independent_generalization']=False;write_new(ROOT/'data_summary_v1.json',result)
    path=Path('docs/evaluation/convnext_c9_sources_v1.csv')
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['kind','source_id','source_path','source_sha256','position_id','session_id','timestamp_ms','historical_split','c9_split','geometry_status','inclusion']);w.writeheader()
        for p in inv['photos']:
            b=next((b for b in m['boards'] if b['dataset']=='chessvision_photos' and b['source_id']==p['id']),None)
            w.writerow(dict(kind='photo',source_id=p['id'],source_path=p['source_path'],source_sha256=p['sha256'],position_id=p['position_id'],session_id=p['session_id'],historical_split=p['historical_split'],c9_split=b['split'] if b else '',geometry_status=b['geometry']['status'] if b else 'unresolved',inclusion='included' if b else 'quarantined'))
        for b in m['boards']:
            if b['dataset']=='chessvision_photos':continue
            w.writerow(dict(kind=b['dataset'],source_id=b['source_id'],source_path=b['original_source']['path'],source_sha256=b['original_source']['sha256'],position_id=b['position_id'],session_id=b['session_id'],timestamp_ms=b['original_frame']['timestamp_ms'],historical_split=b.get('historical_split'),c9_split=b['split'],geometry_status=b['geometry']['status'],inclusion='included'))
    print(json.dumps({k:result[k] for k in ('photographs_inventoried','photographs_included','video_frames_included','candidate_frames','quality_eligible_candidates','quarantined_photos')},indent=2))

if __name__=='__main__':main()
