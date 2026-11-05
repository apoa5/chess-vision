"""Freeze label/source-only C9 connected partitions before recognition outcomes."""
import hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.training.manifest_tools import read,write_new
from src.training.dataset_c9 import connected_groups
from src.training.multidataset import position_key,placement_labels
ROOT=Path('outputs/evaluation/convnext_c9')

def main():
    inventory=read(ROOT/'inventory_v1.json');skeleton=[]
    for p in inventory['photos']:
        if p['status'] in ('excluded_final_holdout','missing'):continue
        skeleton.append(dict(id='photo:'+p['id'],dataset='chessvision_photos',position_id=p['position_id'],placement=p['fen'].split()[0],session_id=p['session_id'],source=dict(sha256=p['sha256'])))
    for s in inventory['sessions']:
        if not s['available']:continue
        skeleton.append(dict(id=s['id'],dataset=s['dataset'],position_id=s['position_id'],placement=s['fen'].split()[0],session_id=s['id'],independence_group_id=s['independence_group_id'],source=s['source']))
    groups=connected_groups(skeleton);groups=sorted(groups,key=lambda g:min(b['id'] for b in g));rng=np.random.default_rng(42)
    forced_train={i for i,g in enumerate(groups) if any(b['dataset']=='a3_pilot' for b in g)}
    val_seed=next(i for i,g in enumerate(groups) if any(b.get('session_id')=='d3c_pos_17' for b in g));test_seed=next(i for i,g in enumerate(groups) if any(b.get('session_id')=='d3c_pos_19' for b in g))
    n_eval=4;free=[i for i in range(len(groups)) if i not in forced_train|{val_seed,test_seed}];attempts=[]
    for attempt in range(128):
        order=list(rng.permutation(free));val={val_seed,*order[:n_eval-1]};test={test_seed,*order[n_eval-1:2*(n_eval-1)]};train=set(range(len(groups)))-val-test
        split={i:'train' if i in train else 'val' if i in val else 'internal_test' for i in range(len(groups))}
        good=True;counts={}
        for role in ['train','val','internal_test']:
            boards=[b for i,g in enumerate(groups) if split[i]==role for b in g];placements={position_key(b['placement']):b['placement'] for b in boards}
            classes=np.bincount([k for fen in placements.values() for k in placement_labels(fen)],minlength=13)
            sources={b['dataset'] for b in boards};counts[role]=dict(positions=len(placements),source_kinds=sorted(sources),unique_position_class_counts=classes.tolist())
            presence=np.sum([np.bincount(placement_labels(fen),minlength=13)>0 for fen in placements.values()],axis=0)
            rare_complete=role=='train' or all(presence[k]>=2 for k in (2,3,4,5,8,9,10,11))
            good &= bool((classes>0).all()) and rare_complete and 'chessvision_photos' in sources and 'd3c_retired' in sources
        if good:
            attempts.append((attempt,split,counts));break
    if not attempts:raise ValueError('No label/source-complete grouping; do not relax silently')
    attempt,split,counts=attempts[0];assignment={position_key(b['placement']):split[i] for i,g in enumerate(groups) for b in g}
    out=dict(schema_version='convnext_c9_split_v2',supersedes='data/metadata/convnext_c9_split_v1.json',revision_reason='Label-only review: require rare occupied classes in at least2 distinct validation/test positions; v1 never used for model fitting or inference.',seed=42,inventory_sha256=hashlib.sha256((ROOT/'inventory_v1.json').read_bytes()).hexdigest(),assignment_by_placement=assignment,
        groups=[dict(group_id=f'C9G{i:02d}',split=split[i],source_identities=[b['id'] for b in g],placement_keys=sorted({position_key(b['placement']) for b in g})) for i,g in enumerate(groups)],counts=counts,
        policy='128-attempt maximum; first label/source-complete seeded assignment. Four connected groups each val/internal-test. Force shared A3 capture cohort to train; D3c17 pawn endgame val and19 sparse endgame test for position-type coverage. No predictions used.',
        attempts_until_valid=attempt+1,internal_test_kind='retrospective_previously_exposed_positions',final_holdout_access=False)
    write_new('data/metadata/convnext_c9_split_v2.json',out);print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
