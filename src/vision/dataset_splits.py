"""Reproducible position-group splits and training-only imbalance statistics."""
import hashlib
import json
import numpy as np
from src.chess.labels import CLASSES

SPLITS = ('train', 'val', 'test')


def split_dataset(manifest, *, seed=42, fractions=(.7,.15,.15), position_fens=None, source_hashes=None, train_only_positions=(), trials=4000):
    rows = manifest['results']
    if not rows or manifest.get('preview_only'):
        raise ValueError('An approved, nonempty crop manifest is required.')
    if any(r.get('geometry_review_required') or r.get('quality_flags') for r in rows):
        raise ValueError('Dataset contains unapproved geometry or flagged targets.')
    fractions = np.asarray(fractions,dtype=float)
    if fractions.shape!=(3,) or np.any(fractions<=0) or not np.isclose(fractions.sum(),1):
        raise ValueError('Three positive split fractions must sum to one.')
    positions = sorted({r['position_id'] for r in rows})
    # Identical piece placements stay together even when registered under different IDs.
    parent = {p:p for p in positions}
    def root(p):
        while parent[p]!=p:
            parent[p]=parent[parent[p]];p=parent[p]
        return p
    seen = {}
    for row in rows:
        position=row['position_id']
        keys=[('placement',position_fens[position].split()[0])] if position_fens else []
        if source_hashes:
            keys.append(('source_pixels',source_hashes[row['source_image']]))
        for key in keys:
            if key in seen:
                parent[root(position)]=root(seen[key])
            else:
                seen[key]=position
    groups = {}
    for position in positions:
        groups.setdefault(root(position),[]).append(position)
    group_positions = sorted(groups.values(),key=lambda x:x[0])
    n = len(group_positions)
    if n<3:
        raise ValueError('At least three independent position groups are needed.')
    val_n,test_n = max(1,round(n*fractions[1])),max(1,round(n*fractions[2]))
    train_n = n-val_n-test_n
    if train_n<1:
        raise ValueError('Split fractions leave no training positions.')
    sizes = (train_n,val_n,test_n)
    counts = np.zeros((n,13),dtype=np.int64)
    group_for = {position:i for i,g in enumerate(group_positions) for position in g}
    source_position = {}
    ids = set()
    for row in rows:
        if row['crop_id'] in ids:
            raise ValueError('Duplicate crop ID.')
        ids.add(row['crop_id'])
        source = row['source_image'];position = row['position_id']
        if source in source_position and source_position[source]!=position:
            raise ValueError('A source image belongs to multiple positions.')
        source_position[source]=position
        counts[group_for[position],row['class_id']]+=1
    unknown = set(train_only_positions)-set(positions)
    if unknown:
        raise ValueError('Training-only positions are absent from the dataset.')
    forced = {group_for[p] for p in train_only_positions}
    if len(forced)>train_n:
        raise ValueError('Too many training-only groups for requested split.')
    total = counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(trials):
        order = rng.permutation(n)
        parts = np.split(order,[train_n,train_n+val_n])
        if not forced.issubset(set(parts[0])):
            continue
        distribution = np.stack([counts[p].sum(axis=0) for p in parts])
        missing = int(((distribution==0)&(total>0)).sum())
        score = missing*1000 + float(np.square(distribution/np.maximum(total,1)-fractions[:,None]).mean())
        if best is None or score<best[0]:
            best=(score,parts,distribution)
    if best is None:
        raise ValueError('No eligible split found; increase search trials.')
    _,parts,distribution = best
    assignment = {position:split for split,part in zip(SPLITS,parts) for i in part for position in group_positions[i]}
    results = [{**r,'split':assignment[r['position_id']]} for r in rows]
    train = distribution[0].astype(float)
    # Square-root inverse frequency avoids extreme weighting of scarce classes.
    weights = np.sqrt(train.sum()/np.maximum(train,1))
    weights[train==0]=0
    weights /= weights[train>0].mean()
    identity=[{k:r.get(k) for k in ('crop_id','crop_sha256','board_review_signature','class_id','position_id','source_image')} for r in sorted(rows,key=lambda r:r['crop_id'])]
    version=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    report = {'dataset_fingerprint':version,'train_only_positions':sorted(train_only_positions),'seed':seed,'requested_fractions':fractions.tolist(),'search_trials':trials,
              'grouping':'position_id; identical FEN placements and duplicate source-photo hashes grouped together',
              'position_assignment':dict(sorted(assignment.items())),
              'position_group_counts':dict(zip(SPLITS,sizes)),
              'splits':{split:{'positions':sorted(p for p,s in assignment.items() if s==split),
                               'source_image_count':len({r['source_image'] for r in results if r['split']==split}),
                               'crop_count':int(distribution[i].sum()),
                               'class_counts':{label.name:int(distribution[i,label.class_id]) for label in CLASSES}}
                        for i,split in enumerate(SPLITS)},
              'missing_classes':{split:[label.name for label in CLASSES if distribution[i,label.class_id]==0]
                                 for i,split in enumerate(SPLITS)},
              'imbalance_strategy':'Square-root inverse-frequency class-weighted loss; weights computed only from training counts. No duplicate oversampling.',
              'class_weights':{label.name:float(weights[label.class_id]) for label in CLASSES}}
    return {**manifest,'results':results,'split_status':'position_grouped','training_ready':not report['missing_classes']['train'],
            'split_report':report},report
