"""C.7 direct Chesscog transfer contract; historical C.4 contracts stay unchanged."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.chess.labels import CLASS_NAMES
from src.training.multidataset import PROFILE, MultiDatasetSquares, board_ready, digest, split_audit

PARENT_SHA = '11c54e55cd77ffa9fd8530dc75283a5d77953bbb66f0018b8a1f9deca049f1c9'
BEST_NAMES = {'B': 'chesscog_personal_photos_best.pt', 'C': 'chesscog_personal_multiview_best.pt'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_parent(path, state):
    if digest(path) != PARENT_SHA:
        raise ValueError('C.7 requires the exact C.5 best checkpoint hash')
    if (state.get('stage') != 'chesscog' or state.get('smoke_only') or
        state.get('architecture') != 'mobilenet_v3_small' or state.get('input_channels') != 4 or
        state.get('input_size') != 224 or state.get('class_names') != list(CLASS_NAMES)):
        raise ValueError('C.7 parent model contract mismatch')


def branch_plan(config, branch):
    if (branch not in BEST_NAMES or config.get('schema_version') != 'personal_direct_c7_v1' or
        not config.get('full_training_authorization') or config.get('parent_sha256') != PARENT_SHA or
        config.get('representation') != PROFILE):
        raise ValueError('Explicit C.7 branch authorization/representation/parent required')
    p = copy.deepcopy(config)
    p['c7_branch'] = branch
    # C4's existing B transfer policy already requires a Chesscog parent. Reuse
    # that policy in both branches; this is not a C4 C/D/E experiment or a
    # fictional ChessReD checkpoint. Scientific identity is c7_branch + schema.
    p['experiment'] = 'B'
    s = p['stages']['personal']
    if (s['max_updates'] != 672 or s['validation_every_updates'] != 112 or
        s['epochs'] != 6 or s['max_seconds'] > 1800 or s['batch_size'] != 64 or
        s['gradient_accumulation'] != 1 or s.get('patience', 0) < 6):
        raise ValueError('Matched C.7 update/validation/time budget changed')
    s.update(video_fraction=0. if branch == 'B' else .5, best_name=BEST_NAMES[branch],
             required_dataset=['chessvision_photos'] if branch == 'B' else ['chessvision_photos', 'd3c_retired'])
    return p


def branch_manifest(manifest, branch, *, enforce_counts=True):
    if branch not in BEST_NAMES:
        raise ValueError('Unknown C.7 branch')
    m = copy.deepcopy(manifest)
    if m.get('final_holdout') or m.get('representation') != PROFILE:
        raise ValueError('Only B.2 development inputs allowed')
    if any(not board_ready(b) or b['split'] not in ('train', 'val') or
           b['dataset'] not in ('chessvision_photos', 'd3c_retired') for b in m['boards']):
        raise ValueError('Unverified/excluded/test/external input in C.7 bundle')
    if any(b['dataset'] == 'd3c_retired' and (b['split'] != 'train' or b['session_id'] == 'd3c_pos_03') for b in m['boards']):
        raise ValueError('Protected overlap or video validation prohibited')
    counts = Counter((b['dataset'], b['split']) for b in m['boards'])
    if enforce_counts and counts != Counter({('chessvision_photos', 'train'): 112, ('chessvision_photos', 'val'): 25, ('d3c_retired', 'train'): 77}):
        raise ValueError('Approved C.7 board counts changed')
    if branch == 'B':
        m['boards'] = [b for b in m['boards'] if b['dataset'] == 'chessvision_photos']
    split_audit(m['boards'])
    return m


def exposure_report(manifest, plan):
    from src.training.staged import BoardBatches
    ds = MultiDatasetSquares(manifest, 'train')
    s = plan['stages']['personal']; sources = Counter(); positions = Counter(); classes = Counter(); frames = Counter()
    for epoch in range(s['epochs']):
        batches = BoardBatches(ds, s['batch_size'], plan['seed'], epoch,
            video_fraction=s['video_fraction'], position_balanced=True, max_batches=s['validation_every_updates'])
        for indices in batches:
            for i in indices:
                b, o = ds.rows[i]; sources[b['dataset']] += 1; frames[b['id']] += 1
                positions[b['dataset'] + ':' + b['position_id']] += 1; classes[o['class_id']] += 1
    return {'planned_observations_by_source': dict(sources), 'planned_observations_by_position': dict(positions),
            'planned_class_exposure': [classes[i] for i in range(13)],
            'unique_class_counts': [sum(o['class_id'] == i for b, o in ds.rows) for i in range(13)],
            'planned_observations_by_board': dict(frames),
            'eligible_video_frames': sum(b['dataset']=='d3c_retired' for b in ds.boards),
            'distinct_video_frames_drawn': sum(b['dataset']=='d3c_retired' and b['id'] in frames for b in ds.boards),
            'undrawn_board_ids': [b['id'] for b in ds.boards if b['id'] not in frames],
            'photograph_class_weights_only': True, 'sampler': 'unchanged C4 position-balanced source/position/frame draws; 112 board batches/interval'}


def bundle_batches(root, batch_size=4):
    """Small deterministic upload batches; publish manifest/inventory last."""
    root=Path(root)
    if batch_size < 1 or batch_size > 4: raise ValueError('Bounded upload batches required')
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name not in ('manifest.json','bundle_inventory.json'))
    if any(p.suffix.lower() in ('.mov','.mp4') for p in files): raise ValueError('Original media upload prohibited')
    groups=[files[i:i+batch_size] for i in range(0,len(files),batch_size)]
    groups.append([root/'manifest.json',root/'bundle_inventory.json'])
    return groups
