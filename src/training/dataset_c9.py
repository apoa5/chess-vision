"""C9-only reviewed sources and connected retrospective splits; frozen B2 packer reused."""
import copy,hashlib,json,math
from collections import OrderedDict,defaultdict,Counter
from pathlib import Path
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import MultiDatasetSquares,PROFILE,SQUARES,placement_labels,position_key

SOURCES={'chessvision_photos','a3_pilot','d3c_retired'}
REVIEWED={'historical_owner_verified','human_verified'}

def validate_board(b):
    if b.get('final_holdout') or b['representation']!=PROFILE or b['dataset'] not in SOURCES:raise ValueError('Forbidden C9 source/profile')
    if b['geometry']['status'] not in REVIEWED or b['annotation']['status'] not in REVIEWED:raise ValueError('Explicit reviewed geometry and labels required')
    if b['inclusion']['status']!='included':raise ValueError('Quarantined example')
    truth=dict(zip(SQUARES,placement_labels(b['placement'])))
    if len(b['observations'])!=64 or {o['square'] for o in b['observations']}!=set(SQUARES):raise ValueError('One observation per canonical square required')
    if any(o['class_id']!=truth[o['square']] for o in b['observations']):raise ValueError('FEN/square label mismatch')
    if len(b['source']['sha256'])!=64 or b['geometry']['kind'] not in ('frozen_context_crop','verified_prepared_context'):raise ValueError('Source hash and compatible context required')
    if b['dataset']!='chessvision_photos' and not b.get('review_binding_verified'):raise ValueError('Video review binding missing')
    return b

def connected_groups(boards):
    parent=list(range(len(boards)));owners={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,b in enumerate(boards):
        keys=[('placement',position_key(b['placement'])),('position',b['dataset'],b['position_id']),('source',(b.get('original_source') or b['source'])['sha256'])]
        for field in ('session_id','independence_group_id','near_duplicate_group'):
            if b.get(field):keys.append((field,b[field]))
        for key in keys:
            if key in owners:parent[find(i)]=find(owners[key])
            else:owners[key]=i
    groups=defaultdict(list)
    for i,b in enumerate(boards):groups[find(i)].append(b)
    return list(groups.values())

def audit_splits(boards):
    if len({b['id'] for b in boards})!=len(boards):raise ValueError('Duplicate frame identity')
    for group in connected_groups(boards):
        if len({b['split'] for b in group})>1:raise ValueError('Position/session/source leakage')
    return dict(connected_groups=len(connected_groups(boards)),physical_positions=len({position_key(b['placement']) for b in boards}),conflicts=0)

class C9Squares(MultiDatasetSquares):
    """Only initialization differs: new source namespace/3-way split gates.

    Inherits unchanged packed RGB/mask extraction and normalization. C4's own
    Dataset and split restrictions are not modified or bypassed for old tasks.
    """
    def __init__(self,manifest,split,*,allow_internal_test=False,seed=42,augmentation=None,cache_boards=2):
        if manifest.get('final_holdout') or manifest['representation']!=PROFILE or manifest['class_names']!=list(CLASS_NAMES):raise ValueError('C9 B2 development manifest required')
        if split not in ('train','val','internal_test') or (split=='internal_test' and not allow_internal_test):raise ValueError('Internal test needs explicit once-only evaluation gate')
        for b in manifest['boards']:validate_board(b)
        audit_splits(manifest['boards'])
        self.boards=[b for b in manifest['boards'] if b['split']==split]
        self.rows=[(b,o) for b in self.boards for o in b['observations'] if o.get('visibility') not in ('ambiguous','severely_occluded')]
        if not self.rows:raise ValueError('Empty C9 split')
        self.split,self.seed,self.epoch,self.augmentation=split,seed,0,augmentation or {};self.cache=OrderedDict();self.cache_boards=cache_boards


    def __getitem__(self,index):
        import torch
        from src.vision.crop_experiments import tensor_from_packed
        if index<0:raise IndexError('Negative observation')
        repeat,index=divmod(int(index),len(self.rows));b,o=self.rows[index];rng=None
        if self.split=='train':
            key=f'{self.seed}:{self.epoch}:{b["id"]}:{o["square"]}:{repeat}'.encode()
            rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'big'))
        return torch.from_numpy(tensor_from_packed(self.packed(index),rng=rng,augmentation=self.augmentation)),o['class_id']


def coverage_batches(ds,batch_size,seed,epoch):
    """Visit every eligible board/square once; round-robin positions and sources.

    No video sampling replaces scheduled photo exposure. No replacement within
    a source pass. Complete photo passes repeat when needed to cap video share at50%. Correlated video frames remain labeled as correlated sessions.
    """
    if batch_size not in (8,16,32,64):raise ValueError('Bounded batch size required')
    rng=np.random.default_rng(seed+epoch);groups=defaultdict(lambda:defaultdict(list));rows=defaultdict(list)
    for i,(b,o) in enumerate(ds.rows):rows[b['id']].append(i)
    photos=sum(b['dataset']=='chessvision_photos' for b in ds.boards);videos=len(ds.boards)-photos
    photo_passes=max(1,math.ceil(videos/max(1,photos)))
    for b in ds.boards:
        repeats=photo_passes if b['dataset']=='chessvision_photos' else 1
        for repeat in range(repeats):groups[b['dataset']][position_key(b['placement'])].append((b['id'],repeat))
    for pool in groups.values():
        for queue in pool.values():rng.shuffle(queue)
    sources=sorted(groups);rng.shuffle(sources);order=[]
    while any(groups.values()):
        for source in sources:
            pool=groups[source]
            if not pool:continue
            positions=list(pool);rng.shuffle(positions)
            # One board per position per round; each source gets its own turn.
            for pos in positions:
                order.append(pool[pos].pop())
                if not pool[pos]:del pool[pos]
    batches=[]
    for board,repeat in order:
        indices=np.array(rows[board])+repeat*len(ds.rows);rng.shuffle(indices)
        batches.extend(indices[i:i+batch_size].tolist() for i in range(0,len(indices),batch_size))
    return batches
