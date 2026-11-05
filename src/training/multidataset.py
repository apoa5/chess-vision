"""C.4 normalized, provenance-preserving boards and B.2 square observations."""
import hashlib
import json
from pathlib import Path
from collections import OrderedDict,defaultdict
import cv2
import numpy as np
from src.chess.labels import CLASS_NAMES,CLASSES
from src.vision.crop_experiments import CropSpec,packed_input,tensor_from_packed
from src.vision.dataset import chess_square_at
from src.vision.grid_geometry import polygon_crop
from src.vision.perspective import warp_board_with_context,order_corners

PROFILE='b2-context1-224-v1'
SQUARES=tuple(f'{f}{r}' for r in range(8,0,-1) for f in 'abcdefgh')
DATASETS={'chesscog','chessred','chessvision_photos','d3c_retired'}
VERIFIED={'human_verified','source_ground_truth','historical_owner_verified'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def placement_labels(fen):
    ranks=fen.split()[0].split('/');labels=[];mapping={c.symbol:c.class_id for c in CLASSES if c.symbol}
    if len(ranks)!=8:raise ValueError('Eight FEN ranks required')
    for rank in ranks:
        row=[]
        for char in rank:
            if char in '12345678':row.extend([0]*int(char))
            elif char in mapping:row.append(mapping[char])
            else:raise ValueError('Unknown placement symbol')
        if len(row)!=8:raise ValueError('Eight squares per rank required')
        labels.extend(row)
    return labels


def position_key(fen):return hashlib.sha256(bytes(placement_labels(fen))).hexdigest()


def board_ready(board):
    return (board['inclusion']['status']=='included' and board['geometry']['status'] in VERIFIED
            and board['annotation']['status'] in VERIFIED and board['split'] in ('train','val','test'))


def validate_board(b):
    required={'id','dataset','source_id','position_id','session_id','source','placement','semantic_orientation','geometry','annotation','split','representation','inclusion','observations'}
    if not required<=set(b):raise ValueError('Missing unified board fields')
    if b['dataset'] not in DATASETS or b['representation']!=PROFILE:raise ValueError('Unapproved source/representation')
    if b.get('final_holdout') or b['dataset'].startswith('a3'):raise ValueError('Holdout/original pilot prohibited')
    if len(b['source']['sha256'])!=64 or any(c not in '0123456789abcdef' for c in b['source']['sha256']):raise ValueError('Source hash required')
    truth=dict(zip(SQUARES,placement_labels(b['placement'])))
    if len(b['observations'])!=64 or {o['square'] for o in b['observations']}!=set(SQUARES):raise ValueError('Exactly 64 canonical observations required')
    if any(o['class_id']!=truth[o['square']] for o in b['observations']):raise ValueError('Labels disagree with ground truth placement')
    if b['inclusion']['status']=='included' and not board_ready(b):raise ValueError('Unverified board cannot be included')
    if board_ready(b) and b['geometry']['kind'] not in ('raw_regular_grid','frozen_context_crop','verified_prepared_context','video_original_pts'):raise ValueError('Unsupported verified geometry')
    if b['geometry']['kind']=='raw_regular_grid':
        if b['geometry'].get('approximation_version')!='b2-annotated-corners-regular-grid-v1':raise ValueError('Explicit external geometry approximation required')
        g=b['geometry'];corners=np.asarray(g.get('corners'),dtype=np.float32)
        if corners.shape!=(4,2) or not np.isfinite(corners).all() or type(g.get('turns')) is not int or g['turns'] not in range(4):raise ValueError('Finite corners and semantic quarter-turn required')
        order_corners(corners)
        if any(o['square']!=chess_square_at(*divmod(o['camera_index'],8),g['turns']) for o in b['observations']):raise ValueError('Canonical labels and camera-square mapping disagree')
    if b['dataset']=='d3c_retired' and board_ready(b):
        if not all(b['annotation'].get('checks',{}).get(c) is True for c in ('fen_agreement','unchanged_position','semantic_orientation','corners','canonical_mapping','crop_alignment','all_crops_reviewed')):
            raise ValueError('Video labels/geometry/crops require explicit review')
    if b['dataset']=='d3c_retired' and b['session_id'] not in {f'd3c_pos_{i:02d}' for i in (1,3,5,7,9,11,13,15,17,19)}:
        raise ValueError('Only ten explicitly retired D3c sessions are authorized')
    return b


def split_audit(boards):
    """Known position/game/session/source connected groups; never random frame splits."""
    if len({b['id'] for b in boards})!=len(boards):raise ValueError('Duplicate board/frame identities')
    parent={i:i for i in range(len(boards))}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    owners={}
    for i,b in enumerate(boards):
        validate_board(b)
        keys=[('placement',position_key(b['placement'])),('source',b['source']['sha256']),('position',b['dataset'],b['position_id'])]
        if b.get('original_source'):keys.append(('source',b['original_source']['sha256']))
        if b.get('game_id') is not None:keys.append(('game',b['dataset'],str(b['game_id'])))
        if b['session_id']:keys.append(('session',b['dataset'],b['session_id']))
        for key in keys:
            if key in owners:parent[find(i)]=find(owners[key])
            else:owners[key]=i
    groups=defaultdict(list)
    for i,b in enumerate(boards):groups[find(i)].append(b)
    conflicts=[]
    for group in groups.values():
        roles={b['split'] for b in group if b['inclusion']['status']=='included'}
        if 'train' in roles and roles.intersection({'val','test'}):conflicts.append([b['id'] for b in group])
    if conflicts:raise ValueError('Cross-source grouped split leakage: '+str(conflicts[:3]))
    return dict(boards=len(boards),connected_groups=len(groups),conflicts=0,
                missing_session_boards=sum(b['session_id'] is None for b in boards))


def quarantine_protected(boards,protected_placements):
    """Preserve frozen roles; quarantine new training conflicts, never reassign them."""
    blocked_games=set();blocked_sessions=set()
    for b in boards:
        if b['split']=='train' and position_key(b['placement']) in protected_placements:
            if b.get('game_id') is not None:blocked_games.add((b['dataset'],b['game_id']))
            if b['session_id']:blocked_sessions.add((b['dataset'],b['session_id']))
            b['inclusion']=dict(status='quarantine',reason='overlaps_protected_personal_validation_or_test')
    for b in boards:
        if b['split']=='train' and ((b['dataset'],b.get('game_id')) in blocked_games or (b['dataset'],b['session_id']) in blocked_sessions):
            b['inclusion']=dict(status='quarantine',reason='related_game_or_session_has_protected_overlap')
    return boards


def regular_observations(fen,turns):
    labels=dict(zip(SQUARES,placement_labels(fen)))
    return [dict(square=chess_square_at(r,c,turns),class_id=labels[chess_square_at(r,c,turns)],camera_index=r*8+c)
            for r in range(8) for c in range(8)]


def chesscog_board(image_path,annotation_path,split):
    a=json.loads(Path(annotation_path).read_text())
    if type(a.get('white_turn')) is not bool:raise ValueError('Published camera-player convention required')
    # Official occupancy crop code explicitly maps white_turn False to Black-near.
    turns=0 if a['white_turn'] else 2
    labels=dict(zip(SQUARES,placement_labels(a['fen'])))
    actual={p['square']:next(c.class_id for c in CLASSES if c.symbol==p['piece']) for p in a['pieces']}
    if actual!={s:k for s,k in labels.items() if k}:raise ValueError('Chesscog piece list and FEN disagree')
    image=cv2.imread(str(image_path))
    if image is None:raise ValueError('Image does not decode')
    corners=order_corners(np.asarray(a['corners'],np.float32))
    if (corners<0).any() or corners[:,0].max()>=image.shape[1] or corners[:,1].max()>=image.shape[0]:raise ValueError('GT corners outside image')
    b=dict(id='chesscog:'+Path(image_path).stem,dataset='chesscog',source_id=Path(image_path).name,
        position_id=position_key(a['fen']),session_id=None,game_id=None,
        source=dict(path=str(image_path),filename=Path(image_path).name,sha256=digest(image_path)),placement=a['fen'].split()[0],original_fen=a['fen'],
        semantic_orientation=['white_near_camera','white_left','black_near_camera','white_right'][turns],split=split,representation=PROFILE,
        geometry=dict(status='source_ground_truth',kind='raw_regular_grid',corners=corners.tolist(),turns=turns,approximation_version='b2-annotated-corners-regular-grid-v1'),
        annotation=dict(status='source_ground_truth',path=str(annotation_path),sha256=digest(annotation_path),provenance='official synthetic FEN + corners + white_turn; cross-checked complete piece list'),
        inclusion=dict(status='included',reason='official_annotations_and_pixel_structure_validated'),observations=regular_observations(a['fen'],turns))
    return validate_board(b)


def chessred_boards(annotation,image_root,verified_geometry,official_game_splits,verified_game_ids=None):
    """Official piece table; geometry must be separately normalized and verified.

    Never assumes every image has corners, numeric category IDs, or a known game
    split. The verified_geometry mapping is explicit canonical corner metadata.
    """
    categories={c['id']:c['name'].replace('-','_').lower() for c in annotation['categories']}
    ids={name:i for i,name in enumerate(CLASS_NAMES)};pieces=defaultdict(list)
    for a in annotation['annotations']['pieces']:pieces[a['image_id']].append(a)
    boards=[];official_roles={}
    for role,info in annotation.get('splits',{}).items():
        role='val' if role=='validation' else role
        for image_id in info['image_ids']:
            if image_id in official_roles:raise ValueError('Duplicate official ChessReD split membership')
            official_roles[image_id]=role
    for im in annotation['images']:
        labels=[0]*64;seen=set()
        for p in pieces[im['id']]:
            square=p['chessboard_position'];name=categories[p['category_id']]
            if square not in SQUARES or square in seen or name not in ids:raise ValueError('Invalid/duplicate ChessReD piece annotation')
            seen.add(square);labels[SQUARES.index(square)]=ids[name]
        from src.training.manifest_tools import labels_to_placement
        placement=labels_to_placement(labels);geometry=verified_geometry.get(str(im['id']))
        game=(verified_game_ids or {}).get(str(im['id']),im.get('game_id'));split=official_game_splits.get(str(game)) if game is not None else None
        filename=im.get('path',im.get('file_name'))
        if not filename:raise ValueError('Official image path required')
        path=Path(image_root)/filename
        if not path.resolve().is_relative_to(Path(image_root).resolve()):raise ValueError('Image path escapes dataset root')
        official_role=official_roles.get(im['id'])
        included=geometry is not None and geometry.get('status') in VERIFIED and split in ('train','val','test') and split==official_role and path.is_file()
        board=dict(id='chessred:'+str(im['id']),dataset='chessred',source_id=str(im['id']),position_id=position_key(placement),session_id=None,game_id=game,
            source=dict(path=str(path),filename=filename,sha256=digest(path) if path.is_file() else '0'*64),placement=placement,
            semantic_orientation=geometry.get('orientation') if geometry else None,split=split or 'quarantine',representation=PROFILE,
            geometry=geometry or dict(status='unresolved',kind='missing'),annotation=dict(status='source_ground_truth',provenance='official pieces/categories; no model-generated labels'),
            inclusion=dict(status='included' if included else 'quarantine',reason='explicit_geometry_and_game_split' if included else 'missing_verified_geometry_game_split_or_image'),
            observations=regular_observations(placement,geometry['turns'] if geometry else 0))
        boards.append(validate_board(board))
    return boards


class MultiDatasetSquares:
    def __init__(self,manifest,split,*,seed=42,augmentation=None,cache_boards=2):
        if manifest['representation']!=PROFILE or manifest.get('final_holdout'):raise ValueError('B2 development manifest required')
        if manifest.get('class_names',list(CLASS_NAMES))!=list(CLASS_NAMES):raise ValueError('Class ordering changed')
        if split not in ('train','val'):raise ValueError('Training interface never opens test/holdout')
        split_audit(manifest['boards']);self.boards=[b for b in manifest['boards'] if b['split']==split and board_ready(b)]
        self.rows=[(b,o) for b in self.boards for o in b['observations'] if o.get('visibility') not in ('ambiguous','severely_occluded')]
        if not self.rows:raise ValueError('No verified eligible observations for '+split)
        self.split,self.seed,self.epoch,self.augmentation=split,seed,0,augmentation or {};self.cache=OrderedDict();self.cache_boards=cache_boards
    def __len__(self):return len(self.rows)
    def set_epoch(self,epoch):self.epoch=epoch
    def observation(self,index):
        b,o=self.rows[index];return dict(b,o=o,target_square=o['square'],piece_class=o['class_id'])
    def packed(self,index):
        b,o=self.rows[index];kind=b['geometry']['kind']
        if kind=='video_original_pts':
            from src.scanning.interim_video import decode_exact,pixel_hash
            from scripts.run_square_evidence import make_inputs
            if b['id'] not in self.cache:
                g=b['geometry'];path=g['candidate_path']
                if digest(path)!=g['candidate_sha256']:raise ValueError('Reviewed video geometry changed')
                row=json.loads(Path(path).read_text())
                image=decode_exact(b['source']['path'],row)
                if pixel_hash(image)!=row['decoded_pixel_sha256']:raise ValueError('Video pixels changed')
                config=dict(board_size=800,context_margin=200,initial_context_padding=200,crop_spec=dict(name='context1_224',input_size=224,padding_squares=1.))
                self.cache[b['id']]={v['canonical_square']:v['packed'] for v in make_inputs(image,row['geometry'],config)}
                while len(self.cache)>self.cache_boards:self.cache.popitem(last=False)
            self.cache.move_to_end(b['id']);result=self.cache[b['id']][o['square']]
            if hashlib.sha256(result.tobytes()).hexdigest()!=o['packed_sha256']:raise ValueError('Reviewed video crop changed')
            return result
        if kind=='frozen_context_crop':
            path=o['crop_path']
            if digest(path)!=o['crop_sha256']:raise ValueError('Changed approved crop')
            image=cv2.imread(path)
            result=packed_input(image,o['target_polygon'],CropSpec('context1_224',224,1.))
        else:
            if b['id'] not in self.cache:
                path=b['source']['path']
                if digest(path)!=b['source']['sha256']:raise ValueError('Source image changed')
                image=cv2.imread(path)
                if kind=='raw_regular_grid':
                    board=warp_board_with_context(image,np.float32(b['geometry']['corners']),800,200)
                    polygons=[[[200+c*100,200+r*100],[300+c*100,200+r*100],[300+c*100,300+r*100],[200+c*100,300+r*100]] for r in range(8) for c in range(8)]
                    payload=(board.image,board.valid_mask,polygons)
                elif kind=='verified_prepared_context':payload=(image,cv2.imread(b['geometry']['valid_mask_path'],0),b['geometry']['context_square_polygons'])
                else:raise ValueError('Unsupported verified geometry')
                self.cache[b['id']]=payload
                while len(self.cache)>self.cache_boards:self.cache.popitem(last=False)
            self.cache.move_to_end(b['id']);image,valid,polygons=self.cache[b['id']]
            crop,_,_,_,poly,coverage=polygon_crop(image,valid,polygons[o['camera_index']],200)
            if coverage<.99:raise ValueError('Target coverage below frozen B2 gate')
            result=packed_input(crop,poly,CropSpec('context1_224',224,1.))
        if o.get('packed_sha256') and hashlib.sha256(result.tobytes()).hexdigest()!=o['packed_sha256']:raise ValueError('B2 packed parity failed')
        return result
    def __getitem__(self,index):
        import torch
        b,o=self.rows[index];rng=None
        if self.split=='train':
            key=f'{self.seed}:{self.epoch}:{b["id"]}:{o["square"]}'.encode();rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8],'big'))
        return torch.from_numpy(tensor_from_packed(self.packed(index),rng=rng,augmentation=self.augmentation)),o['class_id']
