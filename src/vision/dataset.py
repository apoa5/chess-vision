"""Build FEN-labeled context crops with review gates; no training or split assignment."""

from dataclasses import asdict
from collections import Counter
import hashlib
import json
from pathlib import Path

import chess
import cv2
import numpy as np

from src.chess.labels import CLASSES, LABEL_VERSION, class_for_symbol
from src.utils.image_io import load_image, save_image
from src.vision.board_detection import BoardDetection, DetectionSettings, detect_board_corners
from src.vision.perspective import order_corners
from src.vision.orientation import TEMPLATE_PATH, OrientationSettings, resolve_orientation, rotate_rectified
from src.vision.pipeline import rectify_with_config
from src.vision.perspective import RectifiedBoard
from src.vision.grid_geometry import fit_grid, polygon_crop


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_if_changed(path, image):
    """Avoid re-encoding identical large crops on a reviewed dataset rebuild."""
    path = Path(path)
    previous = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
    if previous is not None and np.array_equal(previous, image):
        return path
    return save_image(path, image)


def labels_from_fen(fen, orientation_rotation=0):
    """Map FEN into the camera view; rotation is the CCW turn to White-bottom."""
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise ValueError('Ground-truth FEN must have six fields.')
    if type(orientation_rotation) is not int or orientation_rotation not in range(4):
        raise ValueError('Orientation rotation must be 0, 1, 2, or 3.')
    board = chess.Board(fen)
    labels = [[class_for_symbol(board.piece_at(chess.square(col, 7-row)).symbol()
                             if board.piece_at(chess.square(col, 7-row)) else None)
             for col in range(8)] for row in range(8)]
    return np.rot90(np.array(labels, dtype=object), -orientation_rotation).tolist()


def chess_square_at(row, col, orientation_rotation=0):
    squares = np.array([[f'{chr(97+c)}{8-r}' for c in range(8)] for r in range(8)])
    return str(np.rot90(squares, -orientation_rotation)[row, col])


def board_review_signature(record):
    return fingerprint({key: record.get(key) for key in
                        ('source_sha256', 'context_sha256', 'mask_sha256', 'fen', 'board_bounds', 'orientation_rotation',
                         'label_rotation', 'coordinate_system', 'grid_geometry')})


def review_approved(record, review):
    return bool(record.get('review_signature') and review and review.get('signature') == record['review_signature']
                and review.get('alignment_correct') is True and review.get('labels_correct') is True)


def apply_visibility_reviews(manifest, reviews):
    """Bind visual judgments to crop bytes and the reviewed board/FEN version."""
    allowed = {'sufficient', 'partial', 'severely_occluded', 'clipped'}
    for row in manifest['results']:
        row['visibility_status'] = 'unreviewed'
        review = reviews.get(row['crop_id'], {})
        if (review.get('crop_sha256') == row['crop_sha256']
                and review.get('board_review_signature') == row['board_review_signature']
                and review.get('status') in allowed):
            row['visibility_status'] = review['status']
        row['visual_review_approved'] = (row['visibility_status'] in {'sufficient', 'partial'}
                                         and not row['quality_flags'])
    manifest['visibility_counts'] = {status: sum(r['visibility_status'] == status for r in manifest['results'])
                                     for status in ['unreviewed', *sorted(allowed)]}
    return manifest


def dataset_summary(manifest, records):
    """Compact, versionable results without embedding thousands of crop records."""
    results = manifest['results']
    reviewed = [r for r in results if r['visibility_status'] != 'unreviewed']
    board_ids = {r['source_image'] for r in results}
    return {'source_image_count': len(records),
            'prepared_board_candidate_count': sum(r['board_detected'] for r in records),
            'candidate_count_note': 'Includes manual corner overrides; this is not automatic detector accuracy.',
            'accepted_board_count': len(board_ids), 'excluded_board_count': len(records)-len(board_ids),
            'accepted_position_count': len({r['position_id'] for r in results}),
            'positions_without_accepted_boards': sorted({r['position_id'] for r in records}-{r['position_id'] for r in results}),
            'crop_count': len(results), 'class_counts': manifest['class_counts'],
            'accepted_board_orientation_sources': dict(Counter(r['orientation_source'] for r in records if r['image_id'] in board_ids)),
            'accepted_board_localization_sources': dict(Counter(r.get('localization_source','automatic')
                                                              for r in records if r['image_id'] in board_ids)),
            'coordinate_system': 'camera_view',
            'accepted_board_white_sides': dict(Counter(('bottom','left','top','right')[r.get('label_rotation',0)]
                                                    for r in records if r['image_id'] in board_ids)),
            'visibility_counts': manifest['visibility_counts'],
            'reviewed_examples_by_class': dict(Counter(r['class_name'] for r in reviewed)),
            'target_coverage_flag_count': sum(bool(r['quality_flags']) for r in results),
            'crop_size': manifest['crop_size'],
            'crop_padding_pixels': manifest['crop_padding_pixels'],
            'split_status': manifest['split_status'], 'training_ready': False,
            'assessment': 'Context retains more visible piece structure; viability is qualitative until held-out classifier evaluation. Occlusion and localization failures remain.',
            'review_scope': 'Accepted boards have matching geometry/label reviews. Only crops listed in visibility reviews were inspected individually.'}


def save_label_preview(record):
    board = RectifiedBoard(load_image(record['context_path']), None, tuple(record['board_bounds']))
    grid = board.image.copy()
    labels = labels_from_fen(record['fen'], record.get('label_rotation', 0))
    polygons = record['grid_geometry']['square_polygons']
    rejected_fit = record['grid_geometry'].get('fit_status') == 'rejected_seed_preview_only'
    for index, polygon in enumerate(polygons):
        row, col = divmod(index,8)
        vertices = np.rint(polygon).astype(np.int32)
        cv2.polylines(grid,[vertices],True,(0,0,255) if rejected_fit else (0,255,0),2,cv2.LINE_AA)
        centre = np.mean(vertices,axis=0).astype(int)
        text = chess_square_at(row,col,record.get('label_rotation',0))+':'+(labels[row][col].symbol or '.')
        origin = (int(centre[0])-24,int(centre[1])+6)
        cv2.putText(grid,text,origin,cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),4,cv2.LINE_AA)
        cv2.putText(grid,text,origin,cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(grid,'GEOMETRY REVIEW REQUIRED'+(' - FIT REJECTED' if rejected_fit else ''),(15,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2,cv2.LINE_AA)
    path = Path(record['context_path'])
    save_image(path.with_name(Path(record['image_id']).stem+'_labels.png'), grid)


def update_grid_geometry(record, source_path):
    if not record.get('context_path'):
        return
    version = file_hash(Path(__file__).with_name('grid_geometry.py'))
    if record.get('grid_code_sha256') == version:
        return
    record['grid_geometry'] = fit_grid(load_image(record['context_path']),record['board_bounds'], light_top_left=record.get('label_rotation',0)%2==0)
    record['grid_code_sha256'] = version
    record['review_signature'] = board_review_signature(record)
    save_label_preview(record)
    # The same polygons on the original photo expose warp/corner mistakes.
    source = load_image(source_path)
    override_path = Path('data/metadata/board_corner_overrides.json')
    override = json.loads(override_path.read_text()).get(record['image_id']) if override_path.exists() else None
    # Store the exact homography seed for automatic and manually located boards.
    corners = record.get('source_corners') or (override['corners'] if override else None)
    if corners is None:
        # Recover the existing warp from image features, without rerunning detection.
        finder = cv2.SIFT_create(nfeatures=1200)
        context = load_image(record['context_path'])
        ka,da = finder.detectAndCompute(cv2.cvtColor(context,cv2.COLOR_BGR2GRAY),None)
        kb,db = finder.detectAndCompute(cv2.cvtColor(source,cv2.COLOR_BGR2GRAY),None)
        if da is not None and db is not None:
            matches = cv2.BFMatcher().knnMatch(da,db,k=2)
            good = [m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance < .7*n.distance]
            if len(good)>=12:
                mapping,inliers = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                                                     np.float32([kb[m.trainIdx].pt for m in good]),cv2.RANSAC,2.)
                if mapping is not None and int(inliers.sum())>=12:
                    x0,y0,x1,y1 = record['board_bounds']
                    corners = cv2.perspectiveTransform(np.float32([[[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]]]),mapping)[0].tolist()
                    record['source_corners'] = corners
    if corners is not None:
        x0,y0,x1,y1 = record['board_bounds']
        destination = np.float32([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]])
        transform = cv2.getPerspectiveTransform(destination,np.float32(corners))
        polygons = np.float32(record['grid_geometry']['square_polygons'])
        raw = cv2.perspectiveTransform(polygons.reshape(-1,1,2),transform).reshape(64,4,2)
        record['grid_geometry']['source_square_polygons'] = raw.tolist()
        record['review_signature'] = board_review_signature(record)
        for polygon in raw:
            cv2.polylines(source,[np.rint(polygon).astype(np.int32)],True,(0,255,0),1,cv2.LINE_AA)
        save_image(Path(record['context_path']).with_name(Path(record['image_id']).stem+'_source_grid.png'),source)


def prepare_boards(config, output):
    """Prepare all registered images; cache exact source/settings/code/template versions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = config['data']['metadata']
    images = json.loads((metadata/'images.json').read_text())
    positions = json.loads((metadata/'positions.json').read_text())
    override_path = metadata/'board_corner_overrides.json'
    overrides = json.loads(override_path.read_text()) if override_path.exists() else {}
    cache_path = output/'prepared_boards.json'
    cached = {r['image_id']: r for r in json.loads(cache_path.read_text())} if cache_path.exists() else {}
    code_hashes = {name: file_hash(Path(__file__).with_name(name)) for name in
                   ('board_detection.py', 'perspective.py', 'orientation.py')}
    common = {'board': config['board'], 'detection': config.get('board_detection', {}),
              'orientation': config.get('orientation', {}), 'code': code_hashes,
              'templates': file_hash(TEMPLATE_PATH)}
    expected = {'white_near_camera': 0, 'white_left': 1, 'black_near_camera': 2, 'white_right': 3}
    records = []
    for name, entry in sorted(images.items()):
        path = config['data']['raw']/name
        source_hash = file_hash(path)
        fen = positions[entry['position_id']]['fen']
        override = overrides.get(name)
        if override and override.get('source_sha256') != source_hash:
            raise ValueError(f'Manual corner override is stale: {name}')
        cache_value = {'source': source_hash, 'orientation': entry['orientation'], 'fen': fen, **common}
        if override:
            cache_value['corner_override'] = override
        cache_key = fingerprint(cache_value)
        prior = cached.get(name)
        cache_valid = prior and prior.get('cache_key') == cache_key
        if cache_valid and prior.get('context_path'):
            cache_valid = all(prior.get(key) and Path(prior[key]).exists()
                              and file_hash(prior[key]) == prior.get(hash_key)
                              for key, hash_key in (('context_path', 'context_sha256'), ('mask_path', 'mask_sha256')))
        if cache_valid:
            if prior.get('context_path') and prior.get('coordinate_system') != 'camera_view':
                turns = prior['orientation_rotation']
                board = RectifiedBoard(load_image(prior['context_path']), cv2.imread(prior['mask_path'], 0),
                                       tuple(prior['board_bounds']))
                board = rotate_rectified(board, (-turns) % 4)
                save_image(prior['context_path'], board.image)
                save_image(prior['mask_path'], board.valid_mask)
                prior.update(context_sha256=file_hash(prior['context_path']), mask_sha256=file_hash(prior['mask_path']),
                             board_bounds=list(board.board_bounds), orientation_rotation=0,
                             label_rotation=turns, coordinate_system='camera_view')
                prior['review_signature'] = board_review_signature(prior)
            update_grid_geometry(prior, path)
            records.append(prior)
            print(name, 'cached', flush=True)
            merged_cache = {**cached, **{r['image_id']:r for r in records}}
            cache_path.write_text(json.dumps(list(merged_cache.values()),indent=2)+'\n')
            continue
        photo = load_image(path)
        detection = (BoardDetection(order_corners(override['corners'])) if override else
                     detect_board_corners(photo, DetectionSettings(**config.get('board_detection', {}))))
        record = {'image_id': name, 'position_id': entry['position_id'], 'fen': fen,
                  'source_sha256': source_hash, 'cache_key': cache_key, 'board_detected': detection.success,
                  'reason': detection.reason, 'context_path': None, 'mask_path': None,
                  'orientation_source': None, 'orientation_rotation': None, 'review_signature': None}
        record['localization_source'] = 'manual_corners' if override else 'automatic'
        if detection.success:
            record['source_corners'] = detection.corners.tolist()
            rectified = rectify_with_config(photo, detection.corners, config)
            result = resolve_orientation(rectified, OrientationSettings(**config.get('orientation', {})))
            turns = result.rotation_quarters_ccw
            known = expected.get(entry['orientation'])
            record['automatic_orientation_status'] = result.status
            if turns is not None and known is not None and turns != known:
                record['reason'] = 'Automatic/capture orientation conflict.'
            elif turns is None and known is None:
                record['reason'] = 'Orientation unknown; no labels assigned.'
            else:
                if turns is None:
                    turns = known
                    record['orientation_source'] = 'capture_metadata'
                else:
                    record['orientation_source'] = 'border_markings'
                destination = output/'boards'/Path(name).stem
                context_path = save_image(destination.with_name(destination.name+'_context.png'), rectified.image)
                mask_path = save_image(destination.with_name(destination.name+'_valid.png'), rectified.valid_mask)
                record.update(context_path=str(context_path.resolve()), mask_path=str(mask_path.resolve()),
                              context_sha256=file_hash(context_path), mask_sha256=file_hash(mask_path),
                              orientation_rotation=0, label_rotation=turns, coordinate_system='camera_view',
                              board_bounds=list(rectified.board_bounds))
                record['review_signature'] = board_review_signature(record)
                update_grid_geometry(record, path)
        records.append(record)
        print(name, 'prepared' if record['context_path'] else record['reason'], flush=True)
        merged_cache = {**cached, **{r['image_id']:r for r in records}}
        cache_path.write_text(json.dumps(list(merged_cache.values()), indent=2)+'\n')
    cache_path.write_text(json.dumps(records, indent=2)+'\n')
    return records


def build_crops(records, reviews, config, output, *, preview_only=False):
    """Only matching, approved geometry AND label reviews produce supervised crops."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = {label.name: 0 for label in CLASSES}
    rejected = []
    padding = round(config['board']['normalized_size']/8*config['board']['crop_padding_squares'])
    for record in records:
        if not record.get('context_path') or (not preview_only and not review_approved(record, reviews.get(record['image_id']))):
            reason = (record.get('reason') if not record.get('context_path') else
                      'Board/labels review missing, stale, or rejected.')
            rejected.append({'image_id': record['image_id'], 'reason': reason,
                             'review_notes': reviews.get(record['image_id'], {}).get('notes')})
            continue
        if (record.get('context_sha256') and file_hash(record['context_path']) != record['context_sha256']
                or record.get('mask_sha256') and file_hash(record['mask_path']) != record['mask_sha256']):
            raise ValueError(f"Prepared pixels changed; prepare and review again: {record['image_id']}")
        rectified = RectifiedBoard(load_image(record['context_path']),
                                   cv2.imread(record['mask_path'], cv2.IMREAD_GRAYSCALE), tuple(record['board_bounds']))
        if rectified.valid_mask is None or rectified.valid_mask.shape != rectified.image.shape[:2]:
            raise ValueError(f"Missing or invalid coverage mask: {record['image_id']}")
        if rectified.board_image.shape[:2] != (config['board']['normalized_size'],)*2:
            raise ValueError(f"Prepared board size does not match configuration: {record['image_id']}")
        turns = record.get('label_rotation', 0)
        labels = labels_from_fen(record['fen'], turns)
        polygons = record.get('grid_geometry',{}).get('square_polygons')
        if polygons is None:
            x0,y0,x1,y1 = record['board_bounds']; side = (x1-x0)/8
            polygons = [[[x0+c*side,y0+r*side],[x0+(c+1)*side-1,y0+r*side],
                         [x0+(c+1)*side-1,y0+(r+1)*side-1],[x0+c*side,y0+(r+1)*side-1]]
                        for r in range(8) for c in range(8)]
        for index, polygon in enumerate(polygons):
            row, col = divmod(index,8)
            crop, mask, tight, bounds, target_polygon, coverage = polygon_crop(
                rectified.image, rectified.valid_mask, polygon, padding)
            label = labels[row][col]
            target_array = np.asarray(target_polygon)
            target = [*np.floor(target_array.min(axis=0)).astype(int).tolist(),
                      *(np.ceil(target_array.max(axis=0)).astype(int)+1).tolist()]
            identifier = f"{Path(record['image_id']).stem}_r{row}_c{col}"
            crop_path = save_if_changed(output/'crops'/f'{identifier}.png', crop)
            mask_path = save_if_changed(output/'masks'/f'{identifier}.png', mask)
            tight_path = save_if_changed(output/'tight'/f'{identifier}.png', tight)
            rows.append({'crop_id': identifier, 'crop_path': str(crop_path.resolve()),
                         'mask_path': str(mask_path.resolve()), 'tight_path': str(tight_path.resolve()),
                         'source_image': record['image_id'], 'position_id': record['position_id'],
                         'row': row, 'col': col, 'chess_square': chess_square_at(row, col, turns),
                         'class_id': label.class_id, 'class_name': label.name, 'label_version': LABEL_VERSION,
                         'coordinate_system': record.get('coordinate_system', 'white_at_bottom'),
                         'label_rotation': turns, 'orientation_source': record['orientation_source'],
                         'bounds': bounds, 'target_bounds': target, 'target_polygon': target_polygon,
                         'geometry_review_required': not review_approved(record,reviews.get(record['image_id'])), 'target_valid_fraction': coverage,
                         'context_valid_fraction': float((mask > 0).mean()), 'board_review_signature': record['review_signature'],
                         'visibility_status': 'unreviewed', 'quality_flags': (['target_outside_source'] if coverage < 1 else [])+
                         (['grid_fit_rejected'] if record.get('grid_geometry',{}).get('fit_status')=='rejected_seed_preview_only' else []),
                         'split': None, 'crop_sha256': file_hash(crop_path)})
            counts[label.name] += 1
    manifest = {'dataset_format': 'camera_view_polygon_context_v3',
                'preview_only': preview_only,
                'review_status': 'unapproved_candidates' if preview_only else 'matching_board_reviews_required',
                'label_version': LABEL_VERSION, 'classes': [asdict(label) for label in CLASSES],
                'board_count': len(rows)//64, 'crop_count': len(rows), 'class_counts': counts,
                'crop_padding_pixels': padding, 'split_status': 'unassigned_until_day10',
                'crop_size': config['board']['normalized_size']//8+2*padding,
                'crop_size_note': 'Nominal size; polygon context crops may vary. Resize only at classifier preprocessing.',
                'training_ready': False, 'results': rows, 'excluded_boards': rejected}
    apply_visibility_reviews(manifest, {})
    (output/('candidate_crops.json' if preview_only else 'crops.json')).write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest

# Public Day 10 loader; kept separate from image preparation and review logic.
from src.vision.dataset_loader import ContextSquareDataset
