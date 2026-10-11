import json

import chess
import numpy as np
import pytest

from src.utils.image_io import save_image
from src.vision.dataset import labels_from_fen, board_review_signature, review_approved, build_crops, file_hash
from scripts.inspect_dataset import inspect


def test_fen_labels_are_white_at_bottom():
    labels = labels_from_fen(chess.STARTING_FEN)
    assert ''.join(x.symbol for x in labels[0]) == 'rnbqkbnr'
    assert ''.join(x.symbol for x in labels[7]) == 'RNBQKBNR'
    assert all(x.class_id == 0 for x in labels[3])
    edge = labels_from_fen('7k/8/8/8/8/8/8/K7 w - - 0 1')
    assert edge[0][7].symbol == 'k'
    assert edge[7][0].symbol == 'K'
    with pytest.raises(ValueError):
        labels_from_fen('8/8/8/8/8/8/8/8')


def test_review_is_bound_to_geometry_and_ground_truth():
    record = dict(source_sha256='a', context_sha256='b', mask_sha256='c',
                  fen=chess.STARTING_FEN, board_bounds=[20,20,100,100], orientation_rotation=0)
    record['review_signature'] = board_review_signature(record)
    review = dict(signature=record['review_signature'], alignment_correct=True, labels_correct=True)
    assert review_approved(record, review)
    assert not review_approved(record, None)
    assert not review_approved(record, {**review,'labels_correct':False})
    record['fen'] = '7k/8/8/8/8/8/8/K7 w - - 0 1'
    record['review_signature'] = board_review_signature(record)
    assert not review_approved(record, review)


def test_crops_preserve_context_target_and_provenance(tmp_path):
    image = np.zeros((120,120,3), np.uint8)
    image[90:100,20:30] = [10,20,30]
    context = save_image(tmp_path/'context.png', image)
    mask = save_image(tmp_path/'mask.png', np.full((120,120),255,np.uint8))
    record = dict(image_id='example.jpg', position_id='position_001', context_path=str(context),
                  mask_path=str(mask), board_bounds=[20,20,100,100], fen=chess.STARTING_FEN,
                  orientation_source='capture_metadata', review_signature='verified')
    config = {'board':{'normalized_size':80,'crop_padding_squares':2}}
    assert build_crops([record], {}, config, tmp_path)['crop_count'] == 0
    review = {'example.jpg':{'signature':'verified','alignment_correct':True,'labels_correct':True}}
    manifest = build_crops([record], review, config, tmp_path)
    assert manifest['crop_count'] == 64
    assert manifest['class_counts']['empty'] == 32
    row = manifest['results'][56]
    assert row['class_name']=='white_rook'
    assert row['chess_square']=='a1'
    assert row['target_bounds']==[20,20,30,30]
    assert row['target_valid_fraction']==1
    assert row['split'] is None
    assert manifest['training_ready'] is False
    before = file_hash(row['crop_path'])
    inspect(tmp_path, tmp_path/'review', count=3)
    assert file_hash(row['crop_path']) == before
    ids = json.loads((tmp_path/'review/sample_ids.json').read_text())['crop_ids']
    assert len(ids) == 25  # Three empty/pawn, two minor/rook, one queen/king each.
    record['context_sha256'] = 'changed'
    with pytest.raises(ValueError, match='Prepared pixels changed'):
        build_crops([record], review, config, tmp_path)


def test_visibility_reviews_expire_and_do_not_approve_clipped_targets():
    from src.vision.dataset import apply_visibility_reviews
    row = {'crop_id':'a','crop_sha256':'pixels','board_review_signature':'board','quality_flags':[]}
    manifest = {'results':[row]}
    review = {'a':{'crop_sha256':'pixels','board_review_signature':'board','status':'partial'}}
    apply_visibility_reviews(manifest, review)
    assert row['visual_review_approved']
    row['quality_flags'] = ['target_outside_source']
    apply_visibility_reviews(manifest, review)
    assert not row['visual_review_approved']
    row['crop_sha256'] = 'changed'
    apply_visibility_reviews(manifest, review)
    assert row['visibility_status'] == 'unreviewed'


def test_summary_keeps_excluded_positions_and_review_scope_visible():
    from src.vision.dataset import dataset_summary
    manifest = {'results':[], 'class_counts':{}, 'visibility_counts':{},
                'crop_size':500, 'crop_padding_pixels':200, 'split_status':'unassigned_until_day10'}
    records = [{'image_id':'bad.jpg','position_id':'position_001','board_detected':False}]
    summary = dataset_summary(manifest, records)
    assert summary['excluded_board_count'] == 1
    assert summary['positions_without_accepted_boards'] == ['position_001']
    assert summary['training_ready'] is False


@pytest.mark.parametrize('turns,corners', [
    (0, ['a8','h8','a1','h1']), (1, ['a1','a8','h1','h8']),
    (2, ['h1','a1','h8','a8']), (3, ['h8','h1','a8','a1'])])
def test_camera_view_coordinates_and_fen_labels(turns, corners):
    from src.vision.dataset import chess_square_at
    cells = [(0,0),(0,7),(7,0),(7,7)]
    assert [chess_square_at(r,c,turns) for r,c in cells] == corners
    board = chess.Board('r6k/1p6/2n5/3b4/4B3/5N2/6P1/K6R w - - 0 1')
    labels = labels_from_fen(board.fen(), turns)
    assert len({chess_square_at(r,c,turns) for r in range(8) for c in range(8)})==64
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(chess.parse_square(chess_square_at(r,c,turns)))
            assert labels[r][c].symbol == (piece.symbol() if piece else None)


def test_manual_corners_are_used_and_bound_to_source(tmp_path, monkeypatch):
    import src.vision.dataset as module
    from src.utils.config import load_config
    from src.vision.orientation import OrientationResult
    raw=tmp_path/'raw';metadata=tmp_path/'metadata';raw.mkdir();metadata.mkdir()
    photo=save_image(raw/'example.png', np.full((120,120,3),100,np.uint8))
    (metadata/'images.json').write_text(json.dumps({'example.png':{'position_id':'position_001','orientation':'white_near_camera'}}))
    (metadata/'positions.json').write_text(json.dumps({'position_001':{'fen':chess.STARTING_FEN}}))
    override={'source_sha256':file_hash(photo),'corners':[[20,20],[99,20],[99,99],[20,99]]}
    path=metadata/'board_corner_overrides.json';path.write_text(json.dumps({'example.png':override}))
    def detector_must_not_run(*args):
        raise AssertionError('Manual corners should bypass automatic detection')
    monkeypatch.setattr(module,'detect_board_corners',detector_must_not_run)
    monkeypatch.setattr(module,'resolve_orientation',lambda *args:OrientationResult('uncertain'))
    config=load_config();config['data']['raw']=raw;config['data']['metadata']=metadata
    records=module.prepare_boards(config,tmp_path/'classification')
    assert records[0]['localization_source']=='manual_corners'
    assert records[0]['context_path']
    override['source_sha256']='stale';path.write_text(json.dumps({'example.png':override}))
    with pytest.raises(ValueError,match='Manual corner override is stale'):
        module.prepare_boards(config,tmp_path/'classification')
