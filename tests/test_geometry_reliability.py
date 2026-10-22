"""Independent-evidence recovery guards; no trained model needed for unit tests."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

import src.vision.geometry_reliability as recovery
from src.vision.perspective import RectifiedBoard
from src.vision.position_pipeline import PreparationError


def candidate(turns=0):
    seed = np.float32([[200, 200], [999, 200], [999, 999], [200, 999]])
    semantic = seed[[[3, 2, 1, 0], [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3]][turns]]
    h = cv2.getPerspectiveTransform(np.float32([[0, 0], [32, 0], [32, 32], [0, 32]]), semantic)
    polygons = {}
    for r in range(8):
        for f in range(8):
            poly = np.float32([[4*f, 4*r], [4*f+4, 4*r], [4*f+4, 4*r+4], [4*f, 4*r+4]])
            polygons[f'{chr(97+f)}{r+1}'] = cv2.perspectiveTransform(poly[:, None, :], h).reshape(4, 2).tolist()
    return dict(status='candidate', orientation_source='learned_border_markings', rotation_quarters_ccw=turns,
                learned_marking_evidence=dict(status='candidate', inliers=12, distinct_files=6, distinct_ranks=6,
                                             median_residual_px=1., corners=semantic.tolist()),
                grid_evidence_fraction=.8, refinement_max_shift_squares=.1,
                orientation_evidence=dict(status='resolved', rotation_quarters_ccw=turns),
                geometry=dict(source='markings', version=1, corners=semantic.tolist(), h_board_to_frame=h.ravel().tolist()),
                square_polygons=polygons)


@pytest.mark.parametrize('turns', range(4))
def test_two_axis_registration_keeps_existing_semantic_conventions(turns):
    seed, actual = recovery.validate_recovery(candidate(turns), (1200, 1200, 3))
    np.testing.assert_array_equal(seed, [[200, 200], [999, 200], [999, 999], [200, 999]])
    assert actual == turns


@pytest.mark.parametrize('path,value', [
    (('status',), 'uncertain'),
    (('orientation_source',), 'border_markings'),
    (('learned_marking_evidence', 'status'), 'uncertain'),
    (('learned_marking_evidence', 'inliers'), 7),
    (('learned_marking_evidence', 'distinct_files'), 2),
    (('learned_marking_evidence', 'distinct_ranks'), 2),
    (('learned_marking_evidence', 'median_residual_px'), 6.1),
    (('learned_marking_evidence', 'median_residual_px'), float('nan')),
    (('grid_evidence_fraction',), .24),
    (('grid_evidence_fraction',), float('nan')),
    (('refinement_max_shift_squares',), .46),
    (('refinement_max_shift_squares',), float('nan')),
    (('rotation_quarters_ccw',), 2),
    (('orientation_evidence', 'rotation_quarters_ccw'), 2),
    (('square_polygons',), None),
])
def test_completion_alone_cannot_bypass_independent_evidence(path, value):
    row = candidate()
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        recovery.validate_recovery(row, (1200, 1200, 3))


def test_refined_polygons_cannot_reassign_a_physical_square():
    row = candidate()
    row['square_polygons']['a1'] = row['square_polygons']['b1']
    with pytest.raises(ValueError, match='square identity'):
        recovery.validate_recovery(row, (1200, 1200, 3))


def test_mirrored_or_folded_geometry_is_not_rescued():
    row = candidate()
    row['geometry']['corners'] = row['geometry']['corners'][::-1]
    with pytest.raises(ValueError, match='Mirrored'):
        recovery.validate_recovery(row, (1200, 1200, 3))


def test_legacy_success_preserves_exact_canvas_grid_and_provenance(monkeypatch):
    rect = object()
    grid = {'square_polygons': ['unchanged']}
    provenance = {'localization_source': 'automatic', 'orientation_source': 'border_markings'}
    monkeypatch.setattr(recovery, 'prepare_raw_photo', lambda *args: (rect, grid, 3, provenance))
    monkeypatch.setattr(recovery, 'bootstrap_image', lambda *args: pytest.fail('Successful geometry must not be replaced'))
    actual = recovery.prepare_reliable_raw_photo('arbitrary-name.jpg', {})
    assert actual[0] is rect and actual[1] is grid and actual[2] == 3
    assert actual[3]['localization_source'] == 'automatic'
    assert actual[3]['geometry_reliability']['path'] == 'legacy'
    assert 'geometry_reliability' not in provenance


@pytest.mark.parametrize('stage', ['localization', 'orientation', 'grid'])
@pytest.mark.parametrize('reason', ['competing_alternative_seed_geometries', 'learned_and_template_orientation_conflict', 'reader_unavailable'])
def test_uncertain_or_unavailable_fallback_preserves_failure_and_reason(monkeypatch, stage, reason):
    def fail(*args):
        raise PreparationError(stage, 'legacy failure')
    monkeypatch.setattr(recovery, 'prepare_raw_photo', fail)
    monkeypatch.setattr(recovery, 'load_image', lambda *args: np.zeros((1200, 1200, 3), np.uint8))
    monkeypatch.setattr(recovery, 'bootstrap_image', lambda *args: dict(status='uncertain', reason=reason))
    with pytest.raises(PreparationError, match=reason) as error:
        recovery.prepare_reliable_raw_photo('arbitrary-name.jpg', {})
    assert error.value.stage == stage


@pytest.mark.parametrize('turns', range(4))
def test_failed_classical_geometry_uses_image_registration_without_annotations(monkeypatch, turns):
    image = np.zeros((1200, 1200, 3), np.uint8)
    rect = RectifiedBoard(image, np.full((1200, 1200), 255, np.uint8), (200, 200, 1000, 1000))
    yy, xx = np.mgrid[:9, :9]
    grid = {'intersections': np.stack([200+xx*799/8, 200+yy*799/8], axis=-1).tolist()}
    calls = []
    def prepare(path, config, destination, **kwargs):
        calls.append(kwargs)
        if not kwargs:
            raise PreparationError('orientation', 'incorrect classical polygon')
        return rect, grid, turns, {}
    original_config = {'orientation': {'min_matches': 8}}
    unchanged = deepcopy(original_config)
    def bootstrap(pixels, config):
        assert pixels is image and config['scan_fast_acquisition'] is True
        assert config['orientation']['min_matches'] == 8
        return candidate(turns)
    monkeypatch.setattr(recovery, 'prepare_raw_photo', prepare)
    monkeypatch.setattr(recovery, 'load_image', lambda *args: image)
    monkeypatch.setattr(recovery, 'bootstrap_image', bootstrap)
    actual = recovery.prepare_reliable_raw_photo('arbitrary-name.jpg', original_config)
    assert original_config == unchanged
    assert actual[:3] == (rect, grid, turns)
    assert calls[1]['white_side'] == recovery.SIDES[turns]
    assert actual[3]['localization_source'] == 'learned_border_markings_fallback'
    assert actual[3]['geometry_reliability']['geometry_review_required']


def test_candidate_inspector_restores_profile_even_if_detector_raises(monkeypatch):
    from scripts import evaluate_geometry_reliability as audit
    def fail(*args):
        raise RuntimeError('probe failure')
    monkeypatch.setattr(audit, 'detect_board_corners', fail)
    previous = sys.getprofile()
    with pytest.raises(RuntimeError, match='probe failure'):
        audit.inspect_candidates(np.zeros((10, 10, 3), np.uint8), None)
    assert sys.getprofile() is previous


@pytest.mark.parametrize('arguments', [
    ['--mode', 'rectified'], ['--white-side', 'top'], ['--corners', 'unread.json'],
])
def test_opt_in_cli_rejects_manual_or_rectified_inputs_without_creating_outputs(tmp_path, arguments):
    output = tmp_path/'output'
    command = [sys.executable, 'scripts/predict_position.py', 'missing.jpg', '--geometry-policy', 'marked-board',
               '--output-dir', str(output), *arguments]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'requires automatic raw input' in result.stderr
    assert not output.exists()


def test_opt_in_cli_refuses_existing_output(tmp_path):
    destination = tmp_path/'missing'
    destination.mkdir()
    marker = destination/'predictions.json'
    marker.write_text('{"preserve": true}')
    result = subprocess.run([sys.executable, 'scripts/predict_position.py', 'missing.jpg', '--geometry-policy', 'marked-board',
                             '--output-dir', str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and 'new unused' in result.stderr
    assert json.loads(marker.read_text()) == {'preserve': True}


@pytest.mark.parametrize('stem', ['006_var_05', '006_var_06', '006_var_07',
                                  '010_var_05', '010_var_06', '010_var_07', '016_var_06'])
def test_real_reviewed_failures_recover_geometry_and_semantics_without_annotation_input(stem):
    from scripts.evaluate_geometry_reliability import polygon_metrics, raw_polygons
    from src.utils.config import load_config
    from src.vision.dataset import review_approved

    root = Path(__file__).resolve().parents[1]
    name = f'position_{stem}.jpg'
    if not (root/'data/raw'/name).exists() or not (root/'models/checkpoints/board_marking_reader.pt').exists():
        pytest.skip('Local historical photograph and frozen board-font reader required')
    metadata = json.loads((root/'data/metadata/images.json').read_text())[name]
    p = next(r for r in json.loads((root/'data/classification/prepared_boards.json').read_text()) if r['image_id'] == name)
    reviews = json.loads((root/'data/metadata/day9_board_reviews.json').read_text())
    assert review_approved(p, reviews[name])
    # Only bytes and ordinary runtime settings enter preparation. Metadata and
    # reviewed geometry are joined afterwards as evaluation references.
    rectified, grid, turns, provenance = recovery.prepare_reliable_raw_photo(root/'data/raw'/name, load_config())
    expected = {'black_near_camera': 2, 'white_right': 3}[metadata['orientation']]
    assert turns == expected
    assert provenance['localization_source'] == 'learned_border_markings_fallback'
    actual = raw_polygons(grid, provenance['source_corners'], rectified.board_bounds)
    metrics = polygon_metrics(actual, p['grid_geometry']['source_square_polygons'])
    assert metrics['strict_outer_agreement'], metrics
    assert metrics['camera_cell_centres_in_corresponding_reference'] == 64, metrics
