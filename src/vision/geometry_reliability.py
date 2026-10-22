"""Opt-in raw-photo recovery using the existing marked-board bootstrap.

The frozen photo pipeline is tried first. Only explicit preparation failures may
use the existing board-font reader, spatial registration and grid validation.
No piece predictions, filenames, capture metadata or reference annotations enter
the recovery decision. A returned candidate still requires geometry review.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from src.scanning.bootstrap import bootstrap_image
from src.scanning.replay import _geometry
from src.utils.image_io import load_image
from src.vision.geometry_audit import context_transform, project
from src.vision.perspective import order_corners
from src.vision.position_pipeline import PreparationError, prepare_raw_photo

POLICY_VERSION = 'marked_board_recovery_v1'
SIDES = ('bottom', 'left', 'top', 'right')


def validate_recovery(candidate, image_shape):
    """Require the existing independent character and geometry evidence gates.

    These repeat safety invariants at the adapter boundary, not relaxed checker
    thresholds. Scores/inlier counts are evidence strengths, not probabilities.
    """
    if candidate.get('status') != 'candidate':
        raise ValueError(candidate.get('reason', 'bootstrap_uncertain'))
    evidence = candidate.get('learned_marking_evidence') or {}
    if (evidence.get('status') != 'candidate'
            or candidate.get('orientation_source') != 'learned_border_markings'):
        raise ValueError('independent_marking_registration_required')
    if (evidence.get('inliers', 0) < 8 or evidence.get('distinct_files', 0) < 3
            or evidence.get('distinct_ranks', 0) < 3):
        raise ValueError('insufficient_two_axis_marking_evidence')
    if (not np.isfinite(evidence.get('median_residual_px', np.nan))
            or not 0 <= evidence['median_residual_px'] <= 6):
        raise ValueError('marking_residual_exceeds_existing_ransac_tolerance')
    turns = candidate.get('rotation_quarters_ccw')
    if type(turns) is not int or turns not in range(4):
        raise ValueError('invalid_semantic_rotation')
    if (not np.isfinite(candidate.get('grid_evidence_fraction', np.nan))
            or not .25 <= candidate['grid_evidence_fraction'] <= 1
            or not np.isfinite(candidate.get('refinement_max_shift_squares', np.nan))
            or not 0 <= candidate['refinement_max_shift_squares'] <= .45):
        raise ValueError('insufficient_or_excessively_shifted_grid')
    orientation = candidate.get('orientation_evidence') or {}
    if orientation.get('rotation_quarters_ccw') != turns:
        raise ValueError('orientation_evidence_conflicts_with_geometry')
    height, width = image_shape[:2]
    geometry = candidate.get('geometry') or {}
    if not isinstance(candidate.get('square_polygons'), dict):
        raise ValueError('validated_square_polygons_required')
    _geometry(geometry | {'source': 'manual', 'square_polygons': candidate.get('square_polygons')}, width, height)
    # Semantic a1 must agree with the separately carried quarter-turn value.
    semantic = np.asarray(geometry['corners'], np.float32)
    anchor = int(np.argmin(np.linalg.norm(order_corners(semantic)-semantic[0], axis=1)))
    if {3: 0, 0: 1, 1: 2, 2: 3}[anchor] != turns:
        raise ValueError('semantic_corners_conflict_with_rotation')
    return order_corners(evidence['corners']), turns


def prepare_reliable_raw_photo(image_path, config, destination=None):
    """Same return contract as prepare_raw_photo, preserving all successful inputs.

    Recovery is board-font-specific, uses the already trained frozen reader, and
    explicitly fails if that reader or its spatial evidence is unavailable.
    """
    destination = Path(destination) if destination is not None else None
    legacy_dir = destination/'legacy' if destination is not None else None
    if legacy_dir is not None:
        legacy_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = prepare_raw_photo(image_path, config, legacy_dir)
    except PreparationError as error:
        if error.stage not in ('localization', 'orientation', 'grid'):
            raise
        original_stage = error.stage
        original_reason = str(error)
    else:
        # Do not rerender/reorient crops that already completed preparation.
        rectified, grid, turns, provenance = result
        return rectified, grid, turns, provenance | {
            'geometry_reliability': {'policy_version': POLICY_VERSION, 'path': 'legacy',
                                     'geometry_review_required': True}}

    image = load_image(image_path)
    # Explicit alternate evidence policy; original config and every threshold stay
    # unchanged. No training or checkpoint replacement occurs.
    recovery_config = config | {'learned_markings': {'enabled': True, 'min_character_confidence': .80},
                                'scan_fast_acquisition': True}
    candidate = bootstrap_image(image, recovery_config)
    if destination is not None:
        (destination/'recovery.json').write_text(json.dumps(candidate, indent=2)+'\n')
    try:
        corners, turns = validate_recovery(candidate, image.shape)
    except (ValueError, KeyError, TypeError) as error:
        raise PreparationError(original_stage,
                               f'{original_reason} Marked-board recovery rejected: {error}') from error
    # The recognized semantic side is supplied internally, never read from an
    # annotation. Reuse the verified photo warp/grid implementation unchanged.
    rectified, grid, _, _ = prepare_raw_photo(image_path, config, destination,
                                             corners=corners, white_side=SIDES[turns])
    inverse = np.linalg.inv(context_transform(corners, rectified.board_bounds))
    lattice = np.asarray(grid['intersections'], np.float32)
    outer = project(lattice[[0, 0, 8, 8], [0, 8, 8, 0]], inverse)
    if not np.allclose(order_corners(outer), order_corners(candidate['geometry']['corners']), atol=.02):
        raise PreparationError('grid', 'Recovery geometry changed between validation and crop preparation.')
    provenance = {'localization_source': 'learned_border_markings_fallback',
                  'orientation_source': 'learned_border_markings',
                  'orientation_evidence': candidate['orientation_evidence'],
                  'source_corners': corners.tolist(),
                  'geometry_reliability': {'policy_version': POLICY_VERSION, 'path': 'marked_board_recovery',
                                           'baseline_failure_stage': original_stage,
                                           'geometry_review_required': True,
                                           'grid_evidence_fraction': candidate['grid_evidence_fraction'],
                                           'refinement_max_shift_squares': candidate['refinement_max_shift_squares'],
                                           'learned_marking_evidence': candidate['learned_marking_evidence']}}
    return rectified, grid, turns, provenance
