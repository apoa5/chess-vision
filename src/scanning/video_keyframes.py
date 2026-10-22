"""D.1 offline selection. Scores are heuristics; no FEN, classifier or fusion inputs."""
from fractions import Fraction
import math

import cv2
import numpy as np

from src.scanning.frame_selection import image_signature
from src.scanning.pilot_comparison import nearest_time

STRATEGIES = ('uniform', 'quality', 'quality_temporal', 'quality_diverse')


def validate_settings(c):
    for name in ('interval_ms', 'candidate_limit', 'quality_warp_size', 'temporal_gap_ms',
                 'diverse_gap_ms', 'sharpness_scale', 'min_sharpness', 'min_board_fraction'):
        if not isinstance(c[name], (int, float)) or not math.isfinite(c[name]) or c[name] <= 0:
            raise ValueError('Invalid setting: '+name)
    if not 64 <= c['quality_warp_size'] <= 1024 or not 0 < c['min_board_fraction'] < 1:
        raise ValueError('Invalid board quality settings')
    if not 0 <= c['duplicate_hamming'] <= 64 or not 0 <= c['diversity_weight'] <= 1:
        raise ValueError('Invalid duplicate/diversity settings')
    if any(type(c[k]) is not int for k in ('candidate_limit','quality_warp_size','duplicate_hamming')):
        raise ValueError('Integer limits/sizes required')
    if not 0 < c['max_extreme_fraction'] <= 1 or c['duplicate_view_distance'] < 0:
        raise ValueError('Invalid quality/duplicate gates')
    weights=c['quality_weights']
    if set(weights)!=set(('sharpness','contrast','exposure','board_size','perspective')) or any(not math.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
        raise ValueError('Finite nonnegative quality weights required')
    if c['geometry_policies']!=['legacy','marked-board'] or c['development_only'] is not True:
        raise ValueError('Explicit offline two-policy development configuration required')
    if not c['budgets'] or any(type(b) is not int or not 1 <= b <= 100 for b in c['budgets']):
        raise ValueError('Positive bounded integer budgets required')
    return c


def temporal_candidates(frames, interval_ms, limit=256):
    """Nearest actual PTS to a source-relative regular grid, deterministic ties."""
    if interval_ms <= 0 or not math.isfinite(interval_ms) or not frames:
        raise ValueError('Positive cadence and nonempty native index required')
    times = [Fraction(*f['pts_time']) for f in frames]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Nonmonotonic original PTS')
    delta = Fraction(str(interval_ms))/1000
    n = int((times[-1]-times[0])/delta)+1
    if n > limit:
        raise ValueError('Candidate limit exceeded; explicitly increase cadence/limit')
    selected, seen = [], set()
    for i in range(n):
        target = times[0]+i*delta
        j = nearest_time(times, target)
        if j not in seen:
            selected.append(dict(frames[j], sampling_target_relative_ms=float(i*delta*1000)))
            seen.add(j)
    return selected


def geometry_state(candidate, geometry, binding, review=None):
    """A review cannot repair stale pixels, unresolved semantics or invalid polygons."""
    expected = {k: candidate[k] for k in ('session_id', 'frame_id', 'pts', 'time_base',
                                         'source_video_sha256', 'derivative_sha256')}
    if binding != expected:
        return 'invalid_geometry_binding', ['geometry_not_bound_to_exact_frame']
    if not geometry.get('completed'):
        error = geometry.get('error', '')
        stage = ('rectification_validation' if 'Corners must be ordered' in error
                 else geometry.get('failure_stage', 'unknown'))
        return 'unresolved', [str(stage), error]
    if geometry.get('rotation_quarters_ccw') not in (0, 1, 2, 3):
        return 'unresolved', ['semantic_orientation_not_validated']
    polygons = np.asarray(geometry.get('source_square_polygons'), dtype=float)
    ids = geometry.get('semantic_square_ids', [])
    canonical = {f'{f}{r}' for f in 'abcdefgh' for r in range(1, 9)}
    if (polygons.shape != (64, 4, 2) or not np.isfinite(polygons).all()
            or set(ids) != canonical or len(ids) != 64):
        return 'unresolved', ['invalid_grid_or_semantic_ids']
    if any(not cv2.isContourConvex(p.astype(np.float32)) or cv2.contourArea(p.astype(np.float32)) < 1
           for p in polygons):
        return 'unresolved', ['degenerate_square_polygon']
    if (polygons[..., 0].min() < 0 or polygons[..., 1].min() < 0
            or polygons[..., 0].max() >= candidate['width']
            or polygons[..., 1].max() >= candidate['height']):
        return 'unresolved', ['fitted_grid_outside_source']
    if review is not None:
        if review.get('binding') != expected:
            raise ValueError('Review annotation bound to another frame')
        if review.get('status') == 'rejected_incorrect':
            return 'rejected_incorrect', ['explicit_geometry_review_rejection']
        if review.get('status') == 'visually_verified':
            if not review.get('reviewer') or review.get('reviewer_kind') != 'human':
                raise ValueError('Human geometry verification requires attributed review')
            return 'visually_verified', []
    return 'completed_unverified', []


def board_features(image, geometry, c):
    """Quality at a fixed board-plane raster; no semantic pixel rotation or crops changed."""
    p = np.asarray(geometry['source_square_polygons'], np.float32)
    q = np.array([p[0, 0], p[7, 1], p[63, 2], p[56, 3]], np.float32)
    if not cv2.isContourConvex(q):
        raise ValueError('Invalid fitted outer boundary')
    size = int(c['quality_warp_size'])
    matrix = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]]))
    board = cv2.warpPerspective(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(np.percentile(gray, 90)-np.percentile(gray, 10))
    extremes = float(np.mean((gray < 10) | (gray > 245)))
    edges = np.linalg.norm(q-np.roll(q, -1, axis=0), axis=1)
    ratio = float(edges.min()/edges.max())
    area = float(cv2.contourArea(q)/(image.shape[0]*image.shape[1]))
    centres = dict(zip(geometry['semantic_square_ids'], p.mean(axis=1)))
    axis = centres['h4']-centres['a4']
    angle = math.atan2(float(axis[1]), float(axis[0]))
    # Image-plane geometry proxies, not metric camera pose or elevation.
    descriptor = [angle, math.log(max(1e-6, float((edges[0]+edges[2])/(edges[1]+edges[3])))),
                  math.log(max(1e-6, float(edges[0]/edges[2]))),
                  math.log(max(1e-6, float(edges[1]/edges[3])))]
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0); gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    eigen = np.linalg.eigvalsh([[np.mean(gx*gx), np.mean(gx*gy)],
                              [np.mean(gx*gy), np.mean(gy*gy)]])
    anisotropy = float(eigen[0]/max(eigen[1], 1e-9))
    components = dict(sharpness=min(1., math.log1p(sharp)/math.log1p(c['sharpness_scale'])),
                      contrast=min(1., contrast/100), exposure=1-extremes,
                      board_size=min(1., area/.35), perspective=ratio)
    weights = c['quality_weights']
    score = sum(weights[k]*components[k] for k in weights)/sum(weights.values())
    reasons = []
    if sharp < c['min_sharpness']:reasons.append('board_sharpness_below_experimental_gate')
    if area < c['min_board_fraction']:reasons.append('board_too_small_in_image')
    if extremes > c['max_extreme_fraction']:reasons.append('board_exposure_outside_experimental_gate')
    return dict(quality_score=score, quality_components=components, board_sharpness=sharp,
                board_contrast_p90_p10=contrast, extreme_luma_fraction=extremes,
                board_area_fraction=area, edge_ratio_proxy=ratio, direction_anisotropy=anisotropy,
                view_descriptor=descriptor, image_plane_file_axis_degrees=math.degrees(angle),
                predicted_orientation=geometry['predicted_camera_orientation'],
                appearance_dhash=f'{image_signature(board):016x}', outer_quad=q.tolist(),
                quality_rejection_reasons=reasons, visibility='unknown_occlusion',
                source_bounds_coverage='fitted_grid_inside_image', elevation_degrees=None)


def view_distance(a, b):
    da, db = a['view_descriptor'], b['view_descriptor']
    angle = abs((da[0]-db[0]+math.pi)%(2*math.pi)-math.pi)/math.pi
    shape = min(1., float(np.linalg.norm(np.asarray(da[1:])-np.asarray(db[1:])))/2)
    return .75*angle+.25*shape


def near_duplicate(a, b, c):
    if a['decoded_pixel_sha256'] == b['decoded_pixel_sha256']:return True
    hamming = (int(a['appearance_dhash'], 16)^int(b['appearance_dhash'], 16)).bit_count()
    return hamming <= c['duplicate_hamming'] and view_distance(a, b) < c['duplicate_view_distance']


def row_usable(row):
    return (row.get('eligible') is True and row.get('geometry_state') in ('completed_unverified','visually_verified')
            and math.isfinite(row.get('quality_score',float('nan'))))


def select_keyframes(rows, strategy, budget, c):
    if strategy not in STRATEGIES or type(budget) is not int or budget <= 0:
        raise ValueError('Unknown strategy/invalid budget')
    # Do not inspect FENs, position labels, annotations of pieces, or model scores.
    ordered = sorted(rows, key=lambda r: (r['relative_timestamp_ms'], r['frame_id']))
    if len({r['frame_id'] for r in ordered}) != len(ordered):raise ValueError('Duplicate frame identities')
    eligible = [r for r in ordered if row_usable(r)]
    selected, trace = [], []
    gap = c['temporal_gap_ms'] if strategy == 'quality_temporal' else (
        c['diverse_gap_ms'] if strategy == 'quality_diverse' else 0)
    if eligible:
        targets = np.linspace(ordered[0]['relative_timestamp_ms'], ordered[-1]['relative_timestamp_ms'], budget)
    else:targets = []
    for step in range(budget):
        choices = []
        for row in eligible:
            if any(row['frame_id'] == s['frame_id'] for s in selected):continue
            if any(near_duplicate(row, s, c) for s in selected):continue
            if any(abs(row['relative_timestamp_ms']-s['relative_timestamp_ms']) < gap for s in selected):continue
            novelty = min((view_distance(row, s) for s in selected), default=0.)
            if strategy == 'uniform':
                merit = -abs(row['relative_timestamp_ms']-float(targets[step]))
            elif strategy == 'quality_diverse':
                merit = (1-c['diversity_weight'])*row['quality_score']+c['diversity_weight']*novelty
            else:merit = row['quality_score']
            choices.append((merit, row, novelty))
        if not choices:break
        merit, chosen, novelty = min(choices, key=lambda x: (-x[0], -x[1]['quality_score'],
                                                           x[1]['relative_timestamp_ms'], x[1]['frame_id']))
        selected.append(chosen)
        trace.append(dict(frame_id=chosen['frame_id'], selection_rank=len(selected),
                          quality_score=chosen['quality_score'], min_view_distance_to_selected=novelty,
                          selection_merit=merit, reason=strategy+'_best_remaining_with_shared_duplicate_guard'))
    reasons = []
    selected_ids = {r['frame_id'] for r in selected}
    for row in ordered:
        why = list(row.get('unusable_reasons', []))
        if row['frame_id'] in selected_ids:why = ['selected']
        elif row_usable(row):
            if any(near_duplicate(row, s, c) for s in selected):why.append('redundant_with_selected')
            if any(abs(row['relative_timestamp_ms']-s['relative_timestamp_ms']) < gap for s in selected):why.append('temporal_separation')
            if not why:why.append('budget_or_rank')
        if not row_usable(row) and not why:why=['invalid_geometry_state_or_quality']
        reasons.append(dict(frame_id=row['frame_id'], selected=row['frame_id'] in selected_ids, reasons=why))
    return dict(strategy=strategy, budget=budget, selected=[r['frame_id'] for r in sorted(selected, key=lambda r:r['relative_timestamp_ms'])],
                selection_trace=trace, candidate_decisions=reasons)


def selection_metrics(rows, selection, c):
    chosen = [r for r in rows if r['frame_id'] in selection['selected']]
    times = sorted(r['relative_timestamp_ms']/1000 for r in chosen)
    pairs = [(a, b) for i, a in enumerate(chosen) for b in chosen[i+1:]]
    available = sorted(set(r['predicted_orientation'] for r in rows if row_usable(r)))
    counts = {v:sum(r['predicted_orientation'] == v for r in chosen) for v in available}
    return dict(selected_count=len(chosen), timestamps_seconds=times,
                temporal_span_seconds=times[-1]-times[0] if times else 0.,
                min_gap_seconds=min(np.diff(times)).item() if len(times)>1 else None,
                mean_quality=float(np.mean([r['quality_score'] for r in chosen])) if chosen else None,
                predicted_orientation_counts=counts, available_predicted_orientations=len(available),
                selected_predicted_orientations=sum(v>0 for v in counts.values()),
                dominant_predicted_orientation_fraction=max(counts.values())/len(chosen) if chosen else None,
                mean_pair_view_distance=float(np.mean([view_distance(a,b) for a,b in pairs])) if pairs else None,
                hard_duplicate_pairs=sum(near_duplicate(a,b,c) for a,b in pairs),
                soft_similar_pairs=sum(view_distance(a,b)<.12 for a,b in pairs),
                edge_ratio_proxy_range=[min(r['edge_ratio_proxy'] for r in chosen), max(r['edge_ratio_proxy'] for r in chosen)] if chosen else None,
                visually_verified_selected=sum(r['geometry_state']=='visually_verified' for r in chosen),
                completed_unverified_selected=sum(r['geometry_state']=='completed_unverified' for r in chosen))
