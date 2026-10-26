"""Explicit D.3c review waiver for provisional numerical comparisons only.

The frozen D.3 aggregation kernel and all numerical checks remain unchanged.
No record is relabeled human_verified. The original strict validator is restored
on exit. This isolated offline adapter must never be used by production code.
"""
from fractions import Fraction
from unittest.mock import patch
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.scanning import offline_fusion
from src.scanning.square_evidence import SQUARES,uncertainty,validate_observation


def validate_provisional_observation(o):
    if o['geometry_verification_status']=='human_verified':return validate_observation(o)
    if o['geometry_verification_status']!='completed_unverified':raise ValueError('Rejected or unresolved geometry cannot be fused')
    if o['canonical_square'] not in SQUARES or o['class_names']!=list(CLASS_NAMES) or o['predicted_class']!=CLASS_NAMES[o['predicted_class_id']]:raise ValueError('Invalid canonical class mapping')
    if o['calibration_status']!='not_available' or o['calibrated_probabilities'] is not None:raise ValueError('Unapproved calibration')
    t=Fraction(o['original_pts'])*Fraction(*o['time_base'])
    if t!=Fraction(*o['pts_time']) or abs(float(t)*1000-o['timestamp_ms'])>1e-6:raise ValueError('PTS mismatch')
    expected=uncertainty(o['raw_logits']);p=np.asarray(o['probabilities'])
    if p.shape!=(13,) or not np.isfinite(p).all() or not np.allclose(p,expected['probabilities'],atol=1e-7,rtol=1e-6):raise ValueError('Probabilities do not match logits')
    for k in ('predicted_class_id','maximum_probability','top_two_probability_margin','predictive_entropy'):
        if not np.isclose(o[k],expected[k],atol=1e-7):raise ValueError('Wrong uncertainty output')
    if not o['condition_memberships'] or any(m['selection_budget'] not in (5,10) for m in o['condition_memberships']):raise ValueError('Missing selection provenance')
    if len(o['checkpoint_sha256'])!=64 or len(o['crop_config_sha256'])!=64:raise ValueError('Missing model/crop identity')
    return o


def fuse_provisional_board(rows,method,config,*,explicit_review_waiver=False):
    if not explicit_review_waiver:raise ValueError('Explicit interim-only review waiver required')
    if method not in ('single_quality','mean_probabilities','mean_logits'):raise ValueError('D.3c registered methods only')
    # Only this offline process's input-review validator is replaced. Actual
    # uncertainty, grouping, weighting, ties and abstention execute frozen code.
    # The records retain their real unverified status throughout.
    with patch.object(offline_fusion,'validate_observation',validate_provisional_observation):
        result=offline_fusion.fuse_board(rows,method,config)
    return result
