"""Offline D.2 observation records and descriptive diagnostics. No fusion rule."""
from fractions import Fraction
import hashlib
import json
import math
from typing import Literal, TypedDict

import numpy as np

from src.chess.labels import CLASS_NAMES

SQUARES = tuple(f'{f}{r}' for r in range(8,0,-1) for f in 'abcdefgh')
CHECKS = ('true_board_corners','semantic_orientation','canonical_square_mapping','sufficient_for_crops')


class SquareEvidence(TypedDict):
    canonical_square: str
    state: Literal['unknown','candidate','resolved','conflict']
    class_distribution: list[float] | None
    observation_ids: list[str]
    viewpoint_groups: list[str]
    visibility: str
    last_observation_timestamp_ms: float | None


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify_reviews(packet, reviews, packet_sha256, *, require_primary=True):
    if reviews['schema_version']!='offline_evidence_geometry_reviews_d2_v1' or reviews['packet_sha256']!=packet_sha256:
        raise ValueError('Reviews not bound to this packet')
    lookup={r['review_id']:r for r in reviews['reviews']}
    if len(lookup)!=len(reviews['reviews']):raise ValueError('Duplicate review')
    expected={r['review_id'] for r in packet['frames']}
    if set(lookup)!=expected:raise ValueError('Every unique frame needs a review status')
    pending=[];statuses={}
    for row in packet['frames']:
        review=lookup[row['review_id']];decision=review['decision']
        if review['binding']!=row['binding']:raise ValueError('Stale review binding')
        if decision not in ('pending','human_verified','rejected','unresolved'):raise ValueError('Unknown review decision')
        if decision!='pending':
            if review.get('reviewer_kind')!='human' or not str(review.get('reviewer') or '').strip():raise ValueError('Explicit attributed human review required')
        if decision=='human_verified' and any(review.get('checks',{}).get(k) is not True for k in CHECKS):
            raise ValueError('All four geometry checks must be confirmed')
        if decision in ('rejected','unresolved') and not review.get('notes','').strip():raise ValueError('Rejection/unresolved reason required')
        if decision=='pending' and any(c in row['conditions'] for c in ('diversity_5','uniform_5')):pending.append(row['review_id'])
        statuses[row['review_id']]=dict(geometry_completed=row['geometry_completed'],human_review=decision,
            geometry_status='human_verified' if decision=='human_verified' else ('geometry_rejected' if decision=='rejected' else ('geometry_unresolved' if decision=='unresolved' else 'completed_unverified')))
    if require_primary and pending:raise ValueError('Primary geometry review pending: '+', '.join(pending))
    return statuses


def uncertainty(logits):
    """Stable uncalibrated softmax; entropy is in natural-log units (nats)."""
    z=np.asarray(logits,dtype=np.float64)
    if z.shape!=(13,) or not np.isfinite(z).all():raise ValueError('Thirteen finite logits required')
    exp=np.exp(z-z.max());p=exp/exp.sum();order=np.sort(p)
    return dict(probabilities=p.tolist(),predicted_class_id=int(p.argmax()),maximum_probability=float(order[-1]),
                top_two_probability_margin=float(order[-1]-order[-2]),predictive_entropy=float(-np.sum(p*np.log(np.maximum(p,np.finfo(float).tiny)))))


def validate_observation(o):
    if o['canonical_square'] not in SQUARES:raise ValueError('Unknown canonical square')
    if o['class_names']!=list(CLASS_NAMES) or o['predicted_class']!=CLASS_NAMES[o['predicted_class_id']]:raise ValueError('Wrong class mapping')
    if o['geometry_verification_status']!='human_verified':raise ValueError('Unverified geometry is not primary recognition evidence')
    if o['calibration_status']!='not_available' or o['calibrated_probabilities'] is not None:raise ValueError('No approved B.4 calibration is registered for this run')
    t=Fraction(o['original_pts'])*Fraction(*o['time_base'])
    if t!=Fraction(*o['pts_time']) or abs(float(t)*1000-o['timestamp_ms'])>1e-6:raise ValueError('Observation timestamp mismatch')
    expected=uncertainty(o['raw_logits']);p=np.asarray(o['probabilities'])
    if p.shape!=(13,) or not np.isfinite(p).all() or not np.allclose(p,expected['probabilities'],atol=1e-7,rtol=1e-6):raise ValueError('Probabilities do not match logits')
    for k in ('predicted_class_id','maximum_probability','top_two_probability_margin','predictive_entropy'):
        if not np.isclose(o[k],expected[k],atol=1e-7):raise ValueError('Wrong prediction/uncertainty output')
    if not o['condition_memberships'] or any(x['selection_budget'] not in (5,10) for x in o['condition_memberships']):raise ValueError('Missing selection provenance')
    if len(o['checkpoint_sha256'])!=64 or len(o['crop_config_sha256'])!=64:raise ValueError('Missing configuration identity')
    return o


def square_records(observations):
    """One container per square; no pooled distribution, vote, or resolved state."""
    result={s:dict(canonical_square=s,state='unknown',class_distribution=None,observation_ids=[],viewpoint_groups=[],
                  visibility='unknown',last_observation_timestamp_ms=None) for s in SQUARES}
    seen=set()
    if len({(o['position_id'],o['session_id'],o['source_video_sha256']) for o in observations})>1:raise ValueError('Cannot mix recording/position groups')
    for o in observations:
        validate_observation(o)
        identity=(o['source_video_sha256'],o['original_pts'],tuple(o['time_base']),o['canonical_square'],o['checkpoint_sha256'],o['crop_config_sha256'])
        if identity in seen:raise ValueError('Duplicate physical observation')
        seen.add(identity);r=result[o['canonical_square']];r['state']='candidate';r['observation_ids'].append(o['observation_id'])
        r['viewpoint_groups']=sorted(set(r['viewpoint_groups']+[o['semantic_orientation']]))
        r['last_observation_timestamp_ms']=max(o['timestamp_ms'],r['last_observation_timestamp_ms'] or 0)
    return result


def classification_metrics(truth, predictions):
    y=np.asarray(truth,dtype=int).ravel();p=np.asarray(predictions,dtype=int).ravel()
    if y.shape!=p.shape or not len(y) or (y<0).any() or (y>12).any() or (p<0).any() or (p>12).any():raise ValueError('Matching nonempty class IDs required')
    cm=np.zeros((13,13),dtype=int);np.add.at(cm,(y,p),1);occ=y!=0;empty=~occ
    per_class=[]
    for k in range(13):
        tp=int(cm[k,k]);support=int(cm[k].sum());pred=int(cm[:,k].sum())
        per_class.append(dict(class_id=k,support=support,predicted_count=pred,precision=tp/pred if pred else None,recall=tp/support if support else None))
    return dict(square_count=len(y),correct=int((y==p).sum()),overall_accuracy=float((y==p).mean()),occupied_count=int(occ.sum()),
        occupied_correct=int(((y==p)&occ).sum()),occupied_accuracy=float((y[occ]==p[occ]).mean()) if occ.any() else None,
        empty_count=int(empty.sum()),empty_false_positives=int((empty&(p!=0)).sum()),empty_false_positive_rate=float((p[empty]!=0).mean()) if empty.any() else None,
        occupied_to_empty_misses=int((occ&(p==0)).sum()),occupied_to_empty_miss_rate=float((p[occ]==0).mean()) if occ.any() else None,
        incorrect_piece_identities=int((occ&(p!=0)&(p!=y)).sum()),confusion_matrix=cm.tolist(),per_class=per_class)


def multi_view_metrics(truth, predictions):
    """GT-only diagnostic ceiling; never chooses an operational prediction."""
    y=np.asarray(truth,dtype=int);p=np.asarray(predictions,dtype=int)
    if y.shape!=(64,) or p.ndim!=2 or p.shape[1]!=64:raise ValueError('Canonical 64-square rows required')
    n=len(p)
    if not n:return dict(usable_views=0,status='no_verified_views',ever_correct_squares=None,persistently_wrong_squares=None,oracle_exact_board=None)
    base=classification_metrics(np.tile(y,(n,1)),p);correct=p==y[None];k=correct.sum(axis=0)
    records=[]
    for i,s in enumerate(SQUARES):
        records.append(dict(canonical_square=s,true_class_id=int(y[i]),correct_view_count=int(k[i]),usable_views=n,
            ever_correct=bool(k[i]>0),correct_in_every_view=bool(k[i]==n),correct_in_majority=bool(k[i]>n/2),
            correct_in_only_one_view=bool(k[i]==1),wrong_in_every_view=bool(k[i]==0),
            predictions=p[:,i].tolist()))
    ever=int((k>0).sum());wrong=int((k==0).sum())
    return dict(base,usable_views=n,status='verified_views_only',single_frame_errors=(64-correct.sum(axis=1)).tolist(),
        exact_single_frame_boards=int(correct.all(axis=1).sum()),ever_correct_squares=ever,ever_correct_fraction=ever/64,
        persistently_wrong_squares=wrong,oracle_incorrect_squares=wrong,oracle_exact_board=wrong==0,
        oracle_kind='ground_truth_diagnostic_upper_bound_not_fusion',
        multiplicity=dict(every_view=int((k==n).sum()),majority=int((k>n/2).sum()),only_one=int((k==1).sum()),wrong_every_view=wrong,
                          categories_can_overlap=True),squares=records,
        physical_class_coverage=[dict(class_id=c,physical_square_support=int((y==c).sum()),ever_correct=int(((y==c)&(k>0)).sum()),
                                      persistently_wrong=int(((y==c)&(k==0)).sum())) for c in range(13)])


def paired_errors(truth,a,b):
    y=np.asarray(truth);a=np.asarray(a);b=np.asarray(b);ea=a!=y;eb=b!=y
    if y.shape!=a.shape or a.shape!=b.shape or not y.size:raise ValueError('Matching prediction vectors required')
    n11=int((ea&eb).sum());n10=int((ea&~eb).sum());n01=int((~ea&eb).sum());n00=int((~ea&~eb).sum())
    denom=math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00));union=n11+n10+n01
    return dict(square_count=int(y.size),prediction_disagreement=int((a!=b).sum()),prediction_disagreement_rate=float((a!=b).mean()),
        shared_errors=n11,error_union=union,error_jaccard=n11/union if union else None,
        binary_error_phi=(n11*n00-n10*n01)/denom if denom else None,
        useful_one_correct_other_wrong=n10+n01,both_wrong_different_predictions=int((ea&eb&(a!=b)).sum()))


def descriptive_correlation(xs,ys):
    pairs=[(a,b) for a,b in zip(xs,ys) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs)<3:return dict(pair_count=len(pairs),pearson=None,interpretation='descriptive dependent pairs, no significance claim')
    x,y=np.asarray(pairs).T
    return dict(pair_count=len(pairs),pearson=float(np.corrcoef(x,y)[0,1]) if x.std()>1e-12 and y.std()>1e-12 else None,
                interpretation='descriptive dependent pairs within one physical session; not independent samples')


def validate_bundle(manifest, observations):
    if (manifest['schema_version']!='offline_square_evidence_d2_v1' or manifest['designation']!='development_only' or
        manifest['training_eligible'] or manifest['final_holdout'] or manifest['fusion_implemented'] or
        manifest['completion_enabled'] or manifest['ground_truth_used_for_inference']):raise ValueError('Invalid evidence scope')
    if len(observations)!=manifest['observation_count']:raise ValueError('Observation count mismatch')
    ids={o['observation_id'] for o in observations}
    if len(ids)!=len(observations):raise ValueError('Duplicate observation IDs')
    by_frame={}
    for o in observations:
        validate_observation(o)
        if o['checkpoint_sha256']!=manifest['model']['checkpoint_sha256'] or o['crop_config_sha256']!=manifest['config_sha256']:
            raise ValueError('Observation model/config mismatch')
        by_frame.setdefault((o['session_id'],o['frame_id']),[]).append(o)
    referenced=[]
    for f in manifest['frames']:
        actual=by_frame.get((f['session_id'],f['frame_id']),[])
        if f['observations']:
            if f['human_review']!='human_verified' or len(actual)!=64 or {o['canonical_square'] for o in actual}!=set(SQUARES):
                raise ValueError('Every inferred frame needs 64 verified canonical observations')
        elif actual:raise ValueError('Unreviewed frame contains observations')
        if set(f['observations'])!={o['observation_id'] for o in actual}:raise ValueError('Frame observation references mismatch')
        referenced.extend(f['observations'])
    if len(referenced)!=len(ids) or set(referenced)!=ids:raise ValueError('Orphan or multiply referenced observations')
    for s in manifest['sessions']:
        actual=[o for o in observations if o['session_id']==s['capture_session_id']]
        if s['squares']!=square_records(actual):raise ValueError('Canonical evidence containers do not match observations')
    return dict(valid=True,observations=len(observations),inferred_frames=len(by_frame))
