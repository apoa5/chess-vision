"""D.3 deterministic, GT-blind offline fusion; no runtime integration or scan gate."""
import math
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.scanning.square_evidence import SQUARES, validate_observation
from src.scanning.video_keyframes import view_distance

METHODS=('single_quality','majority_vote','mean_probabilities','mean_logits','clustered_quality_median')


def validate_config(c):
    if tuple(c['methods'])!=METHODS or c['conditions']!=['uniform_5','diversity_5','diversity_10']:raise ValueError('Fixed registered comparisons required')
    if c['development_only'] is not True or c['production_integration'] or c['completion_enabled']:raise ValueError('Offline development only')
    for key in ('view_distance_threshold','temporal_cluster_seconds','quality_weight_floor','confident_error_threshold','nll_probability_floor'):
        if not isinstance(c[key],(int,float)) or not math.isfinite(c[key]) or c[key]<=0:raise ValueError('Invalid '+key)
    if c['quality_weight_floor']>1 or c['confident_error_threshold']>1 or c['nll_probability_floor']>=1:raise ValueError('Invalid bounds')
    a=c['abstention']
    if any(not math.isfinite(a[k]) or not 0<=a[k]<=1 for k in ('minimum_score','minimum_margin','minimum_cluster_support_fraction')):raise ValueError('Invalid abstention parameters')
    if type(a['minimum_supporting_clusters']) is not int or a['minimum_supporting_clusters']<1 or type(c['calibration_bins']) is not int or c['calibration_bins']<1:raise ValueError('Invalid counts')
    return c


def softmax(z):
    z=np.asarray(z,dtype=np.float64);x=np.exp(z-z.max());return x/x.sum()


def entropy(p):
    p=np.asarray(p,dtype=np.float64);return float(-np.sum(p*np.log(np.maximum(p,np.finfo(float).tiny))))


def checked_square(rows):
    if not rows:raise ValueError('No usable observations')
    keys=('position_id','session_id','source_video_sha256','canonical_square','checkpoint_sha256','crop_config_sha256')
    expected=tuple(rows[0][k] for k in keys);seen=set()
    for row in rows:
        validate_observation(row)
        if tuple(row[k] for k in keys)!=expected:raise ValueError('Cannot fuse different physical squares/sessions/models')
        physical=(row['source_video_sha256'],row['original_pts'],tuple(row['time_base']),row['canonical_square'])
        if physical in seen:raise ValueError('Duplicate physical observation')
        seen.add(physical)
        q=row['d1_selection_quality_metadata']['quality_score'];v=row['d1_selection_quality_metadata']['view_descriptor']
        if not math.isfinite(q) or not 0<=q<=1 or len(v)!=4 or not np.isfinite(v).all():raise ValueError('Invalid quality/view metadata')
    return sorted(rows,key=lambda r:(r['timestamp_ms'],r['frame_id'],r['observation_id']))


def groups(rows,c):
    """Fixed-order complete link; no single-link temporal chaining."""
    result=[]
    def close(a,b):
        return abs(a['timestamp_ms']-b['timestamp_ms'])<=1000*c['temporal_cluster_seconds'] or view_distance(a['d1_selection_quality_metadata'],b['d1_selection_quality_metadata'])<c['view_distance_threshold']
    for i,row in enumerate(rows):
        match=next((g for g in result if all(close(row,rows[j]) for j in g)),None)
        if match is None:result.append([i])
        else:match.append(i)
    return result


def choose_single_frame(frame_rows):
    """One frame for the whole board, no per-square confidence/GT selection."""
    if not frame_rows:raise ValueError('No eligible frames')
    return min(frame_rows,key=lambda f:(-f['quality_score'],f['timestamp_ms'],f['frame_id']))['frame_id']


def fuse_square(rows,method,c,*,single_frame_id=None):
    if method not in METHODS:raise ValueError('Unknown fusion method')
    rows=checked_square(rows);p=np.asarray([r['probabilities'] for r in rows]);z=np.asarray([r['raw_logits'] for r in rows]);labels=p.argmax(axis=1)
    gs=groups(rows,c);group_distributions=[];within_weights=[]
    for g in gs:
        w=np.clip([rows[i]['d1_selection_quality_metadata']['quality_score'] for i in g],c['quality_weight_floor'],1);w=w/w.sum()
        within_weights.append(w.tolist());group_distributions.append(np.sum(p[g]*w[:,None],axis=0))
    group_distributions=np.asarray(group_distributions);votes=np.bincount(labels,minlength=13);mean=p.mean(axis=0);tie=[];fallback=False;source=None
    if method=='single_quality':
        candidates=[i for i,r in enumerate(rows) if r['frame_id']==single_frame_id]
        if len(candidates)!=1:raise ValueError('Single frame must be fixed for this board')
        source=rows[candidates[0]]['observation_id'];score=p[candidates[0]].copy();pred=int(score.argmax())
    elif method=='majority_vote':
        score=votes/len(rows);tie=np.flatnonzero(votes==votes.max()).tolist();pred=min(tie,key=lambda k:(-mean[k],k))
    elif method=='mean_probabilities':score=mean;pred=int(score.argmax())
    elif method=='mean_logits':score=softmax(z.mean(axis=0));pred=int(score.argmax())
    else:
        score=np.median(group_distributions,axis=0)
        if score.sum()<=0:score=group_distributions.mean(axis=0);fallback=True
        score=score/score.sum();pred=int(score.argmax())
    order=np.sort(score);margin=float(order[-1]-order[-2]);support=int((group_distributions.argmax(axis=1)==pred).sum())
    strong=sorted(set(int(labels[i]) for i,r in enumerate(rows) if r['maximum_probability']>=c['abstention']['minimum_score']))
    reasons=[];a=c['abstention']
    if float(score[pred])<a['minimum_score']:reasons.append('low_score')
    if margin<a['minimum_margin']:reasons.append('small_margin')
    if method!='single_quality':
        if support<a['minimum_supporting_clusters']:reasons.append('insufficient_supporting_groups')
        if support/len(gs)<a['minimum_cluster_support_fraction']:reasons.append('group_disagreement')
        if len(strong)>1:reasons.append('contradictory_strong_observations')
    cluster_records=[dict(observation_ids=[rows[i]['observation_id'] for i in g],within_quality_weights=w,distribution=d.tolist(),argmax_class_id=int(d.argmax())) for g,w,d in zip(gs,within_weights,group_distributions)]
    return dict(canonical_square=rows[0]['canonical_square'],method=method,predicted_class_id=pred,predicted_class=CLASS_NAMES[pred],
        score_distribution=score.tolist(),score_kind='uncalibrated_'+('vote_fraction' if method=='majority_vote' else 'aggregation_score'),
        maximum_score=float(score[pred]),top_two_margin=margin,entropy_nats=entropy(score),
        observation_ids=[r['observation_id'] for r in rows],input_frame_ids=[r['frame_id'] for r in rows],input_class_ids=labels.tolist(),
        input_max_probabilities=[r['maximum_probability'] for r in rows],vote_counts=votes.tolist(),vote_ties=tie,
        vote_entropy_nats=entropy(votes/len(rows)),variation_ratio=float(1-votes.max()/len(rows)),
        jensen_shannon_disagreement_nats=max(0.,entropy(mean)-float(np.mean([entropy(x) for x in p]))),
        groups=cluster_records,heuristic_group_count=len(gs),supporting_groups=support,group_support_fraction=support/len(gs),
        supporting_input_argmax_count=int((labels==pred).sum()),strong_input_classes=strong,
        strong_conflict=len(strong)>1,abstained=bool(reasons),abstention_reasons=reasons,
        operational_class_id=None if reasons else pred,single_observation_id=source,median_zero_fallback=fallback,
        conditional_independence_assumed=False)


def fuse_board(observations,method,c):
    if not observations:raise ValueError('Empty board evidence')
    group_keys=('session_id','position_id','source_video_sha256','checkpoint_sha256','crop_config_sha256')
    if len({tuple(o[k] for k in group_keys) for o in observations})!=1:raise ValueError('Board identity mismatch')
    by_square={s:[] for s in SQUARES};by_frame={}
    for row in observations:
        if row['canonical_square'] not in by_square:raise ValueError('Invalid canonical square')
        by_square[row['canonical_square']].append(row)
        f=dict(frame_id=row['frame_id'],timestamp_ms=row['timestamp_ms'],quality_score=row['d1_selection_quality_metadata']['quality_score'])
        if row['frame_id'] in by_frame and by_frame[row['frame_id']]!=f:raise ValueError('Inconsistent frame metadata')
        by_frame[row['frame_id']]=f
    expected=set(by_frame)
    if any({r['frame_id'] for r in rows}!=expected for rows in by_square.values()):raise ValueError('Incomplete frame/canonical square mapping')
    single=choose_single_frame(list(by_frame.values()))
    squares=[fuse_square(by_square[s],method,c,single_frame_id=single) for s in SQUARES]
    return dict(session_id=observations[0]['session_id'],position_id=observations[0]['position_id'],method=method,
        input_frame_ids=sorted(by_frame,key=lambda fid:(by_frame[fid]['timestamp_ms'],fid)),input_frame_count=len(by_frame),
        single_frame_id=single if method=='single_quality' else None,squares=squares)
