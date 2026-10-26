"""D.3b descriptive reliability only; no prediction or scan-completion policy."""
import math
import numpy as np
from src.scanning.offline_fusion import entropy
from src.scanning.square_evidence import SQUARES

SIGNALS = {'maximum_probability': True, 'margin': True, 'entropy': False,
           'variation_ratio': False, 'js_disagreement': False,
           'predicted_class_dispersion': False, 'input_max_dispersion': False}


def uncertainty_signals(score, input_probabilities, predicted_class):
    p = np.asarray(score, dtype=float); views = np.asarray(input_probabilities, dtype=float)
    if p.shape != (13,) or views.ndim != 2 or views.shape[1] != 13 or not len(views):
        raise ValueError('Thirteen-class distributions and at least one view required')
    if not np.isfinite(p).all() or not np.isfinite(views).all() or (p < 0).any() or (views < 0).any() or not np.isclose(p.sum(), 1) or not np.allclose(views.sum(axis=1), 1):
        raise ValueError('Invalid probabilities')
    if type(predicted_class) is not int or not 0 <= predicted_class < 13 or p[predicted_class] != p.max():
        raise ValueError('Invalid winning class')
    labels = views.argmax(axis=1); counts = np.bincount(labels, minlength=13); mean = views.mean(axis=0)
    return dict(maximum_probability=float(p[predicted_class]), margin=float(np.sort(p)[-1]-np.sort(p)[-2]),
                entropy=entropy(p), variation_ratio=float(1-counts.max()/len(views)),
                js_disagreement=max(0., entropy(mean)-float(np.mean([entropy(v) for v in views]))),
                predicted_class_dispersion=float(views[:, predicted_class].std(ddof=0)),
                input_max_dispersion=float(views.max(axis=1).std(ddof=0)),
                predicted_class_min_input=float(views[:, predicted_class].min()),
                supporting_fraction=float(np.mean(labels == predicted_class)),
                strong_support_fraction=float(np.mean((labels == predicted_class) & (views.max(axis=1) >= .9))),
                unanimous_argmax=bool(np.all(labels == predicted_class)))


def risk_curve(rows, signal, higher_is_better):
    """Threshold-based curve: never split a score tie or use correctness to rank."""
    if not rows: return []
    if any(not math.isfinite(r[signal]) for r in rows): raise ValueError('Nonfinite ranking signal')
    values = sorted(set(r[signal] for r in rows), reverse=higher_is_better)
    points = [dict(threshold=None, retained=0, total=len(rows), coverage=0., errors=0, risk=None)]
    for value in values:
        kept = [r for r in rows if (r[signal] >= value if higher_is_better else r[signal] <= value)]
        errors = sum(not r['correct'] for r in kept)
        points.append(dict(threshold=value, retained=len(kept), total=len(rows), coverage=len(kept)/len(rows), errors=errors, risk=errors/len(kept)))
    return points


def matched_coverage(rows, signal, higher_is_better, coverage):
    """Exact descriptive budget; stable IDs break ties, and tie risk bounds are exposed."""
    if not rows or not 0 < coverage <= 1: raise ValueError('Nonempty rows and valid coverage required')
    if any(not math.isfinite(r[signal]) for r in rows): raise ValueError('Nonfinite ranking signal')
    ordered = sorted(rows, key=lambda r: ((-1 if higher_is_better else 1)*r[signal], r['session_id'], r['canonical_square']))
    if len({(r['session_id'], r['canonical_square']) for r in rows}) != len(rows): raise ValueError('Duplicate physical square')
    k = max(1, int(math.floor(len(rows)*coverage)))
    value = ordered[k-1][signal]; strict = [r for r in ordered if (r[signal] > value if higher_is_better else r[signal] < value)]
    tied = [r for r in ordered if r[signal] == value]; slots = k-len(strict)
    strict_errors = sum(not r['correct'] for r in strict); tie_errors = sum(not r['correct'] for r in tied)
    return dict(requested_coverage=coverage, retained=k, total=len(rows), actual_coverage=k/len(rows),
                errors=sum(not r['correct'] for r in ordered[:k]), risk=sum(not r['correct'] for r in ordered[:k])/k,
                boundary_tie_count=len(tied), boundary_tie_slots=slots,
                tie_risk_min=(strict_errors+max(0, slots-(len(tied)-tie_errors)))/k,
                tie_risk_max=(strict_errors+min(slots,tie_errors))/k,
                tie_break='session_id,canonical_square; correctness never used')


def board_summary(squares):
    if len(squares) != 64 or {s['canonical_square'] for s in squares} != set(SQUARES):
        raise ValueError('Exactly 64 unique square diagnostics required')
    return dict(mean_probability=float(np.mean([s['maximum_probability'] for s in squares])),
                minimum_probability=min(s['maximum_probability'] for s in squares),
                minimum_margin=min(s['margin'] for s in squares), maximum_entropy=max(s['entropy'] for s in squares),
                maximum_disagreement=max(s['js_disagreement'] for s in squares),
                withheld_squares=sum(s['abstained'] for s in squares),
                all_squares_retained=not any(s['abstained'] for s in squares),
                incorrect_squares=sum(not s['correct'] for s in squares),
                exact_board=all(s['correct'] for s in squares))
