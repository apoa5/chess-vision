"""Label-based D.3 diagnostics only; never imported by prediction generation."""
import numpy as np
from src.chess.labels import CLASSES
from src.chess.board_state import BoardState
from src.chess.validation import validate_position
from src.chess.fen import GameState
from src.scanning.square_evidence import classification_metrics


def reliability(truth,scores,predictions,abstained,config):
    y=np.asarray(truth,dtype=int);p=np.asarray(scores,dtype=float);pred=np.asarray(predictions,dtype=int);abstained=np.asarray(abstained,dtype=bool)
    if p.shape!=(len(y),13) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1):raise ValueError('Invalid score vectors')
    confidence=p[np.arange(len(y)),pred];correct=pred==y;bins=[];ece=0.
    indices=np.minimum((confidence*config['calibration_bins']).astype(int),config['calibration_bins']-1)
    for i in range(config['calibration_bins']):
        keep=indices==i;n=int(keep.sum());acc=float(correct[keep].mean()) if n else None;conf=float(confidence[keep].mean()) if n else None
        if n:ece+=n/len(y)*abs(acc-conf)
        bins.append(dict(bin=i,count=n,accuracy=acc,mean_score=conf))
    true_score=p[np.arange(len(y)),y];target=np.eye(13)[y];accepted=~abstained
    selected=classification_metrics(y[accepted],pred[accepted]) if accepted.any() else None
    return dict(score_calibrated=False,mean_score=float(confidence.mean()),mean_wrong_score=float(confidence[~correct].mean()) if (~correct).any() else None,
        mean_correct_score=float(confidence[correct].mean()) if correct.any() else None,
        nll_with_floor=float(-np.log(np.maximum(true_score,config['nll_probability_floor'])).mean()),zero_true_score_count=int((true_score==0).sum()),
        below_nll_floor_count=int((true_score<config['nll_probability_floor']).sum()),multiclass_brier=float(np.square(p-target).sum(axis=1).mean()),
        ece=ece,reliability_bins=bins,high_score_wrong_count=int(((confidence>=config['confident_error_threshold'])&~correct).sum()),
        accepted_count=int(accepted.sum()),abstained_count=int(abstained.sum()),coverage=float(accepted.mean()),
        accepted_error_count=int((accepted&~correct).sum()),accepted_error_rate=float((~correct[accepted]).mean()) if accepted.any() else None,
        abstained_correct_count=int((abstained&correct).sum()),accepted_metrics=selected)


def legality(predictions):
    if len(predictions)!=64:raise ValueError('Exactly 64 predictions required')
    board=BoardState([[CLASSES[k].symbol for k in predictions[i:i+8]] for i in range(0,64,8)],'white_at_bottom')
    white=validate_position(board,GameState(side_to_move='w'));black=validate_position(board,GameState(side_to_move='b'))
    return dict(white_to_move=white,black_to_move=black,valid_under_either_assumed_turn=white['standard_chess_valid'] or black['standard_chess_valid'],
        note='Read-only standard-chess diagnostics, no target counts/history; either-turn validity is not reachability or recognition correctness. No repairs.')
