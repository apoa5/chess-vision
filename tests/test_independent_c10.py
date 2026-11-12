"""C10 evaluation contracts with synthetic references; no holdout or training."""
import copy,hashlib
import numpy as np
import pytest
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import SQUARES,placement_labels
from src.vision.independent_c10 import (validate_fen,independence,validate_manifest,verify_checkpoint,canonical_inputs,score,cohort_metrics)
FEN='4k3/8/8/8/8/8/8/4K3 w - - 0 1'

def logits_for(y):
    z=np.full((64,13),-4.);z[np.arange(64),y]=4.;return z

def test_full_owner_fen_required():
    assert validate_fen(FEN)=='4k3/8/8/8/8/8/8/4K3'
    for bad in ('8/8/8/8/8/8/8/8 w - - 0 1',FEN.split()[0],'not a FEN'):
        with pytest.raises(ValueError):validate_fen(bad)

def test_independence_uses_placement_not_game_fields_and_source_identity():
    historical=[dict(id='trained',placement=FEN,split='train',source_hashes=['a'*64])]
    result=independence(FEN.replace('w - - 0 1','b - - 4 20'),'b'*64,historical)
    assert not result['independent_eligible'] and result['c9_train_placement_matches']==['trained']
    fresh='4k3/8/8/8/8/8/P7/4K3 w - - 0 1'
    assert independence(fresh,'b'*64,historical)['independent_eligible']
    assert not independence(fresh,'a'*64,historical)['independent_eligible']

def test_checkpoint_hash_and_class_mapping_fail_closed(tmp_path):
    p=tmp_path/'weights.pt';p.write_bytes(b'weights');sha=hashlib.sha256(p.read_bytes()).hexdigest()
    state=dict(class_names=list(CLASS_NAMES),input_channels=4,input_size=224,updates=1578,representation='b2-context1-224-v1')
    verify_checkpoint(p,sha,state,'convnext')
    for bad in (dict(state,input_channels=3),dict(state,class_names=list(CLASS_NAMES)[::-1]),dict(state,updates=3682)):
        with pytest.raises(ValueError):verify_checkpoint(p,sha,bad,'convnext')
    with pytest.raises(ValueError):verify_checkpoint(p,'0'*64)

@pytest.mark.parametrize('turns',range(4))
def test_camera_pixels_preserved_during_canonical_reordering(turns):
    from src.vision.dataset import chess_square_at
    camera=[dict(canonical_square=chess_square_at(i//8,i%8,turns),packed=object()) for i in range(64)]
    ordered=canonical_inputs(camera)
    assert [i['canonical_square'] for i in ordered]==list(SQUARES)
    assert all(o['packed'] is next(c['packed'] for c in camera if c['canonical_square']==o['canonical_square']) for o in ordered)
    bad=copy.copy(camera);bad[0]=camera[1]
    with pytest.raises(ValueError):canonical_inputs(bad)

def test_exact_requires_every_piece_not_merely_valid_fen():
    y=placement_labels(FEN);z=logits_for(y);perfect=score(y,z)
    assert perfect['metrics']['exact_board'] and perfect['metrics']['corrections']==0
    z[0,0]=-4.;z[0,4]=8.;result=score(y,z)
    assert not result['metrics']['exact_board'] and result['metrics']['corrections']==1
    assert result['metrics']['empty_false_positives']==1
    assert result['metrics']['confident_empty_false_positives']==1
    assert result['observations'][0]['square']=='a8'
    assert len(result['observations'][0]['probabilities'])==13

def test_color_type_overlap_not_double_counted_as_corrections():
    y=placement_labels(FEN);z=logits_for(y)
    for index,pred in [(0,4),(4,5),(60,0)]:z[index]=-4.;z[index,pred]=8.
    m=score(y,z)['metrics']
    assert m['corrections']==3 and m['occupied_errors']==2
    assert m['empty_false_positives']==1 and m['occupied_to_empty_misses']==1
    assert m['wrong_colors']==1 and m['wrong_piece_types']==1

def test_failed_geometry_stays_in_pipeline_denominator():
    result=score(placement_labels(FEN),logits_for(placement_labels(FEN)))
    rows=[dict(completed=True,independent_eligible=True,**result),dict(completed=False,independent_eligible=True,failure_stage='localization'),dict(completed=False,independent_eligible=True,failure_stage='orientation')]
    m=cohort_metrics(rows,3)
    assert m['exact_boards']==1 and m['exact_board_denominator']==3
    assert m['scored_square_count']==64 and m['failed_or_excluded_photos']==2
    rows[0]['independent_eligible']=False
    m=cohort_metrics(rows,3)
    assert m['square_metrics'] is None and m['exact_boards']==0
    assert m['independent_exact_board_denominator']==2

def test_invalid_logits_rejected():
    z=logits_for(placement_labels(FEN));z[0,0]=np.nan
    with pytest.raises(ValueError):score(placement_labels(FEN),z)
    with pytest.raises(ValueError):score(placement_labels(FEN),np.zeros((64,12)))

def test_manifest_provenance_and_original_hashes(tmp_path):
    fens=[FEN,'4k3/8/8/8/8/8/P7/4K3 w - - 0 1','4k3/8/8/8/8/8/1P6/4K3 w - - 0 1'];photos=[]
    for i,fen in enumerate(fens):
        p=tmp_path/str(i);p.write_bytes(str(i).encode());photos.append(dict(id=str(i),path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),fen=fen,ground_truth=dict(source='owner_supplied_reference_FEN')))
    m=dict(training_eligible=False,final_holdout_access=False,class_names=list(CLASS_NAMES),photos=photos);validate_manifest(m)
    bad=copy.deepcopy(m);bad['training_eligible']=True
    with pytest.raises(ValueError):validate_manifest(bad)
    bad=copy.deepcopy(m);bad['photos'][0]['ground_truth']['source']='model_prediction'
    with pytest.raises(ValueError):validate_manifest(bad)
    Path=__import__('pathlib').Path;Path(photos[0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):validate_manifest(m)
