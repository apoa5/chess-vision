import pytest
import numpy as np

torch=pytest.importorskip('torch')
pytest.importorskip('torchvision')
from torch.utils.data import DataLoader,TensorDataset
from src.vision.training import EarlyStopping,confusion_metrics,run_epoch


def test_early_stopping_respects_patience_and_minimum_improvement():
    stopper=EarlyStopping(patience=2,min_delta=.01)
    assert stopper.update(1.)==(True,False)
    assert stopper.update(.995)==(False,False)
    assert stopper.update(.994)==(False,True)
    assert stopper.update(.98)==(True,False)


def test_metrics_do_not_hide_rare_piece_errors():
    matrix=np.zeros((13,13),int);matrix[0,0]=100;matrix[1,0]=1
    result=confusion_metrics(matrix)
    assert result['accuracy']> .98
    assert result['macro_f1']< .1
    assert result['per_class_recall'][1]==0


def test_epoch_loss_aggregation_is_correct_for_unequal_batches():
    logits=torch.randn(5,13);labels=torch.tensor([0,0,1,2,12]);weights=torch.arange(1,14).float()
    loader=DataLoader(TensorDataset(logits,labels),batch_size=2)
    model=torch.nn.Identity();result=run_epoch(model,loader,weights)
    expected=torch.nn.functional.cross_entropy(logits,labels,weight=weights)
    assert result['loss']==pytest.approx(float(expected),rel=1e-6)
    assert np.array(result['confusion_matrix']).sum()==5
    assert not model.training
