import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('torchvision')
from src.vision.classifier import create_classifier,set_training_mode


def test_mask_input_gets_gradients_through_frozen_backbone():
    torch.set_num_threads(2);torch.manual_seed(42)
    model=create_classifier(pretrained=False)
    assert model.features[0][0].in_channels==4
    assert torch.count_nonzero(model.features[0][0].weight[:,3])==0
    assert model.features[0].training and not model.features[1].training
    assert model.features[0][0].weight.requires_grad
    assert not any(p.requires_grad for p in model.features[1].parameters())
    images=torch.randn(2,4,224,224);images[:,3]=(torch.rand(2,224,224)>.5).float()
    outputs=model(images);assert outputs.shape==(2,13)
    torch.nn.functional.cross_entropy(outputs,torch.tensor([2,8])).backward()
    gradient=model.features[0][0].weight.grad[:,3]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert model.classifier[-1].weight.grad is not None
    model.eval();set_training_mode(model,'head_and_stem')
    assert model.features[0].training and not model.features[1].training


def test_fine_tune_unfreezes_all_parameters():
    model=create_classifier(pretrained=False,mode='fine_tune')
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError):create_classifier(pretrained=False,mode='bad')
