"""MobileNetV3 Small baseline with RGB context and a trainable target-mask input."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from src.chess.labels import CLASSES


def create_classifier(*, pretrained=True, mode='head_and_stem'):
    if mode not in ('head_and_stem','fine_tune'):
        raise ValueError('Unknown training mode.')
    model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    original=model.features[0][0]
    stem=nn.Conv2d(4,original.out_channels,original.kernel_size,original.stride,
                   original.padding,bias=original.bias is not None)
    with torch.no_grad():
        stem.weight[:,:3].copy_(original.weight)
        stem.weight[:,3:].zero_()
        if original.bias is not None:stem.bias.copy_(original.bias)
    model.features[0][0]=stem
    model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(CLASSES))
    if mode=='head_and_stem':
        for parameter in model.features.parameters():parameter.requires_grad=False
        for parameter in model.features[0].parameters():parameter.requires_grad=True
    set_training_mode(model,mode)
    return model


def set_training_mode(model,mode):
    model.train()
    if mode=='head_and_stem':
        # Keep pretrained batch-normalization statistics in the frozen blocks.
        for block in list(model.features)[1:]:block.eval()
