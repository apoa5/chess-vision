"""Offline C9 model contract. Never imported by production recognition."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from src.chess.labels import CLASS_NAMES

MODEL_NAME='convnextv2_tiny.fcmae_ft_in22k_in1k'
REVISION='b1dd46230e80bf4cc3fa0c3c905db2c3ec53a817'
WEIGHT_SHA='6652fd90fc9c23977659e58515778e16fbbcd43f0b01fc693089cebe6d64c2a9'
WEIGHT_BYTES=114561694

def make_model(checkpoint=None):
    import timm
    if timm.__version__!='1.0.22':raise ValueError('C9 requires pinned timm1.0.22')
    model=timm.create_model(MODEL_NAME,pretrained=False,num_classes=1000)
    if checkpoint is not None:
        from src.training.multidataset import digest
        from safetensors.torch import load_file
        if Path(checkpoint).stat().st_size!=WEIGHT_BYTES or digest(checkpoint)!=WEIGHT_SHA:raise ValueError('Wrong pretrained checkpoint')
        model.load_state_dict(load_file(str(checkpoint)),strict=True)
    original=model.stem[0]
    if original.weight.shape!=(96,3,4,4) or tuple(original.stride)!=(4,4):raise ValueError('Unexpected ConvNeXt V2 stem')
    stem=nn.Conv2d(4,96,4,stride=4,bias=original.bias is not None)
    with torch.no_grad():
        stem.weight[:,:3].copy_(original.weight);stem.weight[:,3:].zero_()
        if original.bias is not None:stem.bias.copy_(original.bias)
    model.stem[0]=stem;model.reset_classifier(len(CLASS_NAMES))
    return model

def phase_parameters(model,phase,config):
    if phase not in ('head_stem','full'):raise ValueError('Unknown phase')
    groups={}
    for name,p in model.named_parameters():
        kind='stem' if name.startswith('stem.') else 'head' if name.startswith('head.') else 'backbone'
        p.requires_grad=phase=='full' or kind!='backbone'
        if not p.requires_grad:continue
        decay=p.ndim>1;key=(kind,decay)
        groups.setdefault(key,[]).append(p)
    return [dict(params=params,lr=config[phase+'_lr'][kind],weight_decay=config['weight_decay'] if decay else 0.,group_name=kind+('_decay' if decay else '_no_decay')) for (kind,decay),params in groups.items()]

def set_phase(model,phase):
    model.train()
    if phase=='head_stem':model.stages.eval()
