"""Training/validation metrics and early stopping; no held-out test evaluation."""
from dataclasses import dataclass
import numpy as np
import torch
from src.vision.classifier import set_training_mode


def confusion_metrics(confusion):
    matrix=np.asarray(confusion,dtype=np.int64)
    true=matrix.sum(axis=1);predicted=matrix.sum(axis=0);correct=matrix.diagonal()
    precision=np.divide(correct,predicted,out=np.zeros(13,dtype=float),where=predicted>0)
    recall=np.divide(correct,true,out=np.zeros(13,dtype=float),where=true>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(13),where=(precision+recall)>0)
    return {'accuracy':float(correct.sum()/max(matrix.sum(),1)),'macro_f1':float(f1.mean()),
            'per_class_recall':recall.tolist(),'per_class_f1':f1.tolist(),'confusion_matrix':matrix.tolist()}


@dataclass
class EarlyStopping:
    patience:int=5
    min_delta:float=1e-4
    best:float=float('inf')
    bad_epochs:int=0

    def update(self,loss):
        improved=loss<self.best-self.min_delta
        if improved:self.best=loss;self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=self.patience


def run_epoch(model,loader,weights,*,optimizer=None,mode='head_and_stem',progress=None):
    if optimizer is None:model.eval()
    else:set_training_mode(model,mode)
    confusion=np.zeros((13,13),np.int64);loss_sum=0.;weight_sum=0.
    with torch.set_grad_enabled(optimizer is not None):
        for step,(images,labels) in enumerate(loader,1):
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            logits=model(images)
            weighted_loss=torch.nn.functional.cross_entropy(logits,labels,weight=weights,reduction='sum')
            denominator=weights[labels].sum()
            loss=weighted_loss/denominator
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss; training stopped.')
            if optimizer is not None:
                loss.backward();optimizer.step()
            loss_sum+=float(weighted_loss.detach());weight_sum+=float(denominator)
            predictions=logits.argmax(dim=1)
            confusion+=torch.bincount(labels*13+predictions,minlength=169).reshape(13,13).cpu().numpy()
            if progress and (step%50==0 or step==len(loader)):progress(step,len(loader))
    return {'loss':loss_sum/weight_sum,**confusion_metrics(confusion)}
