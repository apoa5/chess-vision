"""C9 identical-input metrics and retrospective exposure stratification."""
import copy,time
from collections import Counter,defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.training.multidataset import position_key
from scripts.evaluate_multidataset_c4 import summarize


def evaluate(model,dataset,*,device='cuda',batch_size=32,class_weights=None):
    model.eval();scores=[];start=time.perf_counter();forward=0.
    with torch.inference_mode():
        for x,y in DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0):
            x=x.to(device);t=time.perf_counter();z=model(x)
            if device.startswith('cuda'):torch.cuda.synchronize()
            forward+=time.perf_counter()-t;scores.extend(z.cpu().float().numpy().tolist())
    return summarize_c9(dataset.rows,scores,class_weights=class_weights)|dict(elapsed_seconds=time.perf_counter()-start,forward_seconds=forward)


def summarize_c9(rows,scores,class_weights=None):
    views=[]
    for b,o in rows:
        view=dict(b);view['source_position_id']=b['position_id'];view['position_id']='placement_'+position_key(b['placement'])[:16];views.append((view,o))
    metrics=summarize(views,scores);errors=np.array([b['errors'] for b in metrics['per_board']]);hist=Counter(int(x) for x in errors)
    metrics['corrections_per_board']=dict(mean=float(errors.mean()),median=float(np.median(errors)),histogram=dict(sorted(hist.items())),
        categories={label:dict(count=int(mask.sum()),fraction=float(mask.mean())) for label,mask in [('0',errors==0),('1',errors==1),('2',errors==2),('3-5',(errors>=3)&(errors<=5)),('>5',errors>5)]})
    metrics['worst_errors_by_position']={p:max(b['errors'] for b in metrics['per_board'] if b['position_id']==p) for p in {b['position_id'] for b in metrics['per_board']}}
    z=np.asarray(scores,dtype=np.float64);y=np.array([o['class_id'] for b,o in rows]);exp=np.exp(z-z.max(1,keepdims=True));p=exp/exp.sum(1,keepdims=True)
    loss=-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1))
    if class_weights is not None:loss*=np.asarray(class_weights)[y]
    byposition=defaultdict(list)
    for i,(b,o) in enumerate(rows):byposition[position_key(b['placement'])].append(loss[i])
    metrics['macro_position_validation_loss']=float(np.mean([np.mean(v) for v in byposition.values()]))
    cm=np.array(metrics['confusion_matrix']);metrics['rook_to_king_errors']=int(cm[4,6]+cm[10,12])
    exposure={}
    for name,flag in [('unseen_by_B2_training_placement',False),('exposed_B2_training_placement',True)]:
        ix=[i for i,(b,o) in enumerate(rows) if b['b2_training_placement_exposed']==flag]
        exposure[name]=summarize([views[i] for i in ix],[scores[i] for i in ix]) if ix else None
    metrics['b2_exposure_strata']=exposure
    metrics['exact_training_image_observations']=sum(b['b2_exact_training_image_exposed'] for b,o in rows)
    return dict(metrics=metrics,observations=[dict(board_id=b['id'],position_id=b['position_id'],physical_position='placement_'+position_key(b['placement'])[:16],session_id=b['session_id'],
        dataset=b['dataset'],square=o['square'],truth=o['class_id'],logits=z,b2_training_placement_exposed=b['b2_training_placement_exposed'],semantic_orientation=b['semantic_orientation']) for (b,o),z in zip(rows,scores)])
