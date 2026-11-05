"""Future fixed grouped-validation evaluation; never reads test/final holdout."""
import argparse,json,resource,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.chess.labels import CLASS_NAMES
from src.training.manifest_tools import read,write_new
from src.training.multidataset import MultiDatasetSquares,digest
from src.vision.classifier import create_classifier
from src.scanning.square_evidence import classification_metrics

def summarize(rows,logits):
    y=np.array([o['class_id'] for b,o in rows]);z=np.asarray(logits,dtype=np.float64)
    if z.shape!=(len(y),13) or not np.isfinite(z).all():raise ValueError('Finite 13-class logits in observation order required')
    z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);pred=p.argmax(1);confidence=p.max(1)
    result=classification_metrics(y,pred);boards=[]
    for board in sorted({b['id'] for b,o in rows}):
        ix=[i for i,(b,o) in enumerate(rows) if b['id']==board];b=rows[ix[0]][0]
        complete=len(ix)==64 and len({rows[i][1]['square'] for i in ix})==64
        boards.append(dict(board_id=board,position_id=b['position_id'],session_id=b['session_id'],complete=complete,errors=int((pred[ix]!=y[ix]).sum()),exact=bool(complete and np.all(pred[ix]==y[ix]))))
    result.update(exact_boards=sum(b['exact'] for b in boards),complete_board_denominator=sum(b['complete'] for b in boards),per_board=boards,
        confident_errors_at_0_9=int(((pred!=y)&(confidence>=.9)).sum()),high_confidence_empty_false_positives=int(((y==0)&(pred!=0)&(confidence>=.9)).sum()),
        mean_confidence_when_wrong=float(confidence[pred!=y].mean()) if (pred!=y).any() else None,
        nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),confidence_interpretation='uncalibrated; 0.9 diagnostic only, not a safe operating threshold')
    cm=np.array(result['confusion_matrix']);result['king_to_rook_errors']=int(cm[6,4]+cm[12,10])
    for field in ('position_id','session_id','semantic_orientation'):
        groups={}
        for value in sorted({str(b.get(field)) for b,o in rows}):
            ix=[i for i,(b,o) in enumerate(rows) if str(b.get(field))==value]
            groups[value]=classification_metrics(y[ix],pred[ix])
        result['by_'+field]=groups
    return result

def evaluate(manifest_path,checkpoint,output,bundle_root=None,threads=2):
    if output.exists():raise ValueError('Fresh evaluation output required')
    torch.set_num_threads(threads);m=read(manifest_path)
    if bundle_root:
        for b in m['boards']:
            b['source']['path']=str(bundle_root/b['source']['path'])
            if 'valid_mask_path' in b['geometry']:b['geometry']['valid_mask_path']=str(bundle_root/b['geometry']['valid_mask_path'])
    ds=MultiDatasetSquares(m,'val');state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('class_names')!=list(CLASS_NAMES) or state.get('input_channels')!=4 or state.get('smoke_only'):raise ValueError('Full compatible checkpoint required; smoke metrics are not model improvement')
    model=create_classifier(pretrained=False,mode='fine_tune').eval();model.load_state_dict(state['model_state_dict'],strict=True)
    start=time.perf_counter();forward=0.;scores=[]
    with torch.inference_mode():
        for x,_ in DataLoader(ds,batch_size=16,shuffle=False,num_workers=0):
            t=time.perf_counter();z=model(x);forward+=time.perf_counter()-t;scores.extend(z.numpy().tolist())
    result=dict(schema_version='multidataset_c4_validation_v1',metrics=summarize(ds.rows,scores),checkpoint_sha256=digest(checkpoint),manifest_sha256=digest(manifest_path),
        elapsed_seconds=time.perf_counter()-start,cpu_forward_seconds=forward,cpu_forward_per_64_squares_seconds=forward/len(ds)*64,
        checkpoint_resume_bytes=checkpoint.stat().st_size,model_tensor_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        independent_generalization=False,split='development_validation',original_pilot_used=False,final_holdout_access=False,
        observations=[dict(board_id=b['id'],position_id=b['position_id'],session_id=b['session_id'],square=o['square'],truth=o['class_id'],logits=z) for (b,o),z in zip(ds.rows,scores)])
    write_new(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bundle-root',type=Path);p.add_argument('--threads',type=int,default=2);a=p.parse_args();evaluate(a.manifest,a.checkpoint,a.output,a.bundle_root,a.threads)
