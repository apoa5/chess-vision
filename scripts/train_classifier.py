"""Two-stage CPU baseline training with validation-selected checkpoints."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.vision.classifier import create_classifier
from src.vision.dataset import ContextSquareDataset,file_hash
from src.vision.training import run_epoch,EarlyStopping
from src.chess.labels import CLASSES,LABEL_VERSION


def save_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/week2/training'))
    args=parser.parse_args()
    config=load_config(args.config) if args.config else load_config();settings=config['classifier']
    os.environ.setdefault('TORCH_HOME',str(Path('models/cache').resolve()))
    torch.set_num_threads(settings['cpu_threads']);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    seed=settings['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    path=settings['dataset_paths']['metadata'];manifest=json.loads(path.read_text())
    if not manifest.get('training_ready') or manifest.get('preview_only'):raise ValueError('Dataset not approved and split for training.')
    dataset_sha=file_hash(path)
    train=ContextSquareDataset(path,'train',input_size=settings['input_size'],seed=seed,augmentation=settings['augmentation'])
    val=ContextSquareDataset(path,'val',input_size=settings['input_size'],seed=seed)
    train_loader=DataLoader(train,batch_size=settings['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
    val_loader=DataLoader(val,batch_size=settings['batch_size'],shuffle=False,num_workers=0)
    weights=torch.tensor([manifest['split_report']['class_weights'][c.name] for c in CLASSES],dtype=torch.float32)
    model=create_classifier();mode='head_and_stem'
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=settings['learning_rate'])
    stopper=EarlyStopping(settings['early_stopping_patience'],settings['early_stopping_min_delta'])
    destination=args.output_dir;destination.mkdir(parents=True,exist_ok=True)
    checkpoint=settings['checkpoint_path'];checkpoint.parent.mkdir(parents=True,exist_ok=True)
    if checkpoint.exists():raise ValueError(f'Checkpoint already exists; preserve or relocate it before starting a fresh run: {checkpoint}')
    history=[];start=time.time();best_loss=float('inf');best_epoch=None
    report={'status':'running','pid':os.getpid(),'started_at_unix':start,'architecture':settings['architecture'],
            'device':'cpu','torch_version':torch.__version__,'settings':{k:v for k,v in settings.items() if k not in ('dataset_paths','checkpoint_path')},
            'dataset_fingerprint':manifest['split_report']['dataset_fingerprint'],'manifest_sha256':dataset_sha,
            'train_samples':len(train),'validation_samples':len(val),'test_set_used':False,'epochs':history}
    save_json(destination/'status.json',report)
    try:
        for epoch in range(1,settings['epochs']+1):
            if file_hash(path)!=dataset_sha:raise ValueError('Dataset manifest changed during training.')
            if epoch==settings['head_epochs']+1:
                mode='fine_tune'
                for p in model.parameters():p.requires_grad=True
                optimizer=torch.optim.Adam(model.parameters(),lr=settings['fine_tune_learning_rate'])
                stopper.bad_epochs=0
            train.set_epoch(epoch-1);epoch_start=time.perf_counter()
            print(f'Epoch {epoch}/{settings["epochs"]} {mode} lr={optimizer.param_groups[0]["lr"]}',flush=True)
            train_metrics=run_epoch(model,train_loader,weights,optimizer=optimizer,mode=mode,
                                    progress=lambda step,total:print(f'  training {step}/{total} batches',flush=True))
            val_metrics=run_epoch(model,val_loader,weights)
            result={'epoch':epoch,'mode':mode,'learning_rate':optimizer.param_groups[0]['lr'],
                    'duration_seconds':time.perf_counter()-epoch_start,'train':train_metrics,'validation':val_metrics}
            history.append(result)
            # Save every genuinely lower loss, even if its small improvement doesn't reset patience.
            if val_metrics['loss']<best_loss:
                best_loss=val_metrics['loss'];best_epoch=epoch
                temporary=checkpoint.with_suffix('.tmp')
                torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'mode':mode,'validation':val_metrics,
                            'architecture':settings['architecture'],'input_channels':4,'input_size':settings['input_size'],
                            'label_version':LABEL_VERSION,'class_names':[c.name for c in CLASSES],
                            'dataset_fingerprint':report['dataset_fingerprint'],'manifest_sha256':dataset_sha,
                            'settings':report['settings']},temporary);temporary.replace(checkpoint)
            _,stop=stopper.update(val_metrics['loss'])
            report.update(best_epoch=best_epoch,best_validation_loss=best_loss,elapsed_seconds=time.time()-start)
            save_json(destination/'status.json',report)
            print(f'  train loss={train_metrics["loss"]:.4f} accuracy={train_metrics["accuracy"]:.2%}; val loss={val_metrics["loss"]:.4f} accuracy={val_metrics["accuracy"]:.2%} macro F1={val_metrics["macro_f1"]:.3f}; best epoch={best_epoch}',flush=True)
            if stop and epoch>settings['head_epochs']:
                report['stop_reason']='Validation loss plateaued.';break
        report.update(status='complete',elapsed_seconds=time.time()-start,checkpoint_path=str(checkpoint))
        save_json(destination/'status.json',report)
        save_json(config['data']['metadata']/'day11_training_summary.json',{k:v for k,v in report.items() if k!='pid'})
        print(f'Training finished; best epoch {best_epoch}; checkpoint {checkpoint}',flush=True)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),elapsed_seconds=time.time()-start)
        save_json(destination/'status.json',report);raise

if __name__=='__main__':main()
