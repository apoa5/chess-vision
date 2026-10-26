"""Run only the four registered C.3 arms, then fixed scoring/profiling.

Completed arms may be reused after provenance checks. Partial/blocked arms are
never silently resumed, replaced or given a smaller budget.
"""
import argparse
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.vision.crop_training_c3 import read,sha


def run(root):
    registration=read(root/'registration.json');config=registration['config']
    logs=root/'logs';logs.mkdir(exist_ok=True)
    for seed,arm in config['execution_order']:
        status_path=root/'training'/f'{arm}_seed{seed}'/'status.json'
        if status_path.exists():
            status=read(status_path)
            if status['status']!='complete':raise ValueError('Existing arm is incomplete/running; wait or report blocker: '+str(status_path))
            if len(status['history'])!=config['epochs']+1 or status['config_sha256']!=registration['config_sha256']:
                raise ValueError('Completed arm budget/provenance differs')
            for label in ('best','last'):
                path=status_path.parent/(label+'.pt')
                if sha(path)!=status[label+'_checkpoint_sha256']:raise ValueError('Changed completed checkpoint')
            print('Reusing complete',arm,seed,flush=True)
            continue
        log=logs/f'{arm}_seed{seed}.log'
        if log.exists():raise ValueError('Do not overwrite prior execution logs')
        command=[sys.executable,'scripts/run_crop_representation_c3.py','train','--run',str(root),'--arm',arm,'--seed',str(seed)]
        print('Starting',arm,seed,'log:',log,flush=True)
        with log.open('w') as handle:subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
        print('Completed',arm,seed,flush=True)
    for seed in config['seeds']:
        a=read(root/'training'/f'b2_seed{seed}'/'status.json');b=read(root/'training'/f'taller_seed{seed}'/'status.json')
        if a['initial_state_sha256']!=b['initial_state_sha256']:raise ValueError('Unmatched paired initialization')
        if [h['sample_order_sha256'] for h in a['history']]!=[h['sample_order_sha256'] for h in b['history']]:raise ValueError('Unmatched sample orders')
    for seed,arm in config['execution_order']:
        for action in ('evaluate','profile'):
            finished=(root/'evaluation'/f'{arm}_seed{seed}'/'metrics.json') if action=='evaluate' else (root/'profiles'/f'{arm}_seed{seed}.json')
            if finished.exists():continue
            command=[sys.executable,'scripts/evaluate_crop_representation_c3.py',action,'--run',str(root),'--arm',arm,'--seed',str(seed)]
            log=logs/f'{action}_{arm}_seed{seed}.log'
            if log.exists():raise ValueError('Do not overwrite partial evaluation')
            print(action,arm,seed,flush=True)
            with log.open('w') as handle:subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,check=True)
    print('C3 fixed training/evaluation complete; no further experiments.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    run(parser.parse_args().run)
