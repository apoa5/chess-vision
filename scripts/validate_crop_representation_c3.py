"""C.3 frozen budgets, paired provenance, metric replay and preservation."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.vision.crop_training_c3 import read,sha,historical_inputs,checkpoint_key
from scripts.run_crop_representation_c3 import verify,write
from scripts.evaluate_crop_representation_c3 import score


def run(root,result_path,output):
    if output.exists():raise ValueError('Fresh validation output required')
    registration=read(root/'registration.json');config=registration['config'];result=read(result_path)
    verify(registration['immutable_inputs']);verify(registration['raw_source_sha256'])
    if sha(root/'input_hashes.json')!=registration['input_hashes_sha256']:raise ValueError('Input hashes changed')
    _,rows,_,images,_=historical_inputs(config)
    states={};total_seconds=0
    for seed in config['seeds']:
        for arm in ('b2','taller'):
            path=root/'training'/f'{arm}_seed{seed}'/'status.json';state=read(path);states[(seed,arm)]=state
            assert state['status']=='complete' and state['config_sha256']==registration['config_sha256']
            assert [h['epoch'] for h in state['history']]==list(range(config['epochs']+1))
            assert state['best_epoch']==min(range(config['epochs']+1),key=lambda e:checkpoint_key(state['history'][e]['validation'],e))
            for history in state['history'][1:]:
                assert int(np.sum(history['training']['confusion_matrix']))==7168
                assert int(np.sum(history['validation']['confusion_matrix']))==1600
            assert state['peak_rss_bytes']<=config['resource_limits']['max_peak_rss_bytes']
            assert state['elapsed_seconds']<=config['resource_limits']['max_seconds_per_arm']
            total_seconds+=state['elapsed_seconds']
            evaluation=read(root/'evaluation'/f'{arm}_seed{seed}'/'metrics.json')
            for label in ('best','last'):
                model_path=path.parent/(label+'.pt');assert sha(model_path)==state[label+'_checkpoint_sha256']
                m=evaluation['results'][label];assert sha(m['logits_path'])==m['logits_sha256']
                p=np.load(m['logits_path']).argmax(axis=1);replay=score(rows['val'],p,images)
                for key in replay:assert m[key]==replay[key],key
            reported=next(m for m in result['models'] if m['seed']==seed and m['arm']==arm)
            assert reported['best']==evaluation['results']['best'] and reported['last']==evaluation['results']['last']
        a,b=states[(seed,'b2')],states[(seed,'taller')]
        assert a['initial_state_sha256']==b['initial_state_sha256']
        assert [h['sample_order_sha256'] for h in a['history']]==[h['sample_order_sha256'] for h in b['history']]
    assert total_seconds<=config['resource_limits']['max_total_training_seconds']
    # No new full crop arrays on disk: native uint8 caches lived only in RAM.
    assert all('evaluation' in str(p.relative_to(root)) and p.name.endswith('_logits.npy') for p in root.rglob('*.npy'))
    protected=read('outputs/research/input_c1_audit_v1/preserved_inputs.json')['protected_sha256']
    for index in ('data/metadata/input_representation_c1_index_v1.json','data/metadata/chessqueries_c2_index_v1.json'):
        protected.update(read(index)['artifacts'])
    assert not any('holdout' in p.lower() for p in protected)
    verify(protected)
    videos={s['source']['path']:s['source']['sha256'] for s in read('outputs/scanning/d3c_interim_v1/registration.json')['sessions']}
    verify(videos);verify(result['source_sha256'])
    summary=dict(schema_version='crop_representation_c3_validation_v1',status='passed',matched_seeds=config['seeds'],
        epochs_per_arm=config['epochs'],training_arms=4,validation_boards=25,validation_positions=3,
        all_initialization_and_epoch_order_pairs_match=True,all_best_and_last_predictions_replayed=True,
        protected_historical_artifacts=len(protected),original_D3c_videos=len(videos),
        historical_native_sources=len(registration['raw_source_sha256']),new_native_crop_datasets_persisted=0,
        total_arm_seconds=total_seconds,training_or_selection_on_D2_D3c_or_holdout=False,
        production_changes=False,config_sha256=registration['config_sha256'])
    output.parent.mkdir(parents=True,exist_ok=True);write(output,summary);print(summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.run,a.results,a.output)
