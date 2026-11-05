"""Export compact C7 metadata, class table and learning curves from frozen results."""
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.manifest_tools import write_new
from src.training.multidataset import digest,position_key
from src.training.personal_c7 import PARENT_SHA
from src.chess.labels import CLASS_NAMES

ROOT=Path('outputs/evaluation/personal_c7')

def main():
    comparison=json.loads((ROOT/'comparison_v1.json').read_text())
    before=json.loads((ROOT/'billing_before_frozen_v1.json').read_text());after=json.loads((ROOT/'billing_after_v1.json').read_text())
    resource=json.loads((ROOT/'resources_after_v1.json').read_text());mine=[r for r in resource if r.get('description')=='chess-vision-c7']
    costs={'aggregate_metered_delta_usd':str(Decimal(after['metered_cost'])-Decimal(before['metered_cost'])),
           'app_specific_collected_resources_usd':str(sum(Decimal(r['cost']) for r in mine)),
           'budget_usd':3,'remaining_credits_verified':False,'workspace_hard_cap_verified':False,
           'billing_reconciliation':'Snapshot-dependent aggregate/resource totals differ; not a finalized invoice. Use the larger $.36 observed delta conservatively; storage may continue.'}
    dataset=json.loads((ROOT/'bundle_v1/manifest.json').read_text())
    boards=[]
    for b in dataset['boards']:
        boards.append({k:b[k] for k in ['id','dataset','position_id','session_id','placement','semantic_orientation','split','representation','original_fen']} | {
            'source':b['original_source'],'derived_context':b['source'],'geometry_status':b['geometry']['status'],
            'annotation_status':b['annotation']['status'],'review_checks':b['annotation'].get('checks'),
            'original_frame':b.get('original_frame'),'square_observations':len(b['observations'])})
    manifest={'schema_version':'personal_c7_dataset_v1','representation':dataset['representation'],'class_names':list(CLASS_NAMES),
              'bundle_manifest_path':str(ROOT/'bundle_v1/manifest.json'),'bundle_manifest_sha256':digest(ROOT/'bundle_v1/manifest.json'),
              'inventory_sha256':digest(ROOT/'bundle_v1/bundle_inventory.json'),'boards':boards,
              'counts':{'photo_train':112,'photo_val':25,'video_train_frames':77,'video_square_observations':4928,'video_positions':9,'quarantined_position_03_frames':10},
              'data_use_status':'retired_D3c_cohort_used_for_C7_training_never_independent_for_these_models',
              'future_independent_evaluation_required':True,
              'training_placements':len({position_key(b['placement']) for b in dataset['boards'] if b['split']=='train'}),
              'validation_placements':3,'original_pilot_excluded':True,'historical_test_images_used':False,'final_holdout_access':False,
              'review_replay':json.loads((ROOT/'review_replay_v1.json').read_text()),'exposure':json.loads((ROOT/'frame_exposure_v1.json').read_text())}
    write_new(Path('data/metadata/personal_finetuning_c7_dataset_v1.json'),manifest)
    training={k:json.loads((ROOT/(k+'_result_v1.json')).read_text()) for k in 'BC'}
    results={'schema_version':'personal_finetuning_c7_results_v1','development_only':True,'parent_sha256':PARENT_SHA,
             'comparison':comparison,'training':training,'costs':costs,
             'cpu_timing':json.loads((ROOT/'cpu_timing_v1.json').read_text()),
             'checkpoint_contracts':json.loads((ROOT/'checkpoint_contracts_v1.json').read_text()),
             'preservation':json.loads((ROOT/'integrity_v1.json').read_text()),
             'recommendation':'Retain frozen B.2; neither new model merits promotion. No further training authorized.',
             'primary_evaluation_runtime':'CPU FP32, batch16, threads2, unchanged B2 preprocessing; 25 photos/3 positions',
             'gpu_cpu_discrepancy':'B has a net one-square confusion-matrix difference; C matches. Per-square GPU logits were not preserved, so the exact B disagreement cannot be certified from aggregate matrices.'}
    write_new(Path('data/metadata/personal_finetuning_c7_results_v1.json'),results)
    path=Path('docs/evaluation/personal_finetuning_c7_classes_v1.csv')
    if path.exists():raise ValueError('Fresh class CSV required')
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['model','class_id','class_name','support','predicted_count','precision','recall']);writer.writeheader()
        for name,m in comparison['models'].items():
            for row in m['metrics']['per_class']:writer.writerow({'model':name,'class_name':CLASS_NAMES[row['class_id']],**row})
    print('Exported versioned dataset/results metadata, class CSV')

if __name__=='__main__':main()
