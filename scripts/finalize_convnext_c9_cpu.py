"""Finish a locked C9 evaluation after remote client failure; no training or cloud compute."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2,torch
from src.chess.labels import CLASS_NAMES
from src.training.dataset_c9 import C9Squares
from src.training.convnext_c9 import make_model
from src.training.trainer_c9 import fingerprint
from src.training.evaluation_c9 import evaluate
from src.training.manifest_tools import write_new
from src.training.multidataset import digest
from src.vision.classifier import create_classifier
ROOT=Path('outputs/evaluation/convnext_c9')

def verify_selection(lock,config,manifest,state,checkpoint_sha):
    if lock['checkpoint_sha256']!=checkpoint_sha or lock['plan_hash']!=fingerprint(config) or lock['dataset_fingerprint']!=fingerprint(manifest):raise ValueError('Selection checkpoint/config/dataset mismatch')
    if lock['selected_updates']!=state['updates'] or state['class_names']!=list(CLASS_NAMES) or state['plan_hash']!=lock['plan_hash']:raise ValueError('Selected state provenance mismatch')

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    out=ROOT/'export_v1';training=json.loads((out/'training_result.json').read_text());lock=json.loads((out/'selection_lock.json').read_text());config=json.loads(Path('configs/training/convnext_c9_authorized_v1.json').read_text());manifest=json.loads((ROOT/'frozen_manifest_v1.json').read_text())
    # No remote evaluation marker or output may have existed when the worker
    # stopped. This exact inventory is preserved separately from local outputs.
    inventory=json.loads((ROOT/'continuation_artifacts_v1.json').read_text())
    if any(Path(v['filename']).name in ('internal_test_one_attempt.json','convnext_internal_test.json','b2_internal_test.json') for v in inventory):raise ValueError('Remote test event already exists; do not rerun')
    if (out/'c9_result.json').exists():raise ValueError('C9 evaluation already finalized')
    path=out/'convnextv2_tiny_best.pt';state=torch.load(path,map_location='cpu',weights_only=True)
    verify_selection(lock,config,manifest,state,digest(path))
    if digest(ROOT/'frozen_manifest_v1.json')!=config['manifest_sha256']:raise ValueError('Frozen manifest changed')
    if training['stop_reason']!='early_stopping' or training['history'][-1]['epoch']+1!=training['completed_coverage_epochs']:raise ValueError('Expected completed early-stop receipt')
    selected=min(training['history'],key=lambda h:h['validation_loss'])
    if selected['updates']!=lock['selected_updates']:raise ValueError('Frozen validation-loss selection mismatch')
    # Verify portable originals first, then rebase context paths only. Pixel
    # hashes/labels/order remain enforced by the inherited B2 Dataset packer.
    for b in manifest['boards']:
        b['source']['path']=b['local_source_path'];b['geometry']['valid_mask_path']=b['local_mask_path']
    model=make_model().eval();model.load_state_dict(state['model_state_dict'])
    baseline_path=Path('outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt')
    if digest(baseline_path)!=config['baseline_checkpoint_sha256']:raise ValueError('B2 checkpoint changed')
    baseline_state=torch.load(baseline_path,map_location='cpu',weights_only=True)
    if baseline_state['class_names']!=list(CLASS_NAMES):raise ValueError('B2 class mismatch')
    baseline=create_classifier(pretrained=False,mode='fine_tune').eval();baseline.load_state_dict(baseline_state['model_state_dict'])
    start=time.perf_counter();evaluations={}
    for split in ('val','internal_test'):
        if split=='internal_test':
            write_new(out/'internal_test_one_attempt.json',dict(lock,backend='cpu FP32',batch_size=config['batch_size'],threads=2,reason='remote worker stopped after selection; no prior remote internal test'))
        dataset=C9Squares(manifest,split,allow_internal_test=split=='internal_test')
        for name,m in [('convnext',model),('b2',baseline)]:
            target=out/(name+'_'+split+'.json')
            if target.exists():raise ValueError('Existing evaluation; no automatic replay')
            print('Evaluating',name,split,len(dataset),'squares',flush=True)
            result=evaluate(m,dataset,device='cpu',batch_size=config['batch_size'],class_weights=training['class_weights'])
            result.update(backend='CPU FP32',cpu_threads=2,batch_size=config['batch_size'],checkpoint_sha256=digest(path if name=='convnext' else baseline_path),frozen_manifest_sha256=config['manifest_sha256'])
            write_new(target,result);evaluations[name+'_'+split]=result['metrics']
            print(name,split,json.dumps({k:result['metrics'][k] for k in ('overall_accuracy','occupied_accuracy','empty_false_positives','exact_boards')}),flush=True)
    training.update(evaluations=evaluations,selection_lock=lock,final_evaluation_backend='CPU FP32 batch32 threads2; same for both architectures',evaluation_elapsed_seconds=time.perf_counter()-start,
        gpu_final_evaluation_interrupted=True,additional_training=False,configured_12_cycles_completed=False,registered_early_stopping_reached=True)
    write_new(out/'c9_result.json',training)

if __name__=='__main__':main()
