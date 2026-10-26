"""C.2 result replay and protected-artifact checks; no inference or holdout access."""
import argparse
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.benchmark_chessqueries_c2 import read, configuration, load_cohort, checked
from scripts.report_chessqueries_c2 import metrics, paired
from scripts.select_video_keyframes import check_hashes
from src.vision.chessqueries_offline import sha256, map_outputs, WEIGHT_SHA


def run(output):
    if output.exists():raise ValueError('Fresh validation output required')
    config_path='configs/experiments/chessqueries_c2.json';config=configuration(config_path)
    result_path='docs/evaluation/chessqueries_zero_shot_c2_results_v1.json';result=read(result_path)
    _,frames,_=load_cohort(config)
    raw_path=result['predictions_path'];checked(raw_path,result['predictions_sha256'])
    raw=json.loads(gzip.decompress(Path(raw_path).read_bytes()))
    ps={r['frame_id']:r for r in raw['predictions']}
    assert len(ps)==len(raw['predictions'])==len(frames)==55
    replay={}
    for f in frames:
        p=ps[f['frame_id']];again=map_outputs([p['raw_logits']])
        assert p['predicted_class_ids']==again['predicted_class_ids']
        assert np.allclose(p['probabilities'],again['probabilities'],rtol=0,atol=1e-12)
        assert p['orientation_policy']==config['orientation_policy'] and p['model_eval'] and p['torch_inference_mode']
        assert p['frame_sha256']==f['frame_sha256'] and p['session_id']==f['session_id']
        replay[f['frame_id']]=dict(b2=f['baseline_class_ids'],cq=p['predicted_class_ids'])
    for model in ('b2','cq'):
        assert metrics(frames,replay,model)==result['aggregate'][model]
        for row in result['per_session']:
            assert metrics([f for f in frames if f['session_id']==row['session_id']],replay,model)==row[model]
    assert paired([f['truth'] for f in frames],[f['baseline_class_ids'] for f in frames],
                  [replay[f['frame_id']]['cq'] for f in frames])==result['paired']
    registration=read('outputs/evaluation/chessqueries_c2_run1/registration.json')
    check_hashes(registration['input_sha256']);checked(registration['smoke_path'],registration['smoke_sha256'])
    smoke=read(registration['smoke_path']);assert smoke['status']=='passed' and smoke['repeated_logits_exact']
    release=read(config['release']);checked(release['weights']['path'],WEIGHT_SHA)
    check_hashes({r['path']:r['sha256'] for r in release['sources']})
    protected=read('outputs/research/input_c1_audit_v1/preserved_inputs.json')['protected_sha256']
    c1_index_path='data/metadata/input_representation_c1_index_v1.json'
    protected.update(read(c1_index_path)['artifacts'])
    # Check historical digests rather than accepting current bytes as a new baseline.
    assert not any('holdout' in str(p).lower() for p in protected)
    check_hashes(protected)
    original_videos={s['source']['path']:s['source']['sha256'] for s in read('outputs/scanning/d3c_interim_v1/registration.json')['sessions']}
    check_hashes(original_videos)
    check_hashes(result['source_sha256'])
    status=dict(schema_version='chessqueries_zero_shot_c2_validation_v1',status='passed',
                replayed_frames=55,sessions=5,square_predictions_per_method=3520,
                corrected_label_version_preserved=True,model_sha256=WEIGHT_SHA,
                protected_historical_and_c1_files=len(protected),d3c_original_videos=len(original_videos),
                pinned_official_source_files=len(release['sources']),
                b2_predictions_reused=True,no_training=True,no_production_changes=True,
                final_holdout_access=False,larger_models_downloaded=False,
                probabilities_replayed_from_saved_logits=True,
                peak_process_rss_bytes=raw['environment']['memory']['peak_process_rss_bytes'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
