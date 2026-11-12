"""Summarize immutable C11 measurement artifacts; never selects on C10 labels."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.benchmark_convnext_c11 import ROOT,CONFIG,REGISTRY,read
from src.training.multidataset import digest
from src.training.multidataset import SQUARES
from src.training.manifest_tools import write_new

def main():
    selection=read(ROOT/'primary_selection.json');names=['reference']+read(ROOT/'finalists.json')['candidates'];rows=[]
    for n in names:
        screen=read(ROOT/'screens'/f'{n}.json');confirm=read(ROOT/f'{n}_confirmation.json');pipe=read(ROOT/f'{n}_pipeline.json')
        warm=[r['end_to_end_seconds'] for r in pipe['rows'] if r['repeat']==1 and r['completed']]
        f=read(ROOT/f'{n}_full_parity.json') if n!='reference' else None
        regression=read(ROOT/f'{n}_c10_regression.json') if n!='reference' else None
        cfg=screen['config'];model=cfg.get('onnx_path',cfg['checkpoint']);deltas=[r['parity']['max_abs_logit_error'] for r in regression['rows']] if regression else [0.]
        rows.append(dict(candidate=n,recommended=n==selection['candidate'],median_forward_seconds=confirm['median_forward_seconds'],
            std_forward_seconds=confirm['std_forward_seconds'],median_runtime_seconds=confirm['median_total_seconds'],
            median_warm_pipeline_seconds=float(np.median(warm)) if warm else None,model_initialization_seconds=confirm['initialization_seconds'],
            backend_import_seconds=confirm['backend_library_import_seconds'],peak_rss_bytes=confirm['peak_rss_bytes'],pipeline_peak_rss_bytes=pipe['peak_rss_bytes'],
            model_bytes=Path(model).stat().st_size,model_sha256=digest(model),development_top1_disagreements=f['top1_disagreements'] if f else 0,
            development_board_disagreements=f['board_disagreements'] if f else 0,development_max_abs_logit_error=f['max_abs_logit_error'] if f else 0,
            development_mean_abs_logit_error=f['mean_abs_logit_error'] if f else 0,c10_top1_disagreements=sum(r['parity']['top1_disagreements'] for r in regression['rows']) if regression else 0,
            c10_max_abs_logit_error=max(deltas),c10_regression_pass=regression['passed'] if regression else True,
            pipeline_geometry_and_prediction_parity=all(r['completed'] and not r['parity']['board_changed'] for r in pipe['rows'])))
    reference=read(ROOT/'reference_development.json');reference_values=np.load(ROOT/'reference_development_logits.npz')['logits']
    locations={r['id']:i for i,r in enumerate(reference['rows'])};edges=[]
    candidate_values={n:np.load(ROOT/f'{n}_development_logits.npz')['logits'] for n in names if n!='reference'}
    for case in read(REGISTRY)['edge_cases']:
        i=locations[case['board_id']];j=SQUARES.index(case['square']);base=reference_values[i,j];variants={}
        for n,z in candidate_values.items():
            values=z[i,j];variants[n]=dict(top1=int(values.argmax()),top2_logit_margin=float(np.sort(values)[-1]-np.sort(values)[-2]),max_abs_logit_error=float(np.max(np.abs(values-base))))
        edges.append(dict(**case,reference_top1=int(base.argmax()),reference_top2_logit_margin=float(np.sort(base)[-1]-np.sort(base)[-2]),variants=variants))
    write_new(ROOT/'edge_cases_v1.json',dict(schema_version='convnext_c11_edges_v1',rows=edges,selected_by_reference_margin_before_optimization=True))
    selected=next(r for r in rows if r['recommended'])
    accepted=selected['c10_regression_pass'] and selected['pipeline_geometry_and_prediction_parity']
    recommendation=selection['candidate'] if accepted else 'reference'
    for r in rows:r['recommended']=r['candidate']==recommendation
    # A failed regression veto retains the reference; never retune or choose
    # another optimized candidate from C10 outcomes.
    candidate_config=read(ROOT/'screens'/f'{recommendation}.json')['config']
    write_new(Path('configs/evaluation/convnext_c11_recommended_v1.json'),candidate_config)
    files=[REGISTRY,CONFIG,Path('configs/evaluation/convnext_c11_candidate_v1.json'),Path('configs/evaluation/convnext_c11_recommended_v1.json')]+[p for p in ROOT.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.onnx') and p.name!='results_v1.json']
    result=dict(schema_version='convnext_c11_results_v1',development_boards=41,development_squares=2624,c10_regression_boards=3,c10_regression_squares=192,
        selection=selection,recommended_candidate=recommendation,primary_regression_veto_pass=accepted,numeric_tolerance=read(ROOT/'finalists.json'),comparison=rows,integrity=read(ROOT/'integrity.json'),environment=read(ROOT/'environment.json'),
        tests=read(ROOT/'tests_summary_v1.json'),preprocessing_changed=False,production_changed=False,weights_changed=False,training=False,paid_cloud=False,
        artifacts_sha256={str(p):digest(p) for p in files},
        source_sha256={p:digest(p) for p in ['src/vision/convnext_runtime.py','src/training/convnext_c9.py','src/vision/crop_experiments.py',
            'src/vision/position_pipeline.py','scripts/benchmark_convnext_c11.py','scripts/export_convnext_c11.py','scripts/validate_convnext_c11.py',
            'scripts/report_convnext_c11.py','tests/test_convnext_runtime.py','requirements-inference-c11.txt']})
    write_new(ROOT/'results_v1.json',result)
    write_new(Path('data/metadata/convnext_c11_results_v1.json'),dict(schema_version='convnext_c11_index_v1',
        results_path=str(ROOT/'results_v1.json'),results_sha256=digest(ROOT/'results_v1.json'),recommended_candidate=recommendation,
        comparison=rows,tests=result['tests'],integrity=result['integrity'],source_sha256=result['source_sha256'],
        artifact_sha256=result['artifacts_sha256'],final_holdout_access=False,production_changed=False))
    lines=['| Runtime | Forward median ± SD (s) | Warm pipeline (s) | Load (s) | Peak RSS (MiB) | Model (MiB) | Changed squares (development / C10) | Max logit error (development / C10) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['candidate']} | {r['median_forward_seconds']:.3f} ± {r['std_forward_seconds']:.3f} | {r['median_warm_pipeline_seconds']:.3f} | {r['model_initialization_seconds']:.3f} | {r['peak_rss_bytes']/2**20:.0f} | {r['model_bytes']/2**20:.1f} | {r['development_top1_disagreements']} / {r['c10_top1_disagreements']} | {r['development_max_abs_logit_error']:.3g} / {r['c10_max_abs_logit_error']:.3g} |")
    table=ROOT/'comparison.md'
    if table.exists():raise ValueError('Comparison already exists')
    table.write_text('\n'.join(lines)+'\n');print(table.read_text())

if __name__=='__main__':main()
