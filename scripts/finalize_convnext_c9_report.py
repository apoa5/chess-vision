"""Build C9 result tables/index from completed receipts; no model execution."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.training.multidataset import digest
from src.training.manifest_tools import write_new
ROOT=Path('outputs/evaluation/convnext_c9')

def percent(v):return f'{100*v:.2f}%'

def main():
    comparison=json.loads((ROOT/'comparison_v1.json').read_text());training=json.loads((ROOT/'export_v1/c9_result.json').read_text());cpu={n:json.loads((ROOT/f'cpu_{n}_v1.json').read_text()) for n in ('b2','convnext')}
    conditions=comparison['conditions'];data=json.loads((ROOT/'data_summary_v1.json').read_text());ledger=json.loads((ROOT/'billing_after_continuation_v1.json').read_text());cost=sum(float(v['cost']) for v in ledger if v.get('description')=='chess-vision-c9')
    lines=['## Final matched results (CPU FP32)','',
        'Both architectures use the same frozen square inputs, labels and ordering, batch32 and two CPU threads. Internal testing ran once after the remote selection lock; no test outcome changed weights, thresholds, eligibility or configuration. The selected validation confusion counts match the saved GPU epoch2 results. Figures count board **views**, not independent sessions/positions.','',
        '| Partition/model | Overall | Occupied | Empty FP / empty support | Occupied→empty | Wrong identity | Exact board views |',
        '|---|---:|---:|---:|---:|---:|---:|']
    summaries={}
    for split in ('val','internal_test'):
        for name in ('b2','convnext'):
            item=conditions[name+'_'+split];m=item['metrics'];summaries[name+'_'+split]={k:m[k] for k in ('overall_accuracy','occupied_accuracy','empty_count','empty_false_positives','empty_false_positive_rate','occupied_to_empty_misses','incorrect_piece_identities','exact_boards','complete_board_denominator','king_to_rook_errors','rook_to_king_errors','corrections_per_board','confident_errors_at_0_9','high_confidence_empty_false_positives','mean_confidence_when_wrong')}
            lines.append(f'| {split}/{name} | {percent(m["overall_accuracy"])} | {percent(m["occupied_accuracy"])} | {m["empty_false_positives"]}/{m["empty_count"]} ({percent(m["empty_false_positive_rate"])}) | {m["occupied_to_empty_misses"]} | {m["incorrect_piece_identities"]} | {m["exact_boards"]}/{m["complete_board_denominator"]} |')
    lines+=['','### B.2 training-exposure strata','',
        'B.2 previously trained on all20 new validation photos and10 internal-test photos. Its “unseen” stratum means unseen **training placement**, not never inspected during earlier development. ConvNeXt sees no new validation/test placement in its C.9 training partition. This is not a clean architecture-only experiment: training data, optimization and model capacity all differ.','',
        '| Partition/stratum/model | Occupied | Overall | Exact views | Empty FP |','|---|---:|---:|---:|---:|']
    for split in ('val','internal_test'):
        for label in ('unseen_by_B2_training_placement','exposed_B2_training_placement'):
            for name in ('b2','convnext'):
                m=conditions[name+'_'+split]['metrics']['b2_exposure_strata'][label]
                if m:lines.append(f'| {split}/{label}/{name} | {percent(m["occupied_accuracy"])} | {percent(m["overall_accuracy"])} | {m["exact_boards"]}/{m["complete_board_denominator"]} | {m["empty_false_positives"]} |')
    lines+=['','### Physical-position results','',
        'Each row represents one canonical physical placement. Multiple photos/video views remain correlated. Only four positions occur in each partition; no significance test or broad generalization claim is justified.','',
        '| Partition/position source IDs | Views | B.2 occupied | ConvNeXt occupied | B.2 exact | ConvNeXt exact | ConvNeXt worst errors/view |','|---|---:|---:|---:|---:|---:|---:|']
    manifest=json.loads((ROOT/'frozen_manifest_v1.json').read_text())
    for split in ('val','internal_test'):
        a,b=[conditions[n+'_'+split]['metrics'] for n in ('b2','convnext')]
        for position in sorted(b['by_position_id']):
            boards=[v for v in b['per_board'] if v['position_id']==position];ids={v['board_id'] for v in boards};sources=sorted({v['position_id'] for v in manifest['boards'] if v['id'] in ids})
            lines.append(f'| {split}/{", ".join(sources)} | {len(boards)} | {percent(a["by_position_id"][position]["occupied_accuracy"])} | {percent(b["by_position_id"][position]["occupied_accuracy"])} | {sum(v["exact"] for v in a["per_board"] if v["position_id"]==position)} | {sum(v["exact"] for v in boards)} | {max(v["errors"] for v in boards)} |')
    lines+=['','### Correction burden and paired changes','',
        '| Partition/model | Mean errors/view | Median | 0 corrections | 1 | 2 | 3–5 | >5 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for split in ('val','internal_test'):
        for name in ('b2','convnext'):
            m=conditions[name+'_'+split]['metrics']['corrections_per_board'];counts=[m['categories'][k]['count'] for k in ('0','1','2','3-5','>5')]
            lines.append(f'| {split}/{name} | {m["mean"]:.3f} | {m["median"]:.0f} | '+' | '.join(map(str,counts))+' |')
    for split,v in comparison['paired_changes'].items():lines+=['',f'{split}: ConvNeXt corrects **{v["corrected"]}** B.2 errors and introduces **{v["regressed"]}** errors on previously correct squares; **{v["both_wrong"]}** squares remain wrong for both.']
    lines+=['','### Classes, confidence and viewing conditions','',
        '[Full per-class precision/recall](convnext_c9_classes_v1.csv); raw confusion matrices, per-session/camera-side metrics, declared-photo-angle/lighting/source strata, and all13 logits per observation are retained in the comparison/export artifacts. Declared angles are historical metadata, not newly calibrated poses.','',
        '| Partition/model | King→rook | Rook→king | Wrong at probability≥.9 | Empty FP at probability≥.9 | Mean probability when wrong |','|---|---:|---:|---:|---:|---:|']
    for split in ('val','internal_test'):
        for name in ('b2','convnext'):
            m=conditions[name+'_'+split]['metrics'];lines.append(f'| {split}/{name} | {m["king_to_rook_errors"]} | {m["rook_to_king_errors"]} | {m["confident_errors_at_0_9"]} | {m["high_confidence_empty_false_positives"]} | {percent(m["mean_confidence_when_wrong"] or 0)} |')
    lines+=['','Raw softmax is uncalibrated; .9 is an illustrative diagnostic, not a safe acceptance rule. Wrong confident predictions remain grounds for assisted board review. No automatic scan-completion or legality repair was added.','',
        '[Compact grid/digital-board/crop comparison](../../outputs/evaluation/convnext_c9/visual_v1/index.html) shows corrections, regressions, kings/rooks and empty false positives. Context views are rectified B.2 board contexts; the original native-image geometry packet remains available at `outputs/evaluation/convnext_c9/review_v1/index.html`. Pixel/mask eligibility is fixed across models. A change of prediction on the same crop is a classifier outcome; it does not prove geometry improved or that truncation/neighbor interference was eliminated.','',
        '### Learning curves and inference cost','',
        '![Training and validation loss](convnext_c9_figures/learning_curves_v1.svg)','',
        '| Model | Parameters | Best checkpoint | CPU forward/64 squares | Peak process RSS | Loading |','|---|---:|---:|---:|---:|---:|']
    for name,v in cpu.items():lines.append(f'| {name} | {v["parameter_count"]:,} | {v["checkpoint_bytes"]/1e6:.2f} MB | {v["median_forward_per_64_squares_seconds"]:.3f}s | {v["peak_process_rss_bytes"]/1e6:.0f} MB | {v["loading_seconds"]:.3f}s |')
    lines+=['',f'CPU timing: {cpu["convnext"]["cpu_model"]}, two threads, FP32 inference mode, batch16, two warmup board passes and five measured passes on one validation board. No internal-test inference was repeated for timing. Source preprocessing is recorded separately; these forward-only numbers exclude geometry/video decoding. ConvNeXt is {cpu["convnext"]["median_forward_per_64_squares_seconds"]/cpu["b2"]["median_forward_per_64_squares_seconds"]:.1f}× slower than B.2 in this bounded measurement. No browser/mobile deployment or production latency is claimed.','',
        '### Recommendation and boundaries','',
        '**Promote this ConvNeXt checkpoint for further validation, not automatic production replacement.** The fixed C.9 run provides strong retrospective evidence of improved recognition on these held-out placements, while the exposed-photo stratum, small number of positions, single physical set and nighttime video domain limit interpretation. It does not isolate architecture capacity from expanded-data/training effects. A future matched MobileNet-on-C.9 control could separate those effects, but no further experiment was launched.','',
        'The next model-validation step is a fixed-checkpoint comparison on genuinely new sessions, with no tuning to their outcomes. The assisted MVP can continue independently with explicit user confirmation/correction. ConvNeXt CPU cost and the pretrained weights’ noncommercial restriction require an explicit deployment decision before any integration. No additional data collection, training, checkpoint promotion or production change begins automatically.','']
    report=Path('docs/evaluation/convnext_v2_c9.md');text=report.read_text()
    if '## Final matched results (CPU FP32)' in text:raise ValueError('Final report already written')
    text=text.replace('Final matched CPU evaluation is in progress; no convergence or independent-generalization claim is made.','The matched CPU validation and once-only retrospective internal test are complete. No convergence or independent-generalization claim is made.')
    for line in ['Freeze selected model and complete the once-only retrospective internal test.','Report matched B.2 comparisons with historical exposure caveats, correction burden, confidence and examples.','Verify final preservation and stop without production promotion.']:text=text.replace('- [ ] '+line,'- [x] '+line)
    text=text.replace('No model-selection recommendation is made before results exist.','The recommendation below uses the fixed model and completed retrospective results.')
    report.write_text(text+'\n'+'\n'.join(lines))
    artifacts=[report,Path('configs/training/convnext_c9_authorized_v1.json'),ROOT/'comparison_v1.json',ROOT/'export_receipt_v1.json',ROOT/'local_checkpoint_verification_v1.json',ROOT/'integrity_final_v1.json',ROOT/'data_summary_v1.json',ROOT/'cpu_convnext_v1.json',ROOT/'cpu_b2_v1.json',ROOT/'export_v1/selection_lock.json']
    result=dict(schema_version='convnext_c9_results_v1',independent_generalization=False,selection_lock=training['selection_lock'],summaries=summaries,paired_changes=comparison['paired_changes'],training_status=dict(updates=training['updates'],completed_coverage_epochs=7,configured_cycles=12,stop_reason='early_stopping',stored_partial_training_flag=training['partial_training'],convergence_claimed=False,source_exposure=training['source_exposure']),
        checkpoint_hashes=training['checkpoint_hashes'],metered_app_compute_usd=cost,budget_usd=5,remaining_credits_verified=False,production_changed=False,final_holdout_access=False,artifacts={str(p):digest(p) for p in artifacts},recommendation='Further independent validation; no automatic production promotion; assess CPU and noncommercial-weight restrictions')
    write_new(Path('data/metadata/convnext_c9_results_v1.json'),result)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
