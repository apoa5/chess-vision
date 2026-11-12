"""C10 saved-prediction comparison, error crops and report. No model inference."""
import csv,html,json,os,sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import digest,placement_labels,SQUARES
from src.training.manifest_tools import write_new
from src.vision.independent_c10 import CHECKPOINTS
ROOT=Path('outputs/evaluation/convnext_c10_v1')
MANIFEST=Path('data/metadata/convnext_independent_c10_manifest_v1.json')
GLYPHS=['·','♙','♘','♗','♖','♕','♔','♟','♞','♝','♜','♛','♚']

def pct(v):return '—' if v is None else f'{v*100:.2f}%'
def digital(values,truth):
    return '<div class="board">'+''.join(f'<span class="{("dark " if (i//8+i%8)%2 else "")+("wrong" if p!=truth[i] else "")}" title="{SQUARES[i]}: {CLASS_NAMES[p]}">{GLYPHS[p]}</span>' for i,p in enumerate(values))+'</div>'

def main():
    m=json.loads(MANIFEST.read_text());records={n:json.loads((ROOT/(n+'_result.json')).read_text()) for n in CHECKPOINTS};out=ROOT/'report';out.mkdir(exist_ok=False)
    for record in records.values():
        if record['manifest_sha256']!=digest(MANIFEST):raise ValueError('Compared manifest changed')
    photos=[];all_errors=[]
    for photo in m['photos']:
        rows={n:next(r for r in rec['rows'] if r['position_id']==photo['id']) for n,rec in records.items()};errors=[]
        a,b=[rows[n] for n in ('b2','convnext')]
        if a['completed'] and b['completed']:
            if a['packed_inputs_sha256']!=b['packed_inputs_sha256']:raise ValueError('Models received different crops/masks')
            if [(o['square'],o['truth']) for o in a['observations']]!=[(o['square'],o['truth']) for o in b['observations']]:raise ValueError('Prediction coordinates or labels differ')
            for oa,ob in zip(a['observations'],b['observations']):
                if not oa['correct'] or not ob['correct']:
                    errors.append(dict(position_id=photo['id'],square=oa['square'],actual_class=CLASS_NAMES[oa['truth']],
                        b2_prediction=CLASS_NAMES[oa['prediction']],convnext_prediction=CLASS_NAMES[ob['prediction']],
                        b2_confidence=oa['confidence'],convnext_confidence=ob['confidence'],
                        corrected_by_convnext=not oa['correct'] and ob['correct'],regressed_by_convnext=oa['correct'] and not ob['correct'],both_wrong=not oa['correct'] and not ob['correct']))
        photos.append(dict(position_id=photo['id'],rows=rows,errors=errors));all_errors.extend(errors)
    comparison=dict(schema_version='convnext_independent_c10_comparison_v1',manifest_sha256=digest(MANIFEST),independent_position_count=sum(p['independence']['independent_eligible'] for p in m['photos']),
        photos=photos,models={n:{k:v for k,v in r.items() if k!='rows'} for n,r in records.items()},
        paired_changes=dict(corrected_by_convnext=sum(e['corrected_by_convnext'] for e in all_errors),regressed_by_convnext=sum(e['regressed_by_convnext'] for e in all_errors),both_wrong=sum(e['both_wrong'] for e in all_errors)),
        secondary_verified_geometry=dict(available=False,reason='No independent human-verified corner annotations or owner geometry review supplied; assistant inspection is not a human attestation'),
        original_artifacts_sha256={n:digest(ROOT/(n+'_result.json')) for n in records},training_performed=False,production_changed=False,cloud_cost_usd=0)
    write_new(ROOT/'comparison.json',comparison)
    with (ROOT/'errors.csv').open('x',newline='') as f:
        fields=['position_id','square','actual_class','b2_prediction','convnext_prediction','b2_confidence','convnext_confidence','corrected_by_convnext','regressed_by_convnext','both_wrong']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_errors)
    with Path('docs/evaluation/convnext_independent_c10_classes.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','class_name','class_id','support','predicted_count','precision','recall','f1']);w.writeheader()
        for n,r in records.items():
            if r['aggregate']['square_metrics']:
                for row in r['aggregate']['square_metrics']['per_class']:w.writerow(dict(model=n,class_name=CLASS_NAMES[row['class_id']],**row))
    doc=['<!doctype html><meta charset="utf-8"><title>C10 independent photographs</title><style>body{font:16px system-ui;margin:24px;max-width:1550px}.images,.boards{display:flex;gap:18px;flex-wrap:wrap}.images img{max-width:390px;max-height:500px;object-fit:contain}.board{display:grid;grid-template-columns:repeat(8,30px)}.board span{height:30px;text-align:center;font-size:26px;background:#f0dfc3}.board .dark{background:#b48a61}.board .wrong{outline:2px solid red;outline-offset:-2px}section{padding:22px 0;border-top:1px solid #aaa}td,th{padding:6px;border:1px solid #aaa}table{border-collapse:collapse}.crop img{width:224px}.note{background:#fff0c2;padding:12px}</style><h1>C.10 — Three fresh positions</h1><p>Frozen weights and B.2 context1_224 inputs, CPU only. Ground truth comes from the owner’s numbered FENs, not predictions. Three positions are not192 independent samples. Red board cells mark errors.</p><p class="note">Automatic completion does not certify geometry. Printed-coordinate orientation and gross alignment were inspected by the assistant; fine corners have not received a separate owner review. No verified-corner secondary result is invented.</p>']
    for photo,summary in zip(m['photos'],photos):
        rows=summary['rows'];g=photo['automatic_geometry'];truth=placement_labels(photo['fen']);pid=photo['id']
        doc.append(f'<section><h2>{pid}</h2><p>Source: {html.escape(photo["path"])}; SHA-256 {photo["sha256"]}</p><p>Owner reference FEN: <code>{photo["fen"]}</code></p><p>Geometry completed: {g["completed"]}; orientation: {g.get("predicted_camera_orientation")}; failure stage: {g.get("failure_stage")}; independent eligible: {photo["independence"]["independent_eligible"]}.</p><div class="images">')
        for label,path in [('Original photograph',photo['path']),('Automatic corners/grid',g['overlay_path']),('Rectified board context',str(ROOT/pid/'automatic/context.png'))]:
            if Path(path).exists():doc.append(f'<figure><a href="{os.path.relpath(path,out)}"><img src="{os.path.relpath(path,out)}"></a><figcaption>{label}</figcaption></figure>')
        doc.append('</div><div class="boards"><div>Owner ground truth'+digital(truth,truth)+'</div>')
        for n,row in rows.items():
            if row['completed']:doc.append(f'<div>{n}: {row["metrics"]["corrections"]} corrections'+digital([o['prediction'] for o in row['observations']],truth)+'</div>')
            else:doc.append(f'<div>{n}: failed at {row["failure_stage"]}; no silently substituted geometry</div>')
        doc.append('</div><h3>Errors and disagreements</h3><table><tr><th>Square</th><th>Actual</th><th>B.2 / probability</th><th>ConvNeXt / probability</th><th>Paired outcome</th></tr>')
        packed=np.load(photo['inputs']['path'])['packed'] if photo.get('inputs') else None
        for e in summary['errors']:
            outcome='ConvNeXt correction' if e['corrected_by_convnext'] else 'ConvNeXt regression' if e['regressed_by_convnext'] else 'both wrong'
            doc.append(f'<tr><td>{e["square"]}</td><td>{e["actual_class"]}</td><td>{e["b2_prediction"]} / {e["b2_confidence"]:.4f}</td><td>{e["convnext_prediction"]} / {e["convnext_confidence"]:.4f}</td><td>{outcome}</td></tr>')
        doc.append('</table><div class="images">')
        for e in summary['errors']:
            p=packed[list(SQUARES).index(e['square'])];rgb=p[:3].transpose(1,2,0).copy();mask=p[3].astype(np.uint8);contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,(255,0,0),1)
            name=pid+'_'+e['square']+'.jpg';cv2.imwrite(str(out/name),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,92]);doc.append(f'<figure class="crop"><img src="{name}"><figcaption>{e["square"]}: {e["actual_class"]}<br>B.2 {e["b2_prediction"]}; ConvNeXt {e["convnext_prediction"]}<br>Red: target footprint, not piece silhouette.</figcaption></figure>')
        doc.append('</div></section>')
    (out/'index.html').write_text('\n'.join(doc))
    review=dict(schema_version='c10_optional_owner_geometry_review_v1',reviewer=None,photos=[dict(id=p['id'],photo_sha256=p['sha256'],fen_agreement=None,a1_and_semantic_orientation_correct=None,true_corners_and_grid_sufficient=None,all_target_masks_aligned=None,reported_corners=p['automatic_geometry'].get('returned_polygon'),status='pending') for p in m['photos']],not_automatically_applied=True)
    write_new(ROOT/'owner_review_template.json',review)
    write_markdown(m,records,comparison)
    print(json.dumps({n:r['aggregate'] for n,r in records.items()},indent=2))

def write_markdown(manifest,records,comparison):
    lines=['# C.10 — Independent evaluation on three fresh photographs','',
        'Evaluation-only, 2026-11-10. Two frozen checkpoints; no training, model/crop/geometry/threshold tuning, cloud jobs, production replacement, final-holdout access, commits or pushes. The C.10 manifest and configuration were frozen before classifier inference.','',
        '## References and independence','',
        'The owner supplied each full FEN and its numbered photo association in the C.10 request. All FENs are structurally/chess-valid (32,30,15 pieces respectively). Their piece placements and visible printed a1–h8 orientation were visually cross-checked against the photographs by the assistant; no classifier established a reference label. This is owner-provided human reference provenance, **not a separately recorded owner attestation of every crop or corner**.','',
        'All three images are1920×2560, with no EXIF orientation tag. Precise capture time, camera, room, elevation, lighting type and session identity are unavailable. Indoor patterned bedding and piece shadows are observable; nighttime/artificial lighting is not inferred as fact. All are White-near by printed coordinates, with a1 at the bottom-left playing corner.','',
        'No new canonical piece placement or source hash matches C.9. The audit checks all363 C.9 board records, all180 original-photo inventory entries (including the quarantined photograph), and exact placement keys ignoring turn/move counters. Raw-photo64-bit dHash nearest distances are21,17,16; those proxies do not prove independence. Manual source/FEN inspection and the canonical-placement/source-hash audit support three new positions. The same physical set and similar capture conditions limit generalization; these are three position units, not192 independent trials.','',
        '| Position | File | SHA-256 |','|---|---|---|']
    for p in manifest['photos']:lines.append(f'| {p["id"]} | `{p["filename"]}` | `{p["sha256"]}` |')
    lines+=['','Full owner FENs, EXIF/source provenance, nearest-image candidates, audits, cached-input hashes and geometry are in `data/metadata/convnext_independent_c10_manifest_v1.json`. These photos remain evaluation-only and were not added to C.9.','',
        '## Frozen protocol','',
        '| Model | Checkpoint | SHA-256 |','|---|---|---|']
    for n,spec in CHECKPOINTS.items():lines.append(f'| {n} | `{spec["path"]}` | `{spec["sha256"]}` |')
    lines+=['','ConvNeXt selection remains C.9 update1,578; its plan hash/class order/input contract is verified. B.2 is the tighter-crop reference, not C.5/C.7 or the production scanner profile. Both use RGB+binary target mask, the same13 classes, B.2 `context1_224`, board800/context margin200, initial context padding200, AREA RGB and NEAREST mask resizing, original ImageNet normalization, no augmentation/calibration, canonical output a8–h1, camera-relative input pixels, CPU FP32, batch16, two CPU threads/interop1. Weights are strict-loaded, eval mode, gradients disabled, and `torch.inference_mode()` is used.','',
        'Primary policy is unchanged **legacy/default `prepare_raw_photo`** with the frozen `config/settings.yaml`. B.1b recovery is not silently enabled. FENs and expected orientation never enter automatic localization, marking-based orientation or crop extraction. `scripts/run_square_evidence.py:make_inputs` supplies the existing B.2 packing path to both models. Their192 individual RGB/mask hashes match exactly. No rotation is selected for better recognition.','',
        'The manifest, configuration and execution-source hashes are bound in `outputs/evaluation/convnext_c10_v1/execution_contract.json`; models reject changed execution inputs. Identical-input repeat inferences are timing-only and must reproduce all logits exactly; their predictions do not replace the first pass.','',
        '## Geometry and orientation','',
        'Automatic pipeline completion: **3/3** for both models, with marking-based White-near orientation3/3, consistent with visible coordinates. The assistant inspected all three full-board overlays and identified no grossly incorrect completed polygon or semantic-grid mapping. **Fine geometry remains owner-unverified.** Completion is not a localization-correctness certificate. Detected corners, source polygons, orientation evidence, rectified context and candidate diagnostics remain under each position’s `automatic/` directory.','',
        '**Verified-geometry secondary evaluation: unavailable (0/3)**. No independent human corner annotations or explicit owner geometry review were supplied. No manual corners were invented or silently substituted. The report includes `owner_review_template.json` if the owner later wants to confirm the registered geometry in a versioned supplement. This limits causal attribution of residual errors to classification alone.','',
        '## Recognition and correction effort','',
        'Exact boards mean all64 piece-placement labels match the owner reference. FEN syntax or legality alone never counts as exact. The full automatic-pipeline denominator remains three; failures would have no fabricated square predictions or correction count.','',
        '| Position/model | Overall | Occupied | Empty accuracy | Empty FP | Missing pieces | Wrong occupied identities | Corrections | Exact |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in manifest['photos']:
        for n,record in records.items():
            r=next(v for v in record['rows'] if v['position_id']==p['id'])
            if not r['completed']:lines.append(f'| {p["id"]}/{n} | failed: {r["failure_stage"]} | — | — | — | — | — | — | no |');continue
            v=r['metrics'];lines.append(f'| {p["id"]}/{n} | {pct(v["overall_accuracy"])} | {pct(v["occupied_accuracy"])} | {pct(v["empty_accuracy"])} | {v["empty_false_positives"]} | {v["occupied_to_empty_misses"]} | {v["incorrect_piece_identities"]} | {v["corrections"]} | {v["exact_board"]} |')
    lines+=['','| Aggregate/model | Overall | Occupied | Empty accuracy | Empty FP/rate | Exact / all attempts | Mean corrections |','|---|---:|---:|---:|---:|---:|---:|']
    for n,r in records.items():
        a=r['aggregate'];v=a['square_metrics'];lines.append(f'| {n} | {pct(v["overall_accuracy"])} | {pct(v["occupied_accuracy"])} | {pct(1-v["empty_false_positive_rate"])} | {v["empty_false_positives"]}/{v["empty_count"]} ({pct(v["empty_false_positive_rate"])}) | {a["exact_boards"]}/3 | {a["mean_corrections_completed"]:.2f} |')
    changes=comparison['paired_changes'];lines+=['',f'Paired outcomes: ConvNeXt corrects **{changes["corrected_by_convnext"]}** B.2 errors, introduces **{changes["regressed_by_convnext"]}** errors on previously correct squares, and both remain wrong on **{changes["both_wrong"]}** squares. Each position requires one ConvNeXt correction in this run; neither model reconstructs any of the three boards exactly.','',
        'Color/type error counts can overlap and must not be added to obtain total corrections. Missing pieces are occupied→empty; extra pieces are empty→occupied. All counts,13×13 confusion matrices, per-class precision/recall/F1 and source observations are machine-readable. [Per-class CSV](convnext_independent_c10_classes.csv). Full per-square differences/confidences are in `outputs/evaluation/convnext_c10_v1/errors.csv`.','',
        '## Confidence','',
        '| Model | Mean raw probability, correct | Mean raw probability, wrong | Errors above90% | Empty FP above90% |','|---|---:|---:|---:|---:|']
    for n,r in records.items():
        a=r['aggregate'];lines.append(f'| {n} | {pct(a["mean_confidence_correct"])} | {pct(a["mean_confidence_incorrect"])} | {a["confident_errors_above_0_9"]} | {a["confident_empty_false_positives"]} |')
    lines+=['','Confidence is uncalibrated; >.9 is a diagnostic only. No abstention or user-confirmation threshold was fitted. Every predicted board still requires human review. Raw13-class logits and full softmax distributions remain in each model’s result artifact.','',
        '## Runtime and memory','',
        'The timed wrapper transparently calls the unchanged image loader, detector, rectifier, orientation resolver and grid fitter. Replayed corners/orientation and every packed crop/mask must match registration. The shared D.2 packing helper performs an additional reference-context warp; that cost is included in crop extraction, not hidden as classifier speed. Source scoring/FEN legality diagnostics are included in observed end-to-end timing.','',
        '| Position/model | Load image | Detection | First rectification | Orientation + grid | Packing + normalization | First forward | Warm forward median | FEN | End-to-end after model load |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,record in records.items():
        for r in record['rows']:
            if not r['completed']:continue
            t=r['timings'];pid=r['position_id'];lines.append(f'| {pid}/{n} | {t["load_image"]:.3f}s | {t["detect_board_corners"]:.3f}s | {t["rectify_with_config"]:.3f}s | {t["resolve_orientation"]+t["fit_grid"]:.3f}s | {t["crop_extraction_including_reference_context_warp"]+t["normalization"]:.3f}s | {t["model_forward"]:.3f}s | {record["warm_median_forward_seconds"][pid]:.3f}s | {t["fen_serialization_and_validation"]:.5f}s | {r["end_to_end_after_model_load_seconds"]:.3f}s |')
    for n,r in records.items():lines+=['',f'{n}: model load/strict validation **{r["model_loading_seconds"]:.3f}s**; first model+photo path **{r["cold_first_image_plus_model_seconds"]:.3f}s**; process peak RSS **{r["peak_process_rss_bytes"]/1e6:.0f}MB**.']
    lines+=['',f'Hardware: {records["convnext"]["cpu_model"]}; PyTorch{records["convnext"]["torch_version"]}, OpenCV{records["convnext"]["opencv_version"]}. Models ran in separate processes. Two warmup board passes and five forward-only measurements per image were made; image order rotates between rounds. “Cold” means the first call after Python/library imports; filesystem/OS caches were not flushed. Warm forward timings exclude geometry, loading and extraction. No optimization was applied. C.9’s approximately8s/board measurement uses the same CPU-thread/batch convention; actual C.10 per-image measurements above provide the comparison. Cloud usage: **$0**.','',
        '## Visual report and reproducibility','',
        '[Originals, automatic grids, rectified contexts, reference/predicted boards, complete error lists and actual224px RGB/mask error crops](../../outputs/evaluation/convnext_c10_v1/report/index.html). Predictions from both models use the same source pixels and geometry. Red outlines identify target footprints, not the piece silhouette. Distortion, truncation, shadow and neighboring interference remain hypotheses unless directly supported by the pictured crop; no coordinate-mapping defect is asserted from a misclassification alone.','',
        '```bash',
        '# Registration is already frozen; existing outputs refuse overwrites.',
        '.venv/bin/python scripts/evaluate_convnext_c10.py register',
        '# Each model runs in its own CPU process; no remote resources:',
        'outputs/evaluation/chessqueries_c2_env/bin/python scripts/evaluate_convnext_c10.py infer --model b2',
        'outputs/evaluation/chessqueries_c2_env/bin/python scripts/evaluate_convnext_c10.py infer --model convnext',
        'outputs/evaluation/chessqueries_c2_env/bin/python scripts/report_convnext_c10.py',
        '.venv/bin/python scripts/evaluate_convnext_c10.py check',
        '```','',
        'The optional C.2 environment already contains the pinned timm1.0.22 implementation and CPU PyTorch; application dependencies were not upgraded. Initial geometry diagnostics were produced with the existing `scripts.evaluate_pilot_geometry.evaluate_frame(path, load_config(), destination, "legacy")`; repeat runs time the original pipeline without changing behavior. To reproduce without replacing this evaluation, use a separately versioned output/configuration copy and retain the frozen weights/settings; do not retune on these outcomes.','',
        '## Decision and limitations','',
        '**B — partial generalization.** ConvNeXt substantially reduces correction effort versus B.2 on these new placements, while still producing one error per board and0/3 exact reconstructions. It is promising for a reviewed draft, not an unattended final position. Three same-side photographs of one chess set cannot establish broad robustness across camera sides, lighting, boards or devices. Architecture/training/data differences from B.2 prevent causal architecture-only conclusions. No conclusion of safe confidence calibration follows from three positions.','',
        'Proceed toward the assisted MVP with mandatory confirmation/correction and an explicit, separately authorized recognition profile. A bounded inference-optimization/parity milestone is appropriate given CPU cost; no immediate retraining is justified by this tiny sample. Preserve B.2 fallback and address ConvNeXt’s noncommercial pretrained-weight restriction before any commercial deployment. No integration, optimization, data repurposing or further experiment is started here.','',
        '## Files, validation and acceptance','',
        'New code: `src/vision/independent_c10.py`, `scripts/evaluate_convnext_c10.py`, `scripts/report_convnext_c10.py`, `tests/test_independent_c10.py`. Versioned C.10 config/manifest/results metadata, this report and class CSV are new. All predictions/caches/visualizations live under ignored `outputs/evaluation/convnext_c10_v1/`. No existing production or historical source was edited.','',
        '- [x] Three original photographs registered with hashes, reference FENs, explicit orientation provenance and independence audit.',
        '- [x] Fixed checkpoints, settings, source contract and identical192 square inputs frozen before inference.',
        '- [x] All three default-geometry attempts retained; no truth geometry/orientation substitution or tuning.',
        '- [x] Per-position/aggregate precision, recall, F1, exact boards, corrections, confidence, full logits/probabilities and diagnostics preserved.',
        '- [x] Cold/first-call and repeated warm CPU costs measured separately; no cloud resources.',
        '- [x] No fabricated verified-corner secondary result; optional owner review packet provided.',
        '- [ ] Final focused/regression test counts and protected-file verification recorded in release metadata.',
        '- [x] Stop after C.10; no retraining, optimization, production replacement, final holdout, commit or push.',
    ]
    target=Path('docs/evaluation/convnext_independent_c10.md')
    if target.exists():raise ValueError('Report already exists')
    target.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
