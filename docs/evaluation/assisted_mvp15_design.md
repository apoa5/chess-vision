# MVP-1.5 registered development design

No training, reduced precision, new weights, C.10 tuning or final holdout access.
Preserve all C.9/C.11 and MVP-1 historical reports/configurations. The assisted
single-photo path and original guided default remain available.

Register five A.3 sessions' three D.1 quality-diverse views among D.2's 55
human-reviewed frames (15 frames), plus three C.11 small-margin C.9 validation
examples. No C.9 retrospective internal-test input. The prepared-input corpus
retains original caches, semantic square IDs and reviewed geometry. Labels use
C.9's separately corrected owner-reviewed placements, including middlegame c7;
no label selects a view or prediction. All five pilot positions were exposed in
C.9 development/training: results are not independent generalization evidence.

Benchmark one loaded C.11 model: sequential optimized batch8/T4; chunked shared
batch8/16/32/T4; two shared-model workers batch8/T2 each. No three model copies,
no full 192-input allocation, no GPU work. Warm up each three-view workload once
and repeat three times in separate processes per strategy. Measure wall time,
CPU utilization, process peak RSS, every timed repetition's predictions, plus
nine registered parity frames including the minimum C.11 margin (~0.0143).
Freeze numeric tolerance at C.11's atol=2.6702880859375e-5, rtol=1e-5; require zero
changed classes and complete placements. Select fastest parity-passing median;
if no alternative qualifies retain the unchanged C.11 sequential reference.

Evaluate D.3's best whole quality view, mean probabilities, mean logits on the
same three views/session with ConvNeXt. Promote aggregation only with strictly
fewer errors, no extra empty false positives per session, no worse occupied
accuracy per session, no lost exact boards and no newly confident (>90%) wrong
occupied square. Otherwise default best quality. Report corrected/introduced
errors and per-class results. No probability multiplication or independence
assumption for correlated frames. Do not tune on C.10.

Live application: bounded sampled JPEG candidates with inexpensive D.1 quality
and unchanged automatic geometry at capture; retain compressed frames and exact
geometry binding temporarily. Manual Finish selects up to three complementary
canonical views, then prepares inputs and runs the selected measured backend.
No usable view is an explicit error; one uses MVP-1's unchanged photo path. No
automatic completion. Same editor/confirmation and FEN validation are mandatory.

CPU benchmark jobs run serially without simultaneous heavy regression/browser
work. Optional Intel GPU is read-only feasibility: if OpenVINO/drivers/devices
are unavailable, defer rather than installing drivers or blocking the CPU path.
