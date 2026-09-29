# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\train_rf256\best.pt`
- Reference predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz`
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Threshold source: fixed before final-test evaluation
- Frozen threshold: 0.130000
- Test activation: 5.021%
- Refinement lookback: 256 frames (2.56 s)
- Estimated refinement MACs per selected frame: 3,456
- Estimated refinement MACs per call (64 selected frames): 221,184
- Streaming cache bytes: 165,376

The threshold was fixed before final-test evaluation. The final-test activation rate is observed and is not used to choose the threshold.

## Final-Test Model Comparison

| Model | F1 | Error | AUROC |
| --- | ---: | ---: | ---: |
| short | 0.94363 | 9.409% | 0.94424 |
| long | 0.94936 | 8.481% | 0.94950 |
| refine-only | 0.94700 | 8.879% | 0.94881 |
| adaptive | 0.94635 | 8.976% | 0.94488 |

- Adaptive minus Short F1: 0.00273.
- Adaptive minus Long F1: -0.00300.
- Refine-only minus Short F1: 0.00337.

## CPU Refinement Microbenchmark

- Protocol: batch 1, 257 history frames, 64 selected frames per call, one CPU thread.
- Median: 14.37150 ms/call (224.555 us/selected frame).
- P95: 16.26591 ms/call.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 3070 | 1777 | 8.633% | 0.433% |

- Speaker-cluster 95% CI for net / selected: 5.199% to 12.256%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Refine F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96246 | 0.96218 | 0.00192 | 2587 | 6.262% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96612 | 0.96599 | 0.00060 | 1104 | 2.989% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95878 | 0.95875 | 0.00215 | 1768 | 8.314% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.92970 | 0.92723 | 0.00682 | 2264 | 17.182% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.93326 | 0.93215 | 0.00381 | 2388 | 10.553% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.91349 | 0.91242 | 0.00380 | 3256 | 7.985% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95382 | 0.95389 | 0.00111 | 7363 | 4.468% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.92014 | 0.91752 | 0.00716 | 5028 | 15.951% |
- Seen-noise net / selected 95% CI: 1.436% to 7.523%.
- Unseen-noise net / selected 95% CI: 7.529% to 23.610%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
