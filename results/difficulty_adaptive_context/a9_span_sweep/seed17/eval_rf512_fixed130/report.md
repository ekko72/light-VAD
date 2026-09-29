# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\train_rf512\best.pt`
- Reference predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz`
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Threshold source: fixed before final-test evaluation
- Frozen threshold: 0.130000
- Test activation: 5.021%
- Refinement lookback: 512 frames (5.12 s)
- Estimated refinement MACs per selected frame: 3,456
- Estimated refinement MACs per call (64 selected frames): 221,184
- Streaming cache bytes: 296,448

The threshold was fixed before final-test evaluation. The final-test activation rate is observed and is not used to choose the threshold.

## Final-Test Model Comparison

| Model | F1 | Error | AUROC |
| --- | ---: | ---: | ---: |
| short | 0.94363 | 9.409% | 0.94424 |
| long | 0.94936 | 8.481% | 0.94950 |
| refine-only | 0.94630 | 8.990% | 0.94844 |
| adaptive | 0.94588 | 9.054% | 0.94479 |

- Adaptive minus Short F1: 0.00225.
- Adaptive minus Long F1: -0.00348.
- Refine-only minus Short F1: 0.00267.

## CPU Refinement Microbenchmark

- Protocol: batch 1, 513 history frames, 64 selected frames per call, one CPU thread.
- Median: 33.20225 ms/call (518.785 us/selected frame).
- P95: 39.41459 ms/call.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 2868 | 1807 | 7.084% | 0.356% |

- Speaker-cluster 95% CI for net / selected: 3.349% to 10.837%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Refine F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96295 | 0.96282 | 0.00256 | 2587 | 8.504% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96602 | 0.96597 | 0.00059 | 1104 | 2.808% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95833 | 0.95841 | 0.00181 | 1768 | 7.014% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.92567 | 0.92473 | 0.00432 | 2264 | 11.042% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.93166 | 0.93088 | 0.00255 | 2388 | 7.245% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.91292 | 0.91161 | 0.00299 | 3256 | 6.235% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95398 | 0.95394 | 0.00117 | 7363 | 4.577% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.91646 | 0.91492 | 0.00456 | 5028 | 10.024% |
- Seen-noise net / selected 95% CI: 2.137% to 7.082%.
- Unseen-noise net / selected 95% CI: 0.902% to 18.658%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
