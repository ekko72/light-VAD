# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\train_rf128\best.pt`
- Reference predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz`
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Threshold source: fixed before final-test evaluation
- Frozen threshold: 0.130000
- Test activation: 5.021%
- Refinement lookback: 128 frames (1.28 s)
- Estimated refinement MACs per selected frame: 3,456
- Estimated refinement MACs per call (64 selected frames): 221,184
- Streaming cache bytes: 99,840

The threshold was fixed before final-test evaluation. The final-test activation rate is observed and is not used to choose the threshold.

## Final-Test Model Comparison

| Model | F1 | Error | AUROC |
| --- | ---: | ---: | ---: |
| short | 0.94363 | 9.409% | 0.94424 |
| long | 0.94936 | 8.481% | 0.94950 |
| refine-only | 0.94685 | 8.936% | 0.94718 |
| adaptive | 0.94636 | 8.998% | 0.94472 |

- Adaptive minus Short F1: 0.00274.
- Adaptive minus Long F1: -0.00299.
- Refine-only minus Short F1: 0.00323.

## CPU Refinement Microbenchmark

- Protocol: batch 1, 129 history frames, 64 selected frames per call, one CPU thread.
- Median: 7.33825 ms/call (114.660 us/selected frame).
- P95: 9.61322 ms/call.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 3000 | 1772 | 8.199% | 0.412% |

- Speaker-cluster 95% CI for net / selected: 4.745% to 11.758%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Refine F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96340 | 0.96353 | 0.00328 | 2587 | 10.862% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96623 | 0.96597 | 0.00058 | 1104 | 2.536% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95805 | 0.95799 | 0.00139 | 1768 | 4.921% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.92945 | 0.92686 | 0.00646 | 2264 | 15.636% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.93235 | 0.93144 | 0.00310 | 2388 | 8.082% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.91306 | 0.91237 | 0.00375 | 3256 | 7.371% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95367 | 0.95361 | 0.00084 | 7363 | 2.866% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.91910 | 0.91710 | 0.00673 | 5028 | 14.638% |
- Seen-noise net / selected 95% CI: 0.204% to 5.136%.
- Unseen-noise net / selected 95% CI: 6.423% to 22.046%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
