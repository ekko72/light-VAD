# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed17\best.pt`
- Reference predictions: `results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz`
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Calibration activation target: 10.00%
- Frozen threshold: 0.234942
- Test activation: 10.317%
- Refinement lookback: 64 frames (0.64 s)
- Estimated refinement MACs per selected frame: 3,456

The threshold is chosen only on calibration-speaker Short scores. The final-test activation rate is observed, not forced to the calibration budget.

## Final-Test Model Comparison

| Model | F1 | Error | AUROC |
| --- | ---: | ---: | ---: |
| short | 0.94363 | 9.409% | 0.94424 |
| long | 0.94936 | 8.481% | 0.94950 |
| adaptive | 0.94308 | 9.488% | 0.94397 |

- Adaptive minus Short F1: -0.00055.
- Adaptive minus Long F1: -0.00628.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 30776 | 1690 | 1925 | -0.764% | -0.079% |

- Speaker-cluster 95% CI for net / selected: -2.715% to 1.204%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96113 | 0.00088 | 5498 | 1.401% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96552 | 0.00013 | 2294 | 0.305% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95727 | 0.00067 | 3598 | 1.390% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.91690 | -0.00351 | 4670 | -3.897% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.92751 | -0.00083 | 4888 | -0.757% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.90664 | -0.00198 | 6481 | -1.929% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95300 | 0.00023 | 15063 | 0.558% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.90689 | -0.00347 | 10215 | -3.877% |
- Seen-noise net / selected 95% CI: -0.470% to 1.493%.
- Unseen-noise net / selected 95% CI: -7.268% to 0.780%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
