# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_rf384\best.pt`
- Reference predictions: `results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz`
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Threshold source: fixed before final-test evaluation
- Frozen threshold: 0.130000
- Test activation: 5.021%
- Refinement lookback: 384 frames (3.84 s)
- Estimated refinement MACs per selected frame: 3,456

The threshold was fixed before final-test evaluation. The final-test activation rate is observed and is not used to choose the threshold.

## Final-Test Model Comparison

| Model | F1 | Error | AUROC |
| --- | ---: | ---: | ---: |
| short | 0.94363 | 9.409% | 0.94424 |
| long | 0.94936 | 8.481% | 0.94950 |
| adaptive | 0.94637 | 8.964% | 0.94503 |

- Adaptive minus Short F1: 0.00274.
- Adaptive minus Long F1: -0.00299.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 3094 | 1767 | 8.860% | 0.445% |

- Speaker-cluster 95% CI for net / selected: 5.804% to 11.918%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96150 | 0.00124 | 2587 | 3.904% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96605 | 0.00066 | 1104 | 3.533% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95838 | 0.00178 | 1768 | 7.070% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.92828 | 0.00788 | 2264 | 20.097% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.93147 | 0.00313 | 2388 | 9.087% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.91330 | 0.00468 | 3256 | 9.828% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95361 | 0.00083 | 7363 | 3.653% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.91877 | 0.00841 | 5028 | 19.033% |
- Seen-noise net / selected 95% CI: 0.670% to 6.680%.
- Unseen-noise net / selected 95% CI: 11.029% to 25.401%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
