# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed19_gate013_distill025_rf384\best.pt`
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
| adaptive | 0.94643 | 8.969% | 0.94503 |

- Adaptive minus Short F1: 0.00281.
- Adaptive minus Long F1: -0.00292.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 3017 | 1704 | 8.766% | 0.440% |

- Speaker-cluster 95% CI for net / selected: 5.024% to 12.201%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96113 | 0.00087 | 2587 | 2.474% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96643 | 0.00105 | 1104 | 5.525% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95841 | 0.00181 | 1768 | 6.787% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.92785 | 0.00745 | 2264 | 18.463% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.93214 | 0.00380 | 2388 | 10.343% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.91405 | 0.00543 | 3256 | 11.179% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95370 | 0.00093 | 7363 | 3.572% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.91910 | 0.00874 | 5028 | 19.610% |
- Seen-noise net / selected 95% CI: 0.681% to 6.576%.
- Unseen-noise net / selected 95% CI: 11.978% to 25.735%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
