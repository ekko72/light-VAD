# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed18_gate013_distill025_rf384\best.pt`
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
| adaptive | 0.94631 | 8.983% | 0.94494 |

- Adaptive minus Short F1: 0.00268.
- Adaptive minus Long F1: -0.00305.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 2973 | 1702 | 8.486% | 0.426% |

- Speaker-cluster 95% CI for net / selected: 4.749% to 11.974%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96137 | 0.00111 | 2587 | 3.363% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96586 | 0.00047 | 1104 | 2.355% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95850 | 0.00190 | 1768 | 7.410% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.92706 | 0.00666 | 2264 | 16.696% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.93210 | 0.00376 | 2388 | 10.469% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.91392 | 0.00529 | 3256 | 10.964% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95358 | 0.00080 | 7363 | 3.300% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.91868 | 0.00832 | 5028 | 18.715% |
- Seen-noise net / selected 95% CI: 1.097% to 5.738%.
- Unseen-noise net / selected 95% CI: 10.117% to 25.559%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
