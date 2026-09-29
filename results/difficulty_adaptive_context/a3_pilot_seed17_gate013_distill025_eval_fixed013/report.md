# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025\best.pt`
- Reference predictions: `results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz`
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Threshold source: fixed before final-test evaluation
- Frozen threshold: 0.130000
- Test activation: 5.021%
- Refinement lookback: 64 frames (0.64 s)
- Estimated refinement MACs per selected frame: 3,456

The threshold was fixed before final-test evaluation. The final-test activation rate is observed and is not used to choose the threshold.

## Final-Test Model Comparison

| Model | F1 | Error | AUROC |
| --- | ---: | ---: | ---: |
| short | 0.94363 | 9.409% | 0.94424 |
| long | 0.94936 | 8.481% | 0.94950 |
| adaptive | 0.94447 | 9.284% | 0.94428 |

- Adaptive minus Short F1: 0.00085.
- Adaptive minus Long F1: -0.00488.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 2131 | 1758 | 2.490% | 0.125% |

- Speaker-cluster 95% CI for net / selected: -0.785% to 6.029%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96253 | 0.00227 | 2587 | 7.460% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96600 | 0.00061 | 1104 | 2.899% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95786 | 0.00126 | 1768 | 4.638% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.92000 | -0.00040 | 2264 | -0.751% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.92808 | -0.00026 | 2388 | -0.419% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.90922 | 0.00060 | 3256 | 1.413% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95342 | 0.00065 | 7363 | 2.404% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.91054 | 0.00017 | 5028 | 0.060% |
- Seen-noise net / selected 95% CI: 0.632% to 4.293%.
- Unseen-noise net / selected 95% CI: -5.173% to 7.039%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
