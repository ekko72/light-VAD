# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill050\best.pt`
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
| adaptive | 0.94413 | 9.327% | 0.94430 |

- Adaptive minus Short F1: 0.00051.
- Adaptive minus Long F1: -0.00522.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 14978 | 1894 | 1648 | 1.642% | 0.082% |

- Speaker-cluster 95% CI for net / selected: -1.465% to 5.086%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96188 | 0.00163 | 2587 | 5.412% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96596 | 0.00057 | 1104 | 2.989% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95770 | 0.00110 | 1768 | 4.299% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.91944 | -0.00096 | 2264 | -1.899% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.92817 | -0.00017 | 2388 | 0.000% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.90872 | 0.00010 | 3256 | 0.399% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95329 | 0.00052 | 7363 | 2.119% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.90994 | -0.00042 | 5028 | -0.994% |
- Seen-noise net / selected 95% CI: 0.499% to 3.757%.
- Unseen-noise net / selected 95% CI: -6.399% to 6.305%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
