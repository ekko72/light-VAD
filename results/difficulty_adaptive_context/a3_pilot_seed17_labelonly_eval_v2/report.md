# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed17_labelonly\best.pt`
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
| adaptive | 0.94359 | 9.402% | 0.94424 |

- Adaptive minus Short F1: -0.00004.
- Adaptive minus Long F1: -0.00577.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 30776 | 2039 | 2018 | 0.068% | 0.007% |

- Speaker-cluster 95% CI for net / selected: -1.750% to 1.927%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 57735 | 0.96026 | 0.96639 | 0.96123 | 0.00097 | 5498 | 1.455% |
| 20 | 37782 | 0.96539 | 0.96587 | 0.96554 | 0.00015 | 2294 | 0.349% |
| 10 | 39418 | 0.95660 | 0.95649 | 0.95785 | 0.00125 | 3598 | 2.501% |
| 5 | 35496 | 0.92040 | 0.93784 | 0.91847 | -0.00194 | 4670 | -1.927% |
| 0 | 38534 | 0.92834 | 0.93386 | 0.92749 | -0.00085 | 4888 | -0.614% |
| -5 | 44711 | 0.90862 | 0.91988 | 0.90731 | -0.00132 | 6481 | -1.065% |
| seen | 164375 | 0.95277 | 0.95237 | 0.95309 | 0.00032 | 15063 | 0.836% |
| unseen | 76190 | 0.91036 | 0.92981 | 0.90868 | -0.00168 | 10215 | -1.811% |
- Seen-noise net / selected 95% CI: -0.498% to 2.175%.
- Unseen-noise net / selected 95% CI: -4.761% to 2.135%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
