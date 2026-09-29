# Phase A3: Sparse Shared-Encoder Refinement

## Fixed Protocol

- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_smoke_10step\best.pt`
- Reference predictions: `results\difficulty_adaptive_context\smoke_eval\frame_predictions.npz`
- Calibration speakers: 12
- Final-test speakers: 12
- Speaker split seed: 20260917
- Calibration activation target: 10.00%
- Frozen threshold: 0.217507
- Test activation: 15.632%
- Refinement lookback: 64 frames (0.64 s)
- Estimated refinement MACs per selected frame: 3,456

The threshold is chosen only on calibration-speaker Short scores. The final-test activation rate is observed, not forced to the calibration budget.

## Final-Test Model Comparison

| Model | F1 | Error | AUROC |
| --- | ---: | ---: | ---: |
| short | 0.91617 | 15.469% | 0.36481 |
| long | 0.91617 | 15.469% | 0.58617 |
| adaptive | 0.91617 | 15.469% | 0.36240 |

- Adaptive minus Short F1: 0.00000.
- Adaptive minus Long F1: 0.00000.

## Final-Test Gate Utility

| Selected | Correction | Harm | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: | ---: |
| 1823 | 0 | 0 | 0.000% | 0.000% |

- Speaker-cluster 95% CI for net / selected: 0.000% to 0.000%.

## By Condition

| Condition | Frames | Short F1 | Long F1 | Adaptive F1 | Adaptive-Short | Selected | Net / selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 5831 | 0.91617 | 0.91617 | 0.91617 | 0.00000 | 1071 | 0.000% |
| 20 | 586 | 0.89642 | 0.89642 | 0.89642 | 0.00000 | 147 | 0.000% |
| 10 | 768 | 0.95510 | 0.95510 | 0.95510 | 0.00000 | 46 | 0.000% |
| 5 | 98 | 0.74359 | 0.74359 | 0.74359 | 0.00000 | 0 | n/a |
| 0 | 692 | 0.93538 | 0.93538 | 0.93538 | 0.00000 | 0 | n/a |
| -5 | 2563 | 0.89976 | 0.89976 | 0.89976 | 0.00000 | 430 | 0.000% |
| seen | 5831 | 0.91617 | 0.91617 | 0.91617 | 0.00000 | 752 | 0.000% |
- Seen-noise net / selected 95% CI: 0.000% to 0.000%.

## Interpretation Guardrails

- Adaptive does not need to beat Long-only in absolute F1.
- Accuracy is informative only together with activation rate, CPU latency, cache and MAC accounting.
- The gate is fixed before final-test evaluation; test top-k selection is not used.
