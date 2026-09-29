# A10 Temporal Value Predictability

## Frozen Protocol

- Predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\eval_rf384_fixed130\frame_predictions.npz`
- Adaptive checkpoint: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_rf384\best.pt`
- Refinement span: 384 frames (3.84 s)
- Temporal feature window: 25 frames (250 ms)
- Calibration/test speakers: 20/20
- Primary calibration activation: 5.00%
- Features: `posterior_entropy`, `posterior_change`, `short_term_variance`, `prediction_switch_rate`
- Short hidden-embedding delta was not present in the frozen prediction bundle, so it was not used.

All router fitting and thresholds use calibration speakers only. Test activation is observed, not forced to the calibration target. The random baseline repeats a calibration-thresholded uniform gate.

## Signed Temporal Value

| Split | Frames | +1 Correction | 0 No effect | -1 Harm | E[v] | Correction/frame | Harm/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | 255,232 | 3,203 | 250,119 | 1,910 | 0.005066 | 1.255% | 0.748% |
| test | 298,300 | 3,981 | 291,924 | 2,395 | 0.005317 | 1.335% | 0.803% |

## Primary Gate Comparison

| Gate | Test activation | Correction/selected | Harm/selected | Net/selected | Net/frame | F1 | Cluster 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uncertainty | 5.050% | 26.427% | 15.899% | 10.528% | 0.532% | 0.94638 | [5.271%, 15.734%] |
| value | 5.009% | 26.645% | 16.030% | 10.615% | 0.532% | 0.94646 | [5.334%, 15.870%] |
| random mean | 5.006% | 1.336% | 0.804% | 0.532% | 0.027% | 0.94379 | [0.313%, 0.748%] |

- Value minus uncertainty net/selected: 0.087% (paired speaker-cluster 95% CI [-0.055%, 0.258%]).

## Activation-Budget Comparison

| Calibration target | Gate | Mean test activation | Net/selected | Net/frame | F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2% | uncertainty | 1.983% | 26.818% | 0.532% | 0.94505 |
| 2% | value | 1.917% | 27.742% | 0.532% | 0.94524 |
| 2% | random mean | 1.999% | 54.852% | 0.011% | 0.94369 |
| 5% | uncertainty | 5.050% | 10.528% | 0.532% | 0.94638 |
| 5% | value | 5.009% | 10.615% | 0.532% | 0.94646 |
| 5% | random mean | 5.006% | 53.208% | 0.027% | 0.94379 |
| 10% | uncertainty | 10.317% | 5.153% | 0.532% | 0.94693 |
| 10% | value | 10.297% | 5.164% | 0.532% | 0.94696 |
| 10% | random mean | 10.007% | 53.474% | 0.054% | 0.94396 |
| 20% | uncertainty | 21.459% | 2.478% | 0.532% | 0.94696 |
| 20% | value | 22.005% | 2.416% | 0.532% | 0.94696 |
| 20% | random mean | 20.009% | 53.136% | 0.106% | 0.94429 |

## Temporal Context Value Map

- Coordinates: posterior entropy (x) and short-term posterior variance (y), each split into calibration quantile bins.
- Highest finite-bin E[v]: 0.074092 with 10,460 calibration frames.
- Full bins: `a10_temporal_value_map.csv`.
- Plot: `a10_temporal_value_map.png`.

## GO Assessment

- Status: **UNCERTAINTY_PROXY**.
- PASS: Value gate has higher test net/selected than uncertainty gate.
- FAIL: Paired speaker-cluster 95% CI for the difference excludes zero.
- Interpretation: The value score does not establish a significant advantage over uncertainty at the primary budget. Keep the simple uncertainty gate as the honest low-cost proxy; this is not an A-wide NO-GO.
