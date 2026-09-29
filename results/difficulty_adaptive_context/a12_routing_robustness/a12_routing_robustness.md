# A12 Routing Robustness

## Protocol

- Frozen bundle: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\eval_rf384_fixed130\frame_predictions.npz`
- Frozen gate threshold: `0.130000`
- Refined score source: `full_adaptive_scores`
- SSR levels: 10%, 30%, 50%, 70%, 90%
- Acoustic conditions: 20, 10, 5, 0, -5
- Frames per `(noise_name, condition)` stratum: 160
- Source-rank stabilizer percentile: 5.000%
- The gate is never retuned on a seen/unseen or SNR test cell.
- CPU projection source: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a11_conditional_compute\a11_conditional_compute.json`
- CPU projection: `base_ms + activation * refinement_ms_per_selected_frame`, calibrated to the A11 5% measured endpoint; it is not a new hardware benchmark.

## A12.1 SSR Sweep

| SSR | Pool | Activation | Net / selected | F1 | CPU ms/frame | CPU / AlwaysRefine |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10% | all | 12.778% | 14.891% | 0.40794 | 2.3233 | 73.618% |
| 10% | seen | 14.208% | 20.088% | 0.40548 | 2.3333 | 73.934% |
| 10% | unseen | 8.875% | 3.286% | 0.42186 | 2.2960 | 72.755% |
| 30% | all | 9.736% | 14.408% | 0.71904 | 2.3020 | 72.945% |
| 30% | seen | 12.042% | 19.377% | 0.72224 | 2.3181 | 73.455% |
| 30% | unseen | 8.583% | 6.311% | 0.71327 | 2.2940 | 72.690% |
| 50% | all | 8.931% | 15.241% | 0.83967 | 2.2964 | 72.767% |
| 50% | seen | 8.875% | 18.545% | 0.84409 | 2.2960 | 72.755% |
| 50% | unseen | 7.833% | 7.979% | 0.81953 | 2.2888 | 72.524% |
| 70% | all | 6.667% | 11.458% | 0.90761 | 2.2806 | 72.266% |
| 70% | seen | 6.646% | 11.285% | 0.91801 | 2.2805 | 72.261% |
| 70% | unseen | 6.208% | 18.121% | 0.89859 | 2.2774 | 72.165% |
| 90% | all | 4.736% | 17.009% | 0.95628 | 2.2671 | 71.839% |
| 90% | seen | 3.604% | -1.156% | 0.96098 | 2.2592 | 71.589% |
| 90% | unseen | 5.250% | 16.667% | 0.92891 | 2.2707 | 71.953% |

## A12.2 Seen/Unseen x SNR

| SNR | Domain | Raw activation | Raw net / selected | Raw net CI95 | Raw F1 | Rank activation | Rank net / selected |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | seen | 2.804% | 1.739% | [-2.298%, 5.905%] | 0.96686 | 3.153% | 2.431% |
| 20 | unseen | 3.294% | 8.361% | [-2.521%, 19.603%] | 0.96356 | 3.161% | 8.362% |
| 10 | seen | 4.623% | 7.073% | [3.062%, 10.519%] | 0.95850 | 5.126% | 6.202% |
| 10 | unseen | 3.771% | 7.054% | [-1.284%, 13.250%] | 0.95774 | 4.397% | 4.626% |
| 5 | seen | 4.900% | 5.824% | [-1.133%, 12.693%] | 0.95089 | 5.277% | 6.141% |
| 5 | unseen | 8.440% | 31.655% | [5.216%, 39.194%] | 0.89515 | 6.180% | 23.472% |
| 0 | seen | 5.910% | 3.359% | [-2.069%, 10.051%] | 0.94088 | 5.563% | 5.056% |
| 0 | unseen | 6.681% | 17.623% | [-2.484%, 29.349%] | 0.91538 | 6.263% | 12.681% |
| -5 | seen | 5.533% | -0.925% | [-8.395%, 6.600%] | 0.93660 | 5.585% | 0.244% |
| -5 | unseen | 10.606% | 20.489% | [10.959%, 35.141%] | 0.86308 | 8.537% | 21.353% |

## A12.3 Simple Stabilizer

- Triggered point failures: 2
- Resolved by source-rank gate: 2
- Unresolved failures: 0
- Repaired intervals that still include zero: 2
- Weak speaker-cluster intervals: 11
- Maximum raw activation: 14.208% (2.84x nominal)
- Activation warning threshold: 15.000%; failure threshold: 20.000%

## Decision

**GO_WITH_SOURCE_RANK_STABILIZER**

The raw gate fails at least one major cell. The simple source-rank stabilizer is point-positive with bounded activation in every failed cell, but at least one repaired speaker-cluster interval still includes zero. Treat this as conditional router robustness, not a strong GO.

The result distinguishes overall pooled utility from condition-level stability. A positive pooled value does not satisfy A12 if major development cells have non-positive selected utility.
