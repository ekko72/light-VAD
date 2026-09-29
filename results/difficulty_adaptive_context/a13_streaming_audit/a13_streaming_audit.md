# A13 Streaming and Segment Audit

## Protocol

- Decision threshold: `0.50`.
- Short speech: at most `20` frames (`0.20 s`).
- Onset tolerance: `20` frames.
- Offset delay is measured to the first non-speech frame after a ground-truth speech run; unresolved runs are reported separately.
- Adaptive is compared with Short for non-regression. AlwaysRefine is reported as a diagnostic ceiling.

## Causality Audit

- Device: `cpu`. CPU avoids backend-dependent convolution kernels when comparing chunked and full-sequence execution.

| Check | Status | Detail |
| --- | --- | --- |
| fbank_window_prefix_invariance | PASS | max_abs_diff=0.0, prefix_frames=101 |
| padding_prefix_invariance | PASS | max_abs_diff=0.0 |
| speech_probability_prefix_invariance | PASS | max_abs_diff=0.0 |
| refinement_cache_prefix_invariance | PASS | max_abs_diff=0.0 |
| gate_prefix_invariance | PASS | mismatch_count=0 |
| batching_prefix_isolation | PASS | max_abs_diff=1.1920928955078125e-07, batch_size=2 |
| chunk_boundary_full_equivalence_1 | PASS | max_abs_diff=1.430511474609375e-06 |
| chunk_boundary_gate_equivalence_1 | PASS | mismatch_count=0 |
| chunk_boundary_refinement_cache_equivalence_1 | PASS | max_abs_diff=2.7567148208618164e-07 |
| chunk_boundary_full_equivalence_2 | PASS | max_abs_diff=1.1920928955078125e-06 |
| chunk_boundary_gate_equivalence_2 | PASS | mismatch_count=0 |
| chunk_boundary_refinement_cache_equivalence_2 | PASS | max_abs_diff=1.7881393432617188e-07 |
| normalization_eval_mode | PASS | - |
| hidden_state_recurrent_modules | PASS | recurrent_modules=[] |
| causal_frontend_configuration | PASS | - |

Causality status: **PASS**. No future acoustic frames are consumed by the tested prefix when this audit passes.

## Overall Segment Metrics

| Method | Short recall | Whole-run miss | Onset median | Offset median | Clipping | False activation | Transitions/1k | Runs <=2 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 92.593% | 10 | 3.00 | 2.00 | 165.560s | 115.120s | 18.894 | 25.832% |
| refine_only | 100.000% | 2 | 2.00 | 2.00 | 150.580s | 114.240s | 19.930 | 26.729% |
| adaptive | 98.148% | 5 | 3.00 | 2.00 | 155.300s | 112.110s | 19.360 | 25.957% |

## Adaptive by Group

| Group | Short recall | Whole-run miss | Onset median | Offset median | Clipping | False activation | Transitions/1k | Runs <=2 frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 98.148% | 5 | 3.00 | 2.00 | 155.300s | 112.110s | 19.360 | 25.957% |
| clean | 100.000% | 0 | 3.00 | 0.50 | 27.900s | 8.850s | 18.446 | 18.957% |
| noisy | 97.727% | 5 | 2.00 | 3.00 | 127.400s | 103.260s | 19.579 | 27.528% |
| seen | 100.000% | 0 | 3.00 | 3.00 | 61.740s | 67.200s | 16.955 | 23.760% |
| unseen | 94.737% | 5 | 0.00 | 2.00 | 65.660s | 36.060s | 25.240 | 33.139% |
| condition:clean | 100.000% | 0 | 3.00 | 0.50 | 27.900s | 8.850s | 18.446 | 18.957% |
| condition:20 | 100.000% | 0 | 2.00 | 2.00 | 12.890s | 8.530s | 15.007 | 15.335% |
| condition:10 | 100.000% | 0 | 3.00 | 3.00 | 14.070s | 13.600s | 18.317 | 28.244% |
| condition:5 | 87.500% | 4 | 0.00 | 3.00 | 26.100s | 16.390s | 21.974 | 26.698% |
| condition:0 | 100.000% | 1 | 0.00 | 4.00 | 21.290s | 23.440s | 20.683 | 32.407% |
| condition:-5 | 100.000% | 0 | 1.00 | 3.00 | 38.220s | 26.230s | 24.871 | 34.454% |

## Assessment

A13 status: **PASS**.

- Failed non-regression checks: `0`.
- A12 status remains: `GO_WITH_SOURCE_RANK_STABILIZER`.
- A13 is a streaming/segment diagnostic. It does not override the loaded A12 condition-level routing status (GO_WITH_SOURCE_RANK_STABILIZER).
