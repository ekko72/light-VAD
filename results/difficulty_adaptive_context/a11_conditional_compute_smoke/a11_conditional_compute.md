# A11 Conditional Compute Realization

## Protocol

- Manifest: `C:\Users\20547\Desktop\light-VAD\data\librivad\manifests\LibriSpeech_test_medium.tsv`
- RF384 checkpoint: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_rf384\best.pt`
- RF64 checkpoint: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025\best.pt`
- RF384 predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\eval_rf384_fixed130\frame_predictions.npz`
- RF64 predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\eval_rf64_fixed130\frame_predictions.npz`
- Benchmark utterances: 1
- Timed repeats per utterance: 1
- Warm-up frames: 5
- Primary timing boundary: cached causal MFCC frames
- Streaming chunk: 1 frame(s)
- CPU: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
- Affinity/threads: CPU 0, 1 torch thread(s)
- Power scheme: 电源方案 GUID: 64a64f24-65b9-4b56-befd-5ec1eaced9b3  (Silent)

- The threshold for every adaptive operating point is selected only from calibration-speaker Short scores. Final-test activation is observed, not forced.
- Latency excludes MFCC extraction. RTF therefore describes the cached-feature streaming boundary, not full audio-to-decision RTF.
- Activation percentage is not called compute percentage; MACs and wall-clock timing are reported separately.

## Operating Points

| Operating point | Test activation | Benchmark activation | Threshold | F1 | FAR | MR | Mean ms/frame | Median | P95 | P99 | RTF | Analytical MACs/frame | Cache bytes | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 0.000% | 0.000% | n/a | 0.94363 | 24.580% | 6.584% | 2.11640 | 1.89320 | 3.35932 | 4.05385 | 0.21164 | 87,232 | 34,304 | 605.19 |
| RF64 AlwaysRefine | 100.000% | 100.000% | n/a | 0.94444 | 25.381% | 6.297% | 3.17594 | 2.96990 | 4.78732 | 5.18882 | 0.31759 | 90,688 | 67,072 | 605.24 |
| RF384 AlwaysRefine | 100.000% | 100.000% | n/a | 0.94696 | 24.392% | 5.988% | 2.96482 | 2.78750 | 4.14402 | 4.55182 | 0.29648 | 90,688 | 145,408 | 605.33 |
| Adaptive 2% | 1.983% | 0.000% | 0.054184 | 0.94505 | 24.132% | 6.391% | 2.20568 | 2.03330 | 3.25082 | 3.74625 | 0.22057 | 87,301 | 145,408 | 605.21 |
| Adaptive 5% | 5.050% | 0.000% | 0.130599 | 0.94638 | 23.939% | 6.173% | 2.17182 | 1.93890 | 3.46870 | 4.06714 | 0.21718 | 87,407 | 145,408 | 605.15 |
| Adaptive 10% | 10.317% | 0.461% | 0.234942 | 0.94693 | 24.262% | 6.015% | 2.35414 | 2.14710 | 3.83052 | 4.53988 | 0.23541 | 87,589 | 145,408 | 605.03 |
| Adaptive 20% | 21.459% | 6.452% | 0.366736 | 0.94696 | 24.392% | 5.988% | 2.23757 | 2.05850 | 3.37448 | 3.97770 | 0.22376 | 87,974 | 145,408 | 605.38 |

## Router and Scheduling Overhead

- Short encoder + classifier mean: 1.80849 ms/frame.
- Short + gate mean: 2.11640 ms/frame.
- Gate-only overhead estimate: 0.30792 ms/frame.
- Scheduling overhead below is measured adaptive mean latency minus `Short+gate + observed activation * isolated refinement kernel`; it includes history concatenation, dispatch and other implementation overhead.

| Adaptive point | Observed activation | Isolated refinement kernel ms/frame | Estimated scheduling overhead ms/frame |
| --- | ---: | ---: | ---: |
| Adaptive 2% | 0.000% | 0.55008 | 0.08928 |
| Adaptive 5% | 0.000% | 0.85382 | 0.05542 |
| Adaptive 10% | 0.461% | 0.78332 | 0.23413 |
| Adaptive 20% | 6.452% | 0.89402 | 0.06349 |

## GO Assessment

- Primary operating point: adaptive 5%.
- F1 order `Short < Adaptive < AlwaysRefine`: True.
- Latency order `Short < Adaptive < AlwaysRefine`: True.
- Adaptive is near AlwaysRefine under the predeclared 5.0% relative latency-gap rule: False.
- Status: **STRONG_GO**.
- The 5% operating point satisfies the strict accuracy and latency ordering required by A11.

## Limitations

- The benchmark uses one frame per streaming call and cached MFCC input; it does not measure microphone capture or feature extraction.
- CPU frequency pinning is not enforced. The active Windows power scheme and CPU affinity are recorded, but background frequency changes remain a possible noise source.
- Peak RSS is collected from an isolated worker process per operating point; it is process peak working set, not tensor-level peak allocation.
- Analytical MACs count Conv1d and Linear multiply-accumulates only. They exclude indexing, concatenation, activation, scheduling and memory traffic.
