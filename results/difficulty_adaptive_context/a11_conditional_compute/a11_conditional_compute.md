# A11 Conditional Compute Realization

## Protocol

- Manifest: `C:\Users\20547\Desktop\light-VAD\data\librivad\manifests\LibriSpeech_test_medium.tsv`
- RF384 checkpoint: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_rf384\best.pt`
- RF64 checkpoint: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025\best.pt`
- RF384 predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\eval_rf384_fixed130\frame_predictions.npz`
- RF64 predictions: `C:\Users\20547\Desktop\light-VAD\results\difficulty_adaptive_context\a9_span_sweep\seed17\eval_rf64_fixed130\frame_predictions.npz`
- Benchmark utterances: 8
- Timed repeats per utterance: 1
- Warm-up frames: 200
- Primary timing boundary: cached causal MFCC frames
- Streaming chunk: 1 frame(s)
- CPU: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
- Affinity/threads: CPU 0, 1 torch thread(s)
- Power scheme: 电源方案 GUID: 64a64f24-65b9-4b56-befd-5ec1eaced9b3  (Silent)

- The threshold for every adaptive operating point is selected only from calibration-speaker Short scores. Final-test activation is observed, not forced.
- Latency excludes MFCC extraction. RTF therefore describes the cached-feature streaming boundary, not full audio-to-decision RTF.
- Activation percentage is not called compute percentage; MACs and wall-clock timing are reported separately.

## Operating Points

| Operating point | Test activation | Benchmark activation | Threshold | F1 | FAR | MR | Mean ms/frame | Median | P95 | P99 | RTF | Analytical MACs/frame (test act.) | Cache bytes | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 0.000% | 0.000% | n/a | 0.94363 | 24.580% | 6.584% | 2.19516 | 2.00480 | 3.52298 | 4.26822 | 0.21952 | 87,232 | 34,304 | 604.78 |
| RF64 AlwaysRefine | 100.000% | 100.000% | n/a | 0.94444 | 25.381% | 6.297% | 3.05154 | 2.83180 | 4.63282 | 5.50098 | 0.30515 | 90,688 | 67,072 | 606.21 |
| RF384 AlwaysRefine | 100.000% | 100.000% | n/a | 0.94696 | 24.392% | 5.988% | 3.15586 | 2.99090 | 4.67458 | 5.36036 | 0.31559 | 90,688 | 230,912 | 606.12 |
| Adaptive 2% | 1.983% | 2.689% | 0.054184 | 0.94505 | 24.132% | 6.391% | 2.00250 | 1.80220 | 3.39522 | 4.11552 | 0.20025 | 87,301 | 230,912 | 606.49 |
| Adaptive 5% | 5.050% | 5.423% | 0.130599 | 0.94638 | 23.939% | 6.173% | 2.26933 | 2.08070 | 3.63414 | 4.44857 | 0.22693 | 87,407 | 230,912 | 604.59 |
| Adaptive 10% | 10.317% | 10.276% | 0.234942 | 0.94693 | 24.262% | 6.015% | 2.34580 | 2.15900 | 3.79840 | 4.56973 | 0.23458 | 87,589 | 230,912 | 605.58 |
| Adaptive 20% | 21.459% | 20.141% | 0.366736 | 0.94696 | 24.392% | 5.988% | 2.45723 | 2.31170 | 3.96246 | 4.77383 | 0.24572 | 87,974 | 230,912 | 607.13 |

## Router and Scheduling Overhead

- Short encoder + classifier mean: 1.81556 ms/frame.
- Short + gate mean: 2.19516 ms/frame.
- Gate-only overhead estimate: 0.37961 ms/frame.
- Scheduling overhead below is measured adaptive mean latency minus `Short+gate + observed activation * isolated refinement kernel`; it includes history concatenation, dispatch and other implementation overhead.

| Adaptive point | Observed activation | Isolated refinement kernel ms/frame | Estimated scheduling overhead ms/frame |
| --- | ---: | ---: | ---: |
| Adaptive 2% | 2.689% | 1.03425 | -0.22047 |
| Adaptive 5% | 5.423% | 0.69809 | 0.03631 |
| Adaptive 10% | 10.276% | 0.74913 | 0.07366 |
| Adaptive 20% | 20.141% | 0.64582 | 0.13199 |

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
