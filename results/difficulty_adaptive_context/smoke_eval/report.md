# Difficulty-Adaptive Temporal Context: A1-A5

## A0 Protocol

- Dataset manifest: `data\librivad\manifests\LibriSpeech_test_medium.tsv`
- Evaluation utterances: 56
- Valid frames: 27438
- Context used for training: 4.00 s
- Target segment: 4.00 s
- Seed: 17
- Short-RF: `short`, 123 frames (1.23 s)
- Long-RF: `long`, 383 frames (3.83 s)
- Parameters: Short=89,154, Long=89,154
- Unseen noise classes: SSN_noise, Street_noise, Transport_noise

The two models use the same manifest rows, frame labels, MFCC frontend, channels, kernels, loss, optimizer, seed and training budget. The only intentional difference is the five-block dilation profile.

## A1 Overall Long vs Short

| SNR | Short F1 | Long F1 | Delta F1 | Short error | Long error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.9120 | 0.9120 | 0.0000 | 16.182% | 16.182% |
| 20 | 0.8881 | 0.8881 | 0.0000 | 20.128% | 20.128% |
| 10 | 0.9370 | 0.9370 | 0.0000 | 11.854% | 11.854% |
| 5 | 0.9146 | 0.9146 | 0.0000 | 15.733% | 15.733% |
| 0 | 0.9101 | 0.9101 | 0.0000 | 16.492% | 16.492% |
| -5 | 0.8998 | 0.8998 | 0.0000 | 18.221% | 18.221% |

## A2 Short-Model Difficulty Buckets

Bucket boundaries are global equal-frequency quintiles of `confidence = abs(p_short - 0.5)`.

| Short difficulty | Frames | Confidence range | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Very hard | 5488 | 0.1886-0.2206 | 10.897% | 10.897% | 0.000% |
| Hard | 5487 | 0.2206-0.2302 | 10.151% | 10.151% | 0.000% |
| Medium | 5488 | 0.2302-0.2426 | 13.867% | 13.867% | 0.000% |
| Easy | 5487 | 0.2426-0.2602 | 20.703% | 20.703% | 0.000% |
| Very easy | 5488 | 0.2602-0.3256 | 25.292% | 25.292% | 0.000% |
- Spearman(difficulty rank, error reduction): rho=None, p=None.

## A3 SNR-by-Context Gain

| SNR | Delta F1 (Long-Short) | Error reduction | Delta AUROC |
| --- | ---: | ---: | ---: |
| Clean | 0.0000 | 0.000% | 0.4065 |
| 20 | 0.0000 | 0.000% | -0.1232 |
| 10 | 0.0000 | 0.000% | 0.0528 |
| 5 | 0.0000 | 0.000% | -0.2140 |
| 0 | 0.0000 | 0.000% | -0.2031 |
| -5 | 0.0000 | 0.000% | 0.0332 |
- Spearman(SNR rank, Delta F1): rho=None, p=None. Lower SNR rank means more noise; a negative rho supports larger gains at lower SNR.

## A4 Boundary Distance

| Distance to nearest GT boundary | Frames | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| 0-50 ms | 2962 | 46.050% | 46.050% | 0.000% |
| 50-100 ms | 2432 | 39.885% | 39.885% | 0.000% |
| 100-200 ms | 4066 | 29.267% | 29.267% | 0.000% |
| >200 ms | 17978 | 5.095% | 5.095% | 0.000% |

## A5 Seen vs Unseen Noise

| Noise group | Frames | Short F1 | Long F1 | Delta F1 | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 13719 | 0.9120 | 0.9120 | 0.0000 | 0.000% |
| unseen | 0 | n/a | n/a | n/a | n/a |

- Unseen minus seen error reduction: 0.000%.
- Unseen minus seen Delta F1: 0.0000.
