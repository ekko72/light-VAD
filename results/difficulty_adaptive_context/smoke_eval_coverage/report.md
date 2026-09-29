# Difficulty-Adaptive Temporal Context: A1-A5

## A0 Protocol

- Dataset manifest: `data\librivad\manifests\LibriSpeech_test_medium.tsv`
- Evaluation utterances: 80
- Valid frames: 42708
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
| Clean | 0.9184 | 0.9184 | 0.0000 | 15.083% | 15.083% |
| 20 | 0.8946 | 0.8946 | 0.0000 | 19.062% | 19.062% |
| 10 | 0.9322 | 0.9322 | 0.0000 | 12.691% | 12.691% |
| 5 | 0.9266 | 0.9266 | 0.0000 | 13.676% | 13.676% |
| 0 | 0.9162 | 0.9162 | 0.0000 | 15.456% | 15.456% |
| -5 | 0.9087 | 0.9087 | 0.0000 | 16.726% | 16.726% |

## A2 Short-Model Difficulty Buckets

Bucket boundaries are global equal-frequency quintiles of `confidence = abs(p_short - 0.5)`.

| Short difficulty | Frames | Confidence range | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Very hard | 8542 | 0.1886-0.2215 | 8.956% | 8.956% | 0.000% |
| Hard | 8541 | 0.2215-0.2324 | 9.144% | 9.144% | 0.000% |
| Medium | 8542 | 0.2324-0.2473 | 14.212% | 14.212% | 0.000% |
| Easy | 8541 | 0.2473-0.2655 | 18.944% | 18.944% | 0.000% |
| Very easy | 8542 | 0.2655-0.3286 | 23.987% | 23.987% | 0.000% |
- Spearman(difficulty rank, error reduction): rho=None, p=None.

## A3 SNR-by-Context Gain

| SNR | Delta F1 (Long-Short) | Error reduction | Delta AUROC |
| --- | ---: | ---: | ---: |
| Clean | 0.0000 | 0.000% | 0.4284 |
| 20 | 0.0000 | 0.000% | 0.0762 |
| 10 | 0.0000 | 0.000% | 0.0251 |
| 5 | 0.0000 | 0.000% | -0.0811 |
| 0 | 0.0000 | 0.000% | -0.1184 |
| -5 | 0.0000 | 0.000% | -0.0790 |
- Spearman(SNR rank, Delta F1): rho=None, p=None. Lower SNR rank means more noise; a negative rho supports larger gains at lower SNR.

## A4 Boundary Distance

| Distance to nearest GT boundary | Frames | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| 0-50 ms | 4358 | 45.617% | 45.617% | 0.000% |
| 50-100 ms | 3550 | 38.817% | 38.817% | 0.000% |
| 100-200 ms | 5949 | 27.786% | 27.786% | 0.000% |
| >200 ms | 28851 | 4.880% | 4.880% | 0.000% |

## A5 Seen vs Unseen Noise

| Noise group | Frames | Short F1 | Long F1 | Delta F1 | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 13719 | 0.9120 | 0.9120 | 0.0000 | 0.000% |
| unseen | 9371 | 0.9287 | 0.9287 | 0.0000 | 0.000% |

- Unseen minus seen error reduction: 0.000%.
- Unseen minus seen Delta F1: 0.0000.
