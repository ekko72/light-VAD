# Difficulty-Adaptive Temporal Context: A1-A5

## A0 Protocol

- Dataset manifest: `data\librivad\manifests\LibriSpeech_test_medium.tsv`
- Evaluation utterances: 1024
- Valid frames: 553532
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
| Clean | 0.9637 | 0.9663 | 0.0025 | 6.031% | 5.650% |
| 20 | 0.9674 | 0.9676 | 0.0002 | 5.504% | 5.454% |
| 10 | 0.9568 | 0.9583 | 0.0015 | 7.364% | 7.072% |
| 5 | 0.9386 | 0.9482 | 0.0096 | 10.442% | 8.827% |
| 0 | 0.9275 | 0.9328 | 0.0053 | 12.453% | 11.463% |
| -5 | 0.9163 | 0.9244 | 0.0081 | 14.195% | 12.860% |

## A2 Short-Model Difficulty Buckets

Bucket boundaries are global equal-frequency quintiles of `confidence = abs(p_short - 0.5)`.

| Short difficulty | Frames | Confidence range | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Very hard | 110707 | 0.0000-0.3629 | 31.877% | 27.953% | 3.924% |
| Hard | 110706 | 0.3629-0.4727 | 9.312% | 9.818% | -0.506% |
| Medium | 110706 | 0.4727-0.4949 | 2.083% | 2.259% | -0.176% |
| Easy | 110701 | 0.4949-0.4987 | 0.242% | 0.268% | -0.026% |
| Very easy | 110712 | 0.4987-0.5000 | 0.020% | 0.025% | -0.005% |
- Spearman(difficulty rank, error reduction): rho=0.0, p=1.0.

## A3 SNR-by-Context Gain

| SNR | Delta F1 (Long-Short) | Error reduction | Delta AUROC |
| --- | ---: | ---: | ---: |
| Clean | 0.0025 | 0.381% | 0.0020 |
| 20 | 0.0002 | 0.049% | 0.0007 |
| 10 | 0.0015 | 0.292% | -0.0004 |
| 5 | 0.0096 | 1.615% | 0.0073 |
| 0 | 0.0053 | 0.990% | 0.0129 |
| -5 | 0.0081 | 1.336% | 0.0160 |
- Spearman(SNR rank, Delta F1): rho=-0.6571428571428573, p=0.1561749271137024. Lower SNR rank means more noise; a negative rho supports larger gains at lower SNR.

## A4 Boundary Distance

| Distance to nearest GT boundary | Frames | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| 0-50 ms | 52469 | 40.060% | 39.755% | 0.305% |
| 50-100 ms | 43196 | 18.416% | 17.356% | 1.060% |
| 100-200 ms | 74230 | 10.036% | 8.897% | 1.140% |
| >200 ms | 383637 | 3.068% | 2.523% | 0.545% |

## A5 Seen vs Unseen Noise

| Noise group | Frames | Short F1 | Long F1 | Delta F1 | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 305307 | 0.9543 | 0.9543 | -0.0000 | 0.055% |
| unseen | 142650 | 0.9247 | 0.9378 | 0.0131 | 2.092% |

- Unseen minus seen error reduction: 2.037%.
- Unseen minus seen Delta F1: 0.0131.
