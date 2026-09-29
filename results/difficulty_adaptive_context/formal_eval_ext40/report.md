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
| Clean | 0.9623 | 0.9671 | 0.0048 | 6.257% | 5.509% |
| 20 | 0.9672 | 0.9673 | 0.0001 | 5.532% | 5.506% |
| 10 | 0.9559 | 0.9586 | 0.0027 | 7.496% | 7.018% |
| 5 | 0.9378 | 0.9484 | 0.0105 | 10.557% | 8.814% |
| 0 | 0.9264 | 0.9340 | 0.0076 | 12.609% | 11.313% |
| -5 | 0.9144 | 0.9254 | 0.0110 | 14.468% | 12.728% |

## A2 Short-Model Difficulty Buckets

Bucket boundaries are global equal-frequency quintiles of `confidence = abs(p_short - 0.5)`.

| Short difficulty | Frames | Confidence range | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Very hard | 110707 | 0.0000-0.3598 | 32.261% | 27.552% | 4.709% |
| Hard | 110706 | 0.3598-0.4728 | 9.551% | 9.843% | -0.293% |
| Medium | 110704 | 0.4728-0.4950 | 2.144% | 2.272% | -0.128% |
| Easy | 110702 | 0.4950-0.4988 | 0.251% | 0.265% | -0.014% |
| Very easy | 110713 | 0.4988-0.5000 | 0.019% | 0.020% | -0.001% |
- Spearman(difficulty rank, error reduction): rho=0.0, p=1.0.

## A3 SNR-by-Context Gain

| SNR | Delta F1 (Long-Short) | Error reduction | Delta AUROC |
| --- | ---: | ---: | ---: |
| Clean | 0.0048 | 0.748% | 0.0046 |
| 20 | 0.0001 | 0.026% | 0.0007 |
| 10 | 0.0027 | 0.479% | 0.0004 |
| 5 | 0.0105 | 1.743% | 0.0073 |
| 0 | 0.0076 | 1.296% | 0.0131 |
| -5 | 0.0110 | 1.739% | 0.0160 |
- Spearman(SNR rank, Delta F1): rho=-0.7714285714285715, p=0.07239650145772594. Lower SNR rank means more noise; a negative rho supports larger gains at lower SNR.

## A4 Boundary Distance

| Distance to nearest GT boundary | Frames | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| 0-50 ms | 52469 | 40.058% | 39.541% | 0.516% |
| 50-100 ms | 43196 | 18.474% | 17.235% | 1.239% |
| 100-200 ms | 74230 | 10.162% | 8.852% | 1.309% |
| >200 ms | 383637 | 3.237% | 2.467% | 0.770% |

## A5 Seen vs Unseen Noise

| Noise group | Frames | Short F1 | Long F1 | Delta F1 | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 305307 | 0.9543 | 0.9547 | 0.0004 | 0.099% |
| unseen | 142650 | 0.9221 | 0.9383 | 0.0163 | 2.551% |

- Unseen minus seen error reduction: 2.452%.
- Unseen minus seen Delta F1: 0.0159.
