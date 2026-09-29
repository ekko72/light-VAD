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
| Clean | 0.9661 | 0.9686 | 0.0025 | 5.709% | 5.277% |
| 20 | 0.9671 | 0.9668 | -0.0003 | 5.572% | 5.582% |
| 10 | 0.9578 | 0.9599 | 0.0021 | 7.204% | 6.833% |
| 5 | 0.9485 | 0.9549 | 0.0064 | 8.890% | 7.802% |
| 0 | 0.9340 | 0.9391 | 0.0051 | 11.446% | 10.564% |
| -5 | 0.9274 | 0.9307 | 0.0034 | 12.475% | 12.000% |

## A2 Short-Model Difficulty Buckets

Bucket boundaries are global equal-frequency quintiles of `confidence = abs(p_short - 0.5)`.

| Short difficulty | Frames | Confidence range | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Very hard | 110707 | 0.0000-0.3635 | 28.945% | 26.344% | 2.601% |
| Hard | 110706 | 0.3635-0.4738 | 9.049% | 9.410% | -0.360% |
| Medium | 110706 | 0.4738-0.4947 | 2.081% | 2.105% | -0.023% |
| Easy | 110705 | 0.4947-0.4987 | 0.276% | 0.276% | 0.000% |
| Very easy | 110708 | 0.4987-0.5000 | 0.031% | 0.031% | 0.000% |
- Spearman(difficulty rank, error reduction): rho=0.051298917604257706, p=0.934712848108234.

## A3 SNR-by-Context Gain

| SNR | Delta F1 (Long-Short) | Error reduction | Delta AUROC |
| --- | ---: | ---: | ---: |
| Clean | 0.0025 | 0.432% | 0.0041 |
| 20 | -0.0003 | -0.010% | 0.0016 |
| 10 | 0.0021 | 0.372% | 0.0007 |
| 5 | 0.0064 | 1.088% | 0.0068 |
| 0 | 0.0051 | 0.883% | 0.0169 |
| -5 | 0.0034 | 0.475% | 0.0043 |
- Spearman(SNR rank, Delta F1): rho=-0.6, p=0.208. Lower SNR rank means more noise; a negative rho supports larger gains at lower SNR.

## A4 Boundary Distance

| Distance to nearest GT boundary | Frames | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| 0-50 ms | 52469 | 39.633% | 39.538% | 0.095% |
| 50-100 ms | 43196 | 18.092% | 17.559% | 0.532% |
| 100-200 ms | 74230 | 9.883% | 8.984% | 0.899% |
| >200 ms | 383637 | 2.283% | 1.890% | 0.393% |

## A5 Seen vs Unseen Noise

| Noise group | Frames | Short F1 | Long F1 | Delta F1 | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 305307 | 0.9542 | 0.9549 | 0.0007 | 0.153% |
| unseen | 142650 | 0.9395 | 0.9463 | 0.0068 | 1.073% |

- Unseen minus seen error reduction: 0.920%.
- Unseen minus seen Delta F1: 0.0061.
