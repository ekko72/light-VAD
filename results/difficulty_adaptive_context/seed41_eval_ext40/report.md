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
| Clean | 0.9612 | 0.9646 | 0.0034 | 6.544% | 5.944% |
| 20 | 0.9671 | 0.9670 | -0.0001 | 5.547% | 5.565% |
| 10 | 0.9547 | 0.9564 | 0.0017 | 7.713% | 7.415% |
| 5 | 0.9429 | 0.9515 | 0.0085 | 9.792% | 8.350% |
| 0 | 0.9300 | 0.9338 | 0.0038 | 12.021% | 11.405% |
| -5 | 0.9161 | 0.9252 | 0.0091 | 14.228% | 12.891% |

## A2 Short-Model Difficulty Buckets

Bucket boundaries are global equal-frequency quintiles of `confidence = abs(p_short - 0.5)`.

| Short difficulty | Frames | Confidence range | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Very hard | 110707 | 0.0000-0.3591 | 31.522% | 28.131% | 3.391% |
| Hard | 110706 | 0.3591-0.4732 | 9.069% | 9.303% | -0.234% |
| Medium | 110706 | 0.4732-0.4943 | 2.511% | 2.517% | -0.006% |
| Easy | 110704 | 0.4943-0.4984 | 0.647% | 0.648% | -0.001% |
| Very easy | 110709 | 0.4984-0.5000 | 0.057% | 0.057% | 0.000% |
- Spearman(difficulty rank, error reduction): rho=0.0, p=1.0.

## A3 SNR-by-Context Gain

| SNR | Delta F1 (Long-Short) | Error reduction | Delta AUROC |
| --- | ---: | ---: | ---: |
| Clean | 0.0034 | 0.601% | 0.0039 |
| 20 | -0.0001 | -0.018% | 0.0010 |
| 10 | 0.0017 | 0.297% | 0.0058 |
| 5 | 0.0085 | 1.442% | 0.0163 |
| 0 | 0.0038 | 0.615% | 0.0173 |
| -5 | 0.0091 | 1.337% | 0.0095 |
- Spearman(SNR rank, Delta F1): rho=-0.7714285714285715, p=0.07239650145772594. Lower SNR rank means more noise; a negative rho supports larger gains at lower SNR.

## A4 Boundary Distance

| Distance to nearest GT boundary | Frames | Short error | Long error | Error reduction |
| --- | ---: | ---: | ---: | ---: |
| 0-50 ms | 52469 | 39.318% | 39.532% | -0.213% |
| 50-100 ms | 43196 | 18.300% | 17.965% | 0.336% |
| 100-200 ms | 74230 | 10.399% | 9.441% | 0.958% |
| >200 ms | 383637 | 3.191% | 2.476% | 0.715% |

## A5 Seen vs Unseen Noise

| Noise group | Frames | Short F1 | Long F1 | Delta F1 | Error reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| seen | 305307 | 0.9542 | 0.9536 | -0.0007 | -0.088% |
| unseen | 142650 | 0.9262 | 0.9403 | 0.0140 | 2.188% |

- Unseen minus seen error reduction: 2.276%.
- Unseen minus seen Delta F1: 0.0147.
