# Phase A3 Refiner Utility Strata

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed18_gate013_distill025_rf384_eval_fixed013\frame_predictions.npz`
- Test activation: 5.021%
- Diagnostic only: thresholds, checkpoints and budgets are frozen.
- Stratum 1 is the most uncertain selected region.

## Confidence Strata Inside The Selected Region

| Group | Stratum | Confidence range | Selected | Long net/selected | Refiner net/selected | Retention |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| overall | 1 | 0.0000-0.0274 | 2996 | 11.348% | 10.447% | 92.059% |
| overall | 2 | 0.0274-0.0549 | 2996 | 12.316% | 12.083% | 98.103% |
| overall | 3 | 0.0549-0.0807 | 2996 | 8.144% | 8.278% | 101.639% |
| overall | 4 | 0.0807-0.1057 | 2995 | 9.583% | 6.912% | 72.125% |
| overall | 5 | 0.1057-0.1300 | 2995 | 7.613% | 4.708% | 61.842% |
| seen | 1 | 0.0000-0.0268 | 1473 | 7.807% | 4.684% | 60.000% |
| seen | 2 | 0.0268-0.0537 | 1473 | 9.165% | 5.974% | 65.185% |
| seen | 3 | 0.0537-0.0790 | 1473 | 3.327% | 3.259% | 97.959% |
| seen | 4 | 0.0790-0.1049 | 1472 | 4.008% | 1.902% | 47.458% |
| seen | 5 | 0.1049-0.1300 | 1472 | 2.378% | 0.679% | 28.571% |
| unseen | 1 | 0.0000-0.0281 | 1006 | 14.414% | 21.670% | 150.345% |
| unseen | 2 | 0.0281-0.0562 | 1006 | 16.501% | 24.354% | 147.590% |
| unseen | 3 | 0.0562-0.0825 | 1006 | 12.823% | 17.893% | 139.535% |
| unseen | 4 | 0.0826-0.1068 | 1005 | 15.423% | 16.119% | 104.516% |
| unseen | 5 | 0.1068-0.1300 | 1005 | 13.134% | 13.532% | 103.030% |

## Per Noise Type

| Noise | Group | Selected | Long net/selected | Refiner net/selected |
| --- | --- | ---: | ---: | ---: |
| Babble_noise | seen | 1444 | 6.371% | 2.078% |
| City_noise | seen | 1314 | 4.795% | 2.664% |
| Domestic_noise | seen | 1137 | 1.935% | 2.463% |
| Nature_noise | seen | 1049 | 5.624% | 5.815% |
| Office_noise | seen | 1058 | 4.159% | 0.756% |
| Public_noise | seen | 1361 | 8.303% | 5.952% |
| SSN_noise | unseen | 2853 | 21.206% | 27.059% |
| Street_noise | unseen | 1494 | 8.568% | 11.580% |
| Transport_noise | unseen | 681 | -0.881% | -0.587% |

## Utterance Concentration

| Group | Utterances | Positive | Negative | Top-10% share of refiner corrections | Top-10% share of refiner harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 95 | 56 | 35 | 51.329% | 40.776% |
| seen | 90 | 46 | 34 | 38.940% | 39.112% |
| unseen | 64 | 37 | 16 | 69.588% | 49.492% |

### Worst unseen utterances by refiner net

| Utterance | Selected | Refiner net | Long net |
| --- | ---: | ---: | ---: |
| test-clean/7729/102255/7729-102255-0037.flac | 85 | -14 | -6 |
| test-clean/237/134500/237-134500-0021.flac | 29 | -13 | -2 |
| test-clean/6930/81414/6930-81414-0009.flac | 86 | -12 | -15 |
| test-clean/8463/294828/8463-294828-0028.flac | 36 | -9 | 9 |
| test-clean/908/157963/908-157963-0017.flac | 16 | -5 | 6 |

`Retention` is the refiner net utility divided by the Long probe net utility on the same frames. A flat retention profile means the refiner degrades uniformly; a falling profile means it fails on the marginal, near-threshold frames.
