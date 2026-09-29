# Phase A3 Refiner Utility Strata

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_rf384_eval_fixed013\frame_predictions.npz`
- Test activation: 5.021%
- Diagnostic only: thresholds, checkpoints and budgets are frozen.
- Stratum 1 is the most uncertain selected region.

## Confidence Strata Inside The Selected Region

| Group | Stratum | Confidence range | Selected | Long net/selected | Refiner net/selected | Retention |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| overall | 1 | 0.0000-0.0274 | 2996 | 11.348% | 11.716% | 103.235% |
| overall | 2 | 0.0274-0.0549 | 2996 | 12.316% | 11.649% | 94.580% |
| overall | 3 | 0.0549-0.0807 | 2996 | 8.144% | 9.513% | 116.803% |
| overall | 4 | 0.0807-0.1057 | 2995 | 9.583% | 6.978% | 72.822% |
| overall | 5 | 0.1057-0.1300 | 2995 | 7.613% | 4.441% | 58.333% |
| seen | 1 | 0.0000-0.0268 | 1473 | 7.807% | 5.635% | 72.174% |
| seen | 2 | 0.0268-0.0537 | 1473 | 9.165% | 5.635% | 61.481% |
| seen | 3 | 0.0537-0.0790 | 1473 | 3.327% | 4.481% | 134.694% |
| seen | 4 | 0.0790-0.1049 | 1472 | 4.008% | 1.698% | 42.373% |
| seen | 5 | 0.1049-0.1300 | 1472 | 2.378% | 0.815% | 34.286% |
| unseen | 1 | 0.0000-0.0281 | 1006 | 14.414% | 22.664% | 157.241% |
| unseen | 2 | 0.0281-0.0562 | 1006 | 16.501% | 23.658% | 143.373% |
| unseen | 3 | 0.0562-0.0825 | 1006 | 12.823% | 18.986% | 148.062% |
| unseen | 4 | 0.0826-0.1068 | 1005 | 15.423% | 17.015% | 110.323% |
| unseen | 5 | 0.1068-0.1300 | 1005 | 13.134% | 12.836% | 97.727% |

## Per Noise Type

| Noise | Group | Selected | Long net/selected | Refiner net/selected |
| --- | --- | ---: | ---: | ---: |
| Babble_noise | seen | 1444 | 6.371% | 3.255% |
| City_noise | seen | 1314 | 4.795% | 3.120% |
| Domestic_noise | seen | 1137 | 1.935% | 3.870% |
| Nature_noise | seen | 1049 | 5.624% | 5.052% |
| Office_noise | seen | 1058 | 4.159% | 1.229% |
| Public_noise | seen | 1361 | 8.303% | 5.217% |
| SSN_noise | unseen | 2853 | 21.206% | 27.830% |
| Street_noise | unseen | 1494 | 8.568% | 12.316% |
| Transport_noise | unseen | 681 | -0.881% | -3.084% |

## Utterance Concentration

| Group | Utterances | Positive | Negative | Top-10% share of refiner corrections | Top-10% share of refiner harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 95 | 64 | 29 | 50.356% | 40.917% |
| seen | 90 | 52 | 28 | 37.540% | 43.693% |
| unseen | 64 | 42 | 14 | 69.657% | 48.068% |

### Worst unseen utterances by refiner net

| Utterance | Selected | Refiner net | Long net |
| --- | ---: | ---: | ---: |
| test-clean/237/134500/237-134500-0021.flac | 29 | -16 | -2 |
| test-clean/6930/81414/6930-81414-0009.flac | 86 | -11 | -15 |
| test-clean/8463/294828/8463-294828-0028.flac | 36 | -6 | 9 |
| test-clean/7176/88083/7176-88083-0001.flac | 64 | -5 | 3 |
| test-clean/6930/76324/6930-76324-0028.flac | 87 | -4 | 12 |

`Retention` is the refiner net utility divided by the Long probe net utility on the same frames. A flat retention profile means the refiner degrades uniformly; a falling profile means it fails on the marginal, near-threshold frames.
