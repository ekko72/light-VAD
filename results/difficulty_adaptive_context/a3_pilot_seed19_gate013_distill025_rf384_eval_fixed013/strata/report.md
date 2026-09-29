# Phase A3 Refiner Utility Strata

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed19_gate013_distill025_rf384_eval_fixed013\frame_predictions.npz`
- Test activation: 5.021%
- Diagnostic only: thresholds, checkpoints and budgets are frozen.
- Stratum 1 is the most uncertain selected region.

## Confidence Strata Inside The Selected Region

| Group | Stratum | Confidence range | Selected | Long net/selected | Refiner net/selected | Retention |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| overall | 1 | 0.0000-0.0274 | 2996 | 11.348% | 10.581% | 93.235% |
| overall | 2 | 0.0274-0.0549 | 2996 | 12.316% | 11.682% | 94.851% |
| overall | 3 | 0.0549-0.0807 | 2996 | 8.144% | 8.178% | 100.410% |
| overall | 4 | 0.0807-0.1057 | 2995 | 9.583% | 7.412% | 77.352% |
| overall | 5 | 0.1057-0.1300 | 2995 | 7.613% | 5.977% | 78.509% |
| seen | 1 | 0.0000-0.0268 | 1473 | 7.807% | 5.295% | 67.826% |
| seen | 2 | 0.0268-0.0537 | 1473 | 9.165% | 4.752% | 51.852% |
| seen | 3 | 0.0537-0.0790 | 1473 | 3.327% | 2.919% | 87.755% |
| seen | 4 | 0.0790-0.1049 | 1472 | 4.008% | 2.853% | 71.186% |
| seen | 5 | 0.1049-0.1300 | 1472 | 2.378% | 2.038% | 85.714% |
| unseen | 1 | 0.0000-0.0281 | 1006 | 14.414% | 21.571% | 149.655% |
| unseen | 2 | 0.0281-0.0562 | 1006 | 16.501% | 24.254% | 146.988% |
| unseen | 3 | 0.0562-0.0825 | 1006 | 12.823% | 18.489% | 144.186% |
| unseen | 4 | 0.0826-0.1068 | 1005 | 15.423% | 17.512% | 113.548% |
| unseen | 5 | 0.1068-0.1300 | 1005 | 13.134% | 16.219% | 123.485% |

## Per Noise Type

| Noise | Group | Selected | Long net/selected | Refiner net/selected |
| --- | --- | ---: | ---: | ---: |
| Babble_noise | seen | 1444 | 6.371% | 3.186% |
| City_noise | seen | 1314 | 4.795% | 3.196% |
| Domestic_noise | seen | 1137 | 1.935% | 2.463% |
| Nature_noise | seen | 1049 | 5.624% | 6.482% |
| Office_noise | seen | 1058 | 4.159% | 0.095% |
| Public_noise | seen | 1361 | 8.303% | 5.731% |
| SSN_noise | unseen | 2853 | 21.206% | 28.812% |
| Street_noise | unseen | 1494 | 8.568% | 10.710% |
| Transport_noise | unseen | 681 | -0.881% | 0.587% |

## Utterance Concentration

| Group | Utterances | Positive | Negative | Top-10% share of refiner corrections | Top-10% share of refiner harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 95 | 54 | 35 | 53.066% | 42.899% |
| seen | 90 | 48 | 34 | 40.381% | 42.536% |
| unseen | 64 | 37 | 17 | 70.443% | 51.034% |

### Worst unseen utterances by refiner net

| Utterance | Selected | Refiner net | Long net |
| --- | ---: | ---: | ---: |
| test-clean/237/134500/237-134500-0021.flac | 29 | -15 | -2 |
| test-clean/2300/131720/2300-131720-0028.flac | 151 | -6 | -3 |
| test-clean/7176/88083/7176-88083-0001.flac | 64 | -5 | 3 |
| test-clean/121/121726/121-121726-0011.flac | 11 | -4 | -1 |
| test-clean/8463/294828/8463-294828-0028.flac | 36 | -4 | 9 |

`Retention` is the refiner net utility divided by the Long probe net utility on the same frames. A flat retention profile means the refiner degrades uniformly; a falling profile means it fails on the marginal, near-threshold frames.
