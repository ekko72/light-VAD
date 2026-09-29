# Phase A3 Refiner Utility Strata

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_eval_fixed013\frame_predictions.npz`
- Test activation: 5.021%
- Diagnostic only: thresholds, checkpoints and budgets are frozen.
- Stratum 1 is the most uncertain selected region.

## Confidence Strata Inside The Selected Region

| Group | Stratum | Confidence range | Selected | Long net/selected | Refiner net/selected | Retention |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| overall | 1 | 0.0000-0.0274 | 2996 | 11.348% | 4.773% | 42.059% |
| overall | 2 | 0.0274-0.0549 | 2996 | 12.316% | 5.107% | 41.463% |
| overall | 3 | 0.0549-0.0807 | 2996 | 8.144% | 1.469% | 18.033% |
| overall | 4 | 0.0807-0.1057 | 2995 | 9.583% | 1.068% | 11.150% |
| overall | 5 | 0.1057-0.1300 | 2995 | 7.613% | 0.033% | 0.439% |
| seen | 1 | 0.0000-0.0268 | 1473 | 7.807% | 4.345% | 55.652% |
| seen | 2 | 0.0268-0.0537 | 1473 | 9.165% | 4.752% | 51.852% |
| seen | 3 | 0.0537-0.0790 | 1473 | 3.327% | 2.172% | 65.306% |
| seen | 4 | 0.0790-0.1049 | 1472 | 4.008% | 1.427% | 35.593% |
| seen | 5 | 0.1049-0.1300 | 1472 | 2.378% | -0.679% | -28.571% |
| unseen | 1 | 0.0000-0.0281 | 1006 | 14.414% | 0.795% | 5.517% |
| unseen | 2 | 0.0281-0.0562 | 1006 | 16.501% | 1.789% | 10.843% |
| unseen | 3 | 0.0562-0.0825 | 1006 | 12.823% | -0.795% | -6.202% |
| unseen | 4 | 0.0826-0.1068 | 1005 | 15.423% | -1.194% | -7.742% |
| unseen | 5 | 0.1068-0.1300 | 1005 | 13.134% | -0.299% | -2.273% |

## Per Noise Type

| Noise | Group | Selected | Long net/selected | Refiner net/selected |
| --- | --- | ---: | ---: | ---: |
| Babble_noise | seen | 1444 | 6.371% | 1.524% |
| City_noise | seen | 1314 | 4.795% | 1.979% |
| Domestic_noise | seen | 1137 | 1.935% | 3.342% |
| Nature_noise | seen | 1049 | 5.624% | 7.054% |
| Office_noise | seen | 1058 | 4.159% | -0.662% |
| Public_noise | seen | 1361 | 8.303% | 1.763% |
| SSN_noise | unseen | 2853 | 21.206% | -3.680% |
| Street_noise | unseen | 1494 | 8.568% | 7.430% |
| Transport_noise | unseen | 681 | -0.881% | -0.441% |

## Utterance Concentration

| Group | Utterances | Positive | Negative | Top-10% share of refiner corrections | Top-10% share of refiner harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 95 | 55 | 34 | 44.158% | 48.521% |
| seen | 90 | 51 | 31 | 39.877% | 45.069% |
| unseen | 64 | 34 | 22 | 56.481% | 67.907% |

### Worst unseen utterances by refiner net

| Utterance | Selected | Refiner net | Long net |
| --- | ---: | ---: | ---: |
| test-clean/1188/133604/1188-133604-0026.flac | 689 | -79 | 43 |
| test-clean/1188/133604/1188-133604-0016.flac | 405 | -48 | 4 |
| test-clean/8230/279154/8230-279154-0038.flac | 471 | -41 | 132 |
| test-clean/7176/92135/7176-92135-0043.flac | 281 | -19 | 74 |
| test-clean/2961/960/2961-960-0016.flac | 71 | -10 | 21 |

`Retention` is the refiner net utility divided by the Long probe net utility on the same frames. A flat retention profile means the refiner degrades uniformly; a falling profile means it fails on the marginal, near-threshold frames.
