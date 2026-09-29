# Phase A3 Gate-vs-Refiner Decomposition

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill050_eval_fixed013\frame_predictions.npz`
- Fixed confidence threshold: 0.130000
- Prediction rule: probability >= 0.50

| Group | Frames | Selected | Activation | Long correction | Long harm | Long net/selected | Refiner correction | Refiner harm | Refiner net/selected | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| overall | 298300 | 14978 | 5.021% | 3741 | 2273 | 9.801% | 1894 | 1648 | 1.642% | gate and refiner aligned |
| seen | 164375 | 7363 | 4.479% | 1591 | 1198 | 5.337% | 892 | 736 | 2.119% | gate and refiner aligned |
| unseen | 76190 | 5028 | 6.599% | 1448 | 721 | 14.459% | 579 | 629 | -0.994% | refiner failure on a valid Long-positive region |
| clean | 57735 | 2587 | 4.481% | 702 | 354 | 13.452% | 423 | 283 | 5.412% | gate and refiner aligned |
| 20 | 37782 | 1104 | 2.922% | 235 | 186 | 4.438% | 136 | 103 | 2.989% | gate and refiner aligned |
| 10 | 39418 | 1768 | 4.485% | 391 | 294 | 5.486% | 236 | 160 | 4.299% | gate and refiner aligned |
| 5 | 35496 | 2264 | 6.378% | 707 | 301 | 17.933% | 250 | 293 | -1.899% | refiner failure on a valid Long-positive region |
| 0 | 38534 | 2388 | 6.197% | 563 | 379 | 7.705% | 244 | 244 | 0.000% | refiner failure on a valid Long-positive region |
| -5 | 44711 | 3256 | 7.282% | 794 | 508 | 8.784% | 410 | 397 | 0.399% | gate and refiner aligned |

A positive Long net utility means the selected region contains transferable temporal value in the original Short/Long probe. A non-positive Refiner net utility then isolates the failure to the learned refinement.
