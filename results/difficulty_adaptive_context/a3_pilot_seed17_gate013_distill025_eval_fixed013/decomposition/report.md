# Phase A3 Gate-vs-Refiner Decomposition

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_eval_fixed013\frame_predictions.npz`
- Fixed confidence threshold: 0.130000
- Prediction rule: probability >= 0.50

| Group | Frames | Selected | Activation | Long correction | Long harm | Long net/selected | Refiner correction | Refiner harm | Refiner net/selected | Diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| overall | 298300 | 14978 | 5.021% | 3741 | 2273 | 9.801% | 2131 | 1758 | 2.490% | gate and refiner aligned |
| seen | 164375 | 7363 | 4.479% | 1591 | 1198 | 5.337% | 978 | 801 | 2.404% | gate and refiner aligned |
| unseen | 76190 | 5028 | 6.599% | 1448 | 721 | 14.459% | 648 | 645 | 0.060% | gate and refiner aligned |
| clean | 57735 | 2587 | 4.481% | 702 | 354 | 13.452% | 505 | 312 | 7.460% | gate and refiner aligned |
| 20 | 37782 | 1104 | 2.922% | 235 | 186 | 4.438% | 149 | 117 | 2.899% | gate and refiner aligned |
| 10 | 39418 | 1768 | 4.485% | 391 | 294 | 5.486% | 250 | 168 | 4.638% | gate and refiner aligned |
| 5 | 35496 | 2264 | 6.378% | 707 | 301 | 17.933% | 274 | 291 | -0.751% | refiner failure on a valid Long-positive region |
| 0 | 38534 | 2388 | 6.197% | 563 | 379 | 7.705% | 250 | 260 | -0.419% | refiner failure on a valid Long-positive region |
| -5 | 44711 | 3256 | 7.282% | 794 | 508 | 8.784% | 472 | 426 | 1.413% | gate and refiner aligned |

A positive Long net utility means the selected region contains transferable temporal value in the original Short/Long probe. A non-positive Refiner net utility then isolates the failure to the learned refinement.
