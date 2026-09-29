# Phase A3 Refiner-vs-Long Alignment

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill050_eval_fixed013\frame_predictions.npz`
- Margins use `logit(p)` differences from the shared Short output.
- This is a diagnostic only; it does not alter thresholds or checkpoints.

| Group | Selected | Probe net | Refiner net | On probe corrections: refiner correct | On probe harms: refiner correct | Residual same direction | Median residual magnitude ratio | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 14978 | 1468 | 246 | 1248/3741 (33.360%) | 1620/2273 (71.271%) | 8511/14978 (56.823%) | 0.3449 | 0.2074 |
| seen | 7363 | 393 | 156 | 568/1591 (35.701%) | 868/1198 (72.454%) | 4233/7363 (57.490%) | 0.3709 | 0.2751 |
| unseen | 5028 | 727 | -50 | 388/1448 (26.796%) | 486/721 (67.406%) | 2762/5028 (54.932%) | 0.3347 | 0.1227 |
| clean | 2587 | 348 | 140 | 292/702 (41.595%) | 266/354 (75.141%) | 1516/2587 (58.601%) | 0.2923 | 0.1897 |
| 20 | 1104 | 49 | 33 | 80/235 (34.043%) | 152/186 (81.720%) | 598/1104 (54.167%) | 0.3544 | 0.1922 |
| 10 | 1768 | 97 | 76 | 158/391 (40.409%) | 204/294 (69.388%) | 1095/1768 (61.934%) | 0.3539 | 0.3348 |
| 5 | 2264 | 406 | -43 | 177/707 (25.035%) | 196/301 (65.116%) | 1207/2264 (53.313%) | 0.3521 | 0.1465 |
| 0 | 2388 | 184 | 0 | 152/563 (26.998%) | 267/379 (70.449%) | 1339/2388 (56.072%) | 0.3303 | 0.1391 |
| -5 | 3256 | 286 | 13 | 270/794 (34.005%) | 345/508 (67.913%) | 1830/3256 (56.204%) | 0.3966 | 0.2292 |

`On probe corrections` asks how often the refiner also corrects frames where Long corrects Short. `On probe harms` asks how often the refiner avoids Long's wrong flips. A positive Pearson and a small magnitude ratio indicate an underpowered refiner; a near-zero or negative Pearson indicates a mismatched refinement signal.
