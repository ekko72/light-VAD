# Phase A3 Refiner-vs-Long Alignment

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_rf384_eval_fixed013\frame_predictions.npz`
- Margins use `logit(p)` differences from the shared Short output.
- This is a diagnostic only; it does not alter thresholds or checkpoints.

| Group | Selected | Probe net | Refiner net | On probe corrections: refiner correct | On probe harms: refiner correct | Residual same direction | Median residual magnitude ratio | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 14978 | 1468 | 1327 | 2062/3741 (55.119%) | 1511/2273 (66.476%) | 9305/14978 (62.124%) | 0.5387 | 0.3166 |
| seen | 7363 | 393 | 269 | 805/1591 (50.597%) | 735/1198 (61.352%) | 4489/7363 (60.967%) | 0.6060 | 0.3504 |
| unseen | 5028 | 727 | 957 | 935/1448 (64.572%) | 531/721 (73.648%) | 3317/5028 (65.971%) | 0.5498 | 0.2860 |
| clean | 2587 | 348 | 101 | 322/702 (45.869%) | 245/354 (69.209%) | 1499/2587 (57.944%) | 0.3520 | 0.1784 |
| 20 | 1104 | 49 | 39 | 99/235 (42.128%) | 137/186 (73.656%) | 635/1104 (57.518%) | 0.4793 | 0.2329 |
| 10 | 1768 | 97 | 125 | 206/391 (52.685%) | 188/294 (63.946%) | 1143/1768 (64.649%) | 0.5858 | 0.3698 |
| 5 | 2264 | 406 | 455 | 461/707 (65.205%) | 208/301 (69.103%) | 1472/2264 (65.018%) | 0.6427 | 0.3968 |
| 0 | 2388 | 184 | 217 | 351/563 (62.345%) | 232/379 (61.214%) | 1578/2388 (66.080%) | 0.6112 | 0.3991 |
| -5 | 3256 | 286 | 320 | 453/794 (57.053%) | 331/508 (65.157%) | 2003/3256 (61.517%) | 0.6248 | 0.3021 |

`On probe corrections` asks how often the refiner also corrects frames where Long corrects Short. `On probe harms` asks how often the refiner avoids Long's wrong flips. A positive Pearson and a small magnitude ratio indicate an underpowered refiner; a near-zero or negative Pearson indicates a mismatched refinement signal.
