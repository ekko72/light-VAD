# Phase A3 Refiner-vs-Long Alignment

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed17_gate013_distill025_eval_fixed013\frame_predictions.npz`
- Margins use `logit(p)` differences from the shared Short output.
- This is a diagnostic only; it does not alter thresholds or checkpoints.

| Group | Selected | Probe net | Refiner net | On probe corrections: refiner correct | On probe harms: refiner correct | Residual same direction | Median residual magnitude ratio | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 14978 | 1468 | 373 | 1369/3741 (36.594%) | 1586/2273 (69.776%) | 8604/14978 (57.444%) | 0.3878 | 0.2024 |
| seen | 7363 | 393 | 177 | 599/1591 (37.649%) | 850/1198 (70.952%) | 4257/7363 (57.816%) | 0.4182 | 0.2736 |
| unseen | 5028 | 727 | 3 | 432/1448 (29.834%) | 478/721 (66.297%) | 2815/5028 (55.986%) | 0.3624 | 0.1209 |
| clean | 2587 | 348 | 193 | 338/702 (48.148%) | 258/354 (72.881%) | 1532/2587 (59.219%) | 0.3624 | 0.1612 |
| 20 | 1104 | 49 | 32 | 86/235 (36.596%) | 144/186 (77.419%) | 604/1104 (54.710%) | 0.4114 | 0.2169 |
| 10 | 1768 | 97 | 82 | 160/391 (40.921%) | 199/294 (67.687%) | 1103/1768 (62.387%) | 0.4019 | 0.3523 |
| 5 | 2264 | 406 | -17 | 192/707 (27.157%) | 197/301 (65.449%) | 1266/2264 (55.919%) | 0.3532 | 0.1572 |
| 0 | 2388 | 184 | -10 | 159/563 (28.242%) | 260/379 (68.602%) | 1343/2388 (56.240%) | 0.3477 | 0.1584 |
| -5 | 3256 | 286 | 46 | 298/794 (37.531%) | 341/508 (67.126%) | 1849/3256 (56.787%) | 0.4544 | 0.2228 |

`On probe corrections` asks how often the refiner also corrects frames where Long corrects Short. `On probe harms` asks how often the refiner avoids Long's wrong flips. A positive Pearson and a small magnitude ratio indicate an underpowered refiner; a near-zero or negative Pearson indicates a mismatched refinement signal.
