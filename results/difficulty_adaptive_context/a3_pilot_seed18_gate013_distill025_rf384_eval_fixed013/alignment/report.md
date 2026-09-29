# Phase A3 Refiner-vs-Long Alignment

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed18_gate013_distill025_rf384_eval_fixed013\frame_predictions.npz`
- Margins use `logit(p)` differences from the shared Short output.
- This is a diagnostic only; it does not alter thresholds or checkpoints.

| Group | Selected | Probe net | Refiner net | On probe corrections: refiner correct | On probe harms: refiner correct | Residual same direction | Median residual magnitude ratio | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 14978 | 1468 | 1271 | 1946/3741 (52.018%) | 1544/2273 (67.928%) | 9087/14978 (60.669%) | 0.4974 | 0.2998 |
| seen | 7363 | 393 | 243 | 758/1591 (47.643%) | 757/1198 (63.189%) | 4474/7363 (60.763%) | 0.5476 | 0.3421 |
| unseen | 5028 | 727 | 941 | 881/1448 (60.843%) | 534/721 (74.064%) | 3204/5028 (63.723%) | 0.5226 | 0.2658 |
| clean | 2587 | 348 | 87 | 307/702 (43.732%) | 253/354 (71.469%) | 1409/2587 (54.465%) | 0.3385 | 0.1542 |
| 20 | 1104 | 49 | 26 | 95/235 (40.426%) | 131/186 (70.430%) | 629/1104 (56.975%) | 0.4665 | 0.2286 |
| 10 | 1768 | 97 | 131 | 216/391 (55.243%) | 192/294 (65.306%) | 1121/1768 (63.405%) | 0.5265 | 0.3694 |
| 5 | 2264 | 406 | 378 | 404/707 (57.143%) | 207/301 (68.771%) | 1413/2264 (62.412%) | 0.5298 | 0.3205 |
| 0 | 2388 | 184 | 250 | 346/563 (61.456%) | 247/379 (65.172%) | 1578/2388 (66.080%) | 0.5775 | 0.4094 |
| -5 | 3256 | 286 | 357 | 429/794 (54.030%) | 348/508 (68.504%) | 1987/3256 (61.026%) | 0.5768 | 0.2983 |

`On probe corrections` asks how often the refiner also corrects frames where Long corrects Short. `On probe harms` asks how often the refiner avoids Long's wrong flips. A positive Pearson and a small magnitude ratio indicate an underpowered refiner; a near-zero or negative Pearson indicates a mismatched refinement signal.
