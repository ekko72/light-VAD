# Phase A3 Refiner-vs-Long Alignment

- Frame predictions: `results\difficulty_adaptive_context\a3_pilot_seed19_gate013_distill025_rf384_eval_fixed013\frame_predictions.npz`
- Margins use `logit(p)` differences from the shared Short output.
- This is a diagnostic only; it does not alter thresholds or checkpoints.

| Group | Selected | Probe net | Refiner net | On probe corrections: refiner correct | On probe harms: refiner correct | Residual same direction | Median residual magnitude ratio | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 14978 | 1468 | 1313 | 1992/3741 (53.248%) | 1558/2273 (68.544%) | 9100/14978 (60.756%) | 0.5286 | 0.2581 |
| seen | 7363 | 393 | 263 | 741/1591 (46.574%) | 766/1198 (63.940%) | 4466/7363 (60.655%) | 0.5027 | 0.3551 |
| unseen | 5028 | 727 | 986 | 950/1448 (65.608%) | 531/721 (73.648%) | 3255/5028 (64.737%) | 0.6690 | 0.2146 |
| clean | 2587 | 348 | 64 | 301/702 (42.877%) | 261/354 (73.729%) | 1379/2587 (53.305%) | 0.3740 | 0.0756 |
| 20 | 1104 | 49 | 61 | 105/235 (44.681%) | 138/186 (74.194%) | 640/1104 (57.971%) | 0.5355 | 0.2453 |
| 10 | 1768 | 97 | 120 | 196/391 (50.128%) | 190/294 (64.626%) | 1146/1768 (64.819%) | 0.5055 | 0.3761 |
| 5 | 2264 | 406 | 418 | 443/707 (62.659%) | 214/301 (71.096%) | 1424/2264 (62.898%) | 0.6501 | 0.3270 |
| 0 | 2388 | 184 | 247 | 332/563 (58.970%) | 258/379 (68.074%) | 1516/2388 (63.484%) | 0.5415 | 0.3245 |
| -5 | 3256 | 286 | 364 | 465/794 (58.564%) | 335/508 (65.945%) | 2060/3256 (63.268%) | 0.6332 | 0.2777 |

`On probe corrections` asks how often the refiner also corrects frames where Long corrects Short. `On probe harms` asks how often the refiner avoids Long's wrong flips. A positive Pearson and a small magnitude ratio indicate an underpowered refiner; a near-zero or negative Pearson indicates a mismatched refinement signal.
