# A-ANALYSIS-v1 Multi-Seed Summary

## Scope

This summary uses the frozen LibriVAD official test split and the same
553,532-frame evaluation population for seeds 17, 23, and 41. It packages
already-produced frame predictions only. No model was trained, no seed was
selected after seeing results, and no threshold or routing rule was changed.

## Data Sources

| Seed | Prediction source |
| ---: | --- |
| 17 | `results/difficulty_adaptive_context/formal_eval_ext40/frame_predictions.npz` |
| 23 | `results/difficulty_adaptive_context/seed23_eval_ext40/frame_predictions.npz` |
| 41 | `results/difficulty_adaptive_context/seed41_eval_ext40/frame_predictions.npz` |

Each file contains 553,532 frames, 470,043 speech frames, 83,489 silence
frames, and 197 source clusters. The actual seed assignment is taken from the
evaluation directory and checkpoint path. The `protocol.seed` field in the
seed-23 and seed-41 result JSON files is inconsistent with those paths and was
not used for aggregation.

## Metric Rules

- F1 uses a fixed threshold of 0.5 and is computed from pooled true positives,
  false positives, and false negatives. Per-condition F1 values are not
  averaged.
- AUC is computed from all frames in the stated population.
- Proper loss is binary log-loss with probabilities clipped to
  `[1e-12, 1 - 1e-12]` before scoring.
- Mean and standard deviation are computed across the three seeds.
- Standard deviation is the sample standard deviation (`ddof=1`).
- No best-seed selection is performed.

## Results

| Model | Seed | F1 | AUC | Proper loss |
| --- | ---: | ---: | ---: | ---: |
| Short | 17 | 0.9476 | 0.9449 | 0.2094 |
| Short | 23 | 0.9527 | 0.9475 | 0.2001 |
| Short | 41 | 0.9484 | 0.9402 | 0.2131 |
| Short | mean | 0.9496 | 0.9442 | 0.2075 |
| Short | std | 0.0027 | 0.0037 | 0.0067 |
| Long-RF | 17 | 0.9528 | 0.9500 | 0.1967 |
| Long-RF | 23 | 0.9552 | 0.9519 | 0.1904 |
| Long-RF | 41 | 0.9522 | 0.9484 | 0.1985 |
| Long-RF | mean | 0.9534 | 0.9501 | 0.1952 |
| Long-RF | std | 0.0016 | 0.0018 | 0.0043 |
| Adaptive | 17 | not_available | not_available | not_available |
| Adaptive | 23 | not_available | not_available | not_available |
| Adaptive | 41 | not_available | not_available | not_available |
| Adaptive | mean | not_available | not_available | not_available |
| Adaptive | std | not_available | not_available | not_available |

The CSV file contains full-precision values; this report rounds only for
display. The corresponding summary rows in the CSV use `row_type=mean` and
`row_type=std`.

## Adaptive Coverage

No Adaptive result was merged into the unified three-seed summary. A seed-17
Adaptive prediction file exists in
`results/difficulty_adaptive_context/a3_pilot_seed17_gate013_distill025_rf384_eval_fixed013/`,
but it uses a different evaluated population: 298,300 evaluated frames and 96
source keys. That population is not the 553,532-frame pooled population used
by the three-seed Short and Long-RF summary. Combining it with the Short and
Long-RF rows would mix evaluation populations, so Adaptive is reported as
`not_available` in the unified table.

## Missing and Processing Rules

- All unavailable cells are written as `not_available`.
- The Long-RF row is retained for every available seed.
- No values are inferred from reports with a different population or metric
  definition.
- The summary does not use internal experiment labels or select a preferred
  seed.
