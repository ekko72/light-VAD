# Context Robustness Checks

## Protocol

- Predictions: `results\difficulty_adaptive_context\formal_eval_ext40\frame_predictions.npz`
- Utterance clusters: 197
- Frames: 553532
- Bootstrap repeats: 1000
- Confidence thresholds come from the original global A2 buckets.

## Very Hard Frames by SNR

| SNR | Frames | Short error | Long error | Error reduction | 95% CI | F1 delta | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 20118 | 26.772% | 21.866% | 4.906% | [3.079, 6.902]% | 0.0703 | [0.0504, 0.0913] |
| 20 | 10211 | 28.469% | 27.735% | 0.735% | [-0.875, 2.306]% | 0.0031 | [-0.0203, 0.0244] |
| 10 | 13843 | 30.781% | 27.472% | 3.309% | [1.202, 5.452]% | 0.0309 | [0.0023, 0.0538] |
| 5 | 14687 | 37.148% | 29.073% | 8.075% | [2.100, 14.483]% | 0.0954 | [0.0156, 0.1740] |
| 0 | 17767 | 35.054% | 29.392% | 5.662% | [1.141, 10.695]% | 0.0519 | [0.0015, 0.1051] |
| -5 | 20389 | 36.701% | 30.168% | 6.533% | [2.427, 10.890]% | 0.0657 | [0.0223, 0.1098] |

## Utterance-Clustered Bootstrap

| Check | Point estimate | 95% CI |
| --- | ---: | ---: |
| All valid frames | 0.855% | [0.495, 1.251]% |
| Very hard frames | 4.709% | [3.165, 6.352]% |
| Within 200 ms of a boundary | 1.047% | [0.609, 1.537]% |
| Unseen minus seen error reduction | 2.452% | [1.174, 3.820]% |
