# A-ANALYSIS-v1 Main Table

## Scope

The main table reports one detection result per model on the AVA-Speech official
test split, using the `overall` condition. The table is a cross-model packaging
of already-frozen results; no model was retrained, no threshold was changed, and
no routing decision was tuned.

## Data Sources

- Detection metrics and cost metadata: `results/difficulty_adaptive_context/ava_speech_v1/ava_speech_main_results.csv`.
- Baseline inventory check: `results/difficulty_adaptive_context/e3_baseline_comparison/baseline_comparison.csv`.
  This file contains a router utility comparison, not detector F1, AUC, or
  proper-loss measurements, so it was not merged into the detection table.
- LibriVAD main-result files exist under `results/difficulty_adaptive_context/`
  but use different evaluation populations and are not mixed into the AVA-Speech
  rows.

All unavailable cells are written as `not_available`. No missing value was
replaced by an estimate. The Long-RF row is retained.

## Detection Results

| Model | F1 | F1 95% CI | AUC | AUC 95% CI | Proper loss | Proper-loss 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 0.7414 | [0.7220, 0.7585] | 0.8009 | [0.7563, 0.8586] | 0.7795 | [0.7554, 0.8035] |
| Long-RF | 0.7435 | [0.7166, 0.7683] | 0.7874 | [0.7631, 0.8414] | 0.7562 | [0.7074, 0.8051] |
| Adaptive | 0.7366 | [0.7074, 0.7622] | 0.7998 | [0.7560, 0.8556] | 0.7924 | [0.7752, 0.8096] |
| AlwaysRefine | 0.7322 | [0.6986, 0.7618] | 0.7950 | [0.7598, 0.8390] | 0.8525 | [0.8489, 0.8561] |
| Shallow GBDT | 0.8114 | [0.7790, 0.8343] | 0.8810 | [0.8695, 0.8877] | 0.4291 | [0.4235, 0.4348] |
| WebRTC VAD | 0.7139 | [0.6884, 0.7361] | 0.6216 | [0.5912, 0.6545] | 10.0039 | [9.8842, 10.1237] |
| Silero VAD | 0.8455 | [0.8167, 0.8658] | 0.9347 | [0.9302, 0.9377] | 0.4715 | [0.4376, 0.5053] |
| NeMo MarbleNet | 0.7943 | [0.7166, 0.8457] | 0.9121 | [0.8987, 0.9161] | 0.6003 | [0.5377, 0.6630] |

The intervals are the source-cluster bootstrap intervals already present in the
AVA-Speech result file. The official test split has only two source clusters, so
source-level uncertainty is necessarily coarse.

## Cost and Output Metadata

| Model | MACs/frame | Streaming cache (bytes) | Latency (ms/frame) | Causal output |
| --- | ---: | ---: | ---: | --- |
| Short | 45312 | 34304 | 0.0224 | True |
| Long-RF | 45312 | 103936 | 0.0213 | True |
| Adaptive | 45634.60684643845 | 230912 | 0.0339 | True |
| AlwaysRefine | 48768 | 230912 | 0.0300 | True |
| Shallow GBDT | not_available | not_available | 0.0507 | True |
| WebRTC VAD | not_available | not_available | not_available | True |
| Silero VAD | not_available | not_available | 0.2479 | True |
| NeMo MarbleNet | not_available | not_available | 0.0342 | True |

`NeMo MarbleNet` is included as a supplement to the original Stage 2 model
inventory. It is not part of the shorter priority-2 list, but it is available
in the frozen AVA-Speech result file.

## Missing Items

- WebRTC VAD: MACs, streaming cache, and latency are `not_available`.
- Silero VAD: MACs and streaming cache are `not_available`.
- NeMo MarbleNet: MACs and streaming cache are `not_available`.
- Shallow GBDT: MACs and streaming cache are `not_available`.
- The E3 baseline file does not provide the missing detector-cost fields and
  was not used to fill them.

## Processing Rules

- Dataset: AVA-Speech official test split.
- Condition: `overall`.
- Point estimates and confidence intervals are copied from the frozen AVA
  result file.
- Missing values are explicitly represented as `not_available`.
- No internal experiment labels are used in this report.
