# CAR Claim Freeze

## OBSERVATIONS

- Final frozen status: `STRONG_CROSS_ARCH_REPLICATION`.
- Gates passed: `5/5`.
- Aggregate P(R), P(I), P(H), and NetRefinability: `0.020581`, `0.108005`, `0.009226`, `0.011355`.
- Aggregate non-monotone rate and long-history share: `0.006902`, `0.801945`.
- Passing CAR3 structures: `transition, onset`.
- CAR4 condition-cell mean-v range: `0.041892`.
- CAR5 maximum seed positive-frame share: `0.211263`.

## SUPPORTED INTERPRETATIONS

- The sparse and condition-dependent structure of temporal refinement value observed with the convolutional backbone was reproduced in the tested recurrent backbone.
- The comparison is restricted to the frozen A9 evaluation set and the five preregistered Tiny-GRU seeds.

## UNSUPPORTED CLAIMS

- Universal across neural VAD architectures.
- Architecture-independent temporal value.
- Tiny GRU proves the MarbleNet mechanism.
- Long context universally improves VAD.
- Refinability-aware routing is superior.
- Deployment-level generalization.
- Any claim based on the sealed NEW_FINAL_OOD split or on post-outcome method, architecture, or seed selection.

- `NEXT_METHOD_SEARCH_AUTHORIZED=false`
- `NEW_FINAL_OOD_TOUCHED=false`
