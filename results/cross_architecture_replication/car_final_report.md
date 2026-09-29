# A-v2 Cross-Architecture Replication Final Report

- Protocol: `C6878F55BB18005807CE205239903EFF45A76756EBD213235FE8BC7D20C6669B`
- Status: **STRONG_CROSS_ARCH_REPLICATION**
- Gates passed: `5/5`
- `NEXT_METHOD_SEARCH_AUTHORIZED=false`
- `NEW_FINAL_OOD_TOUCHED=false`

## Gate Results

| Gate | Pass | Result |
|---|---:|---|
| R1 | true | all five seed aggregate NetRefinability > 0 |
| R2 | true | aggregate P(R)>0 with source lower bound >0 and P(I)>P(R) |
| R3 | true | at least two transition/persistence/onset structures pass |
| R4 | true | 12-cell mean-v range meets point and source-support rules |
| R5 | true | full and LOSO means positive and no seed dominates |

## CAR1-CAR2

- Aggregate taxonomy: P(SS)=0.862187, P(R)=0.020581, P(I)=0.108005, P(H)=0.009226.
- Aggregate NetRefinability: 0.011355; source lower bound: 0.007941.
- Aggregate mean v: 0.020044.
- CAR2 non-monotone rate: 0.006902; long-history share: 0.801945.

## CAR3-CAR5

- CAR3 passing structures: `transition, onset`.
- CAR4 12-cell mean-v range: 0.041892; source lower bound: 0.023464.
- CAR5 full mean v: 0.020044; maximum seed positive share: 0.211263.

## Cross-Architecture Diagnostics

| Item | Classification | Tiny-GRU | MarbleNet |
|---|---:|---:|---:|
| A. Refinable prevalence | REPLICATED | 0.023871 | 0.014864 |
| B. Irreducible prevalence | REPLICATED | 0.108005 | 0.080748 |
| C. Harm prevalence | REPLICATED | 0.009226 | 0.008029 |
| D. Net refinability | REPLICATED | 0.011355 | 0.005317 |
| E. Value sparsity | NOT_REPLICATED | 0.538415 | 0.999935 |
| F. Transition enrichment | REPLICATED | 0.038196 | 0.084643 |
| G. Uncertainty-persistence enrichment | PARTIALLY_REPLICATED | 0.014118 | 0.133758 |
| H. Condition heterogeneity | REPLICATED | 0.034911 | 0.035970 |
| I. Long-history demand | REPLICATED | 0.801945 | 1.267786 |
| J. Non-monotonicity | NOT_REPLICATED | 0.006902 | 0.128082 |

## Claim Boundary

The result is limited to the frozen LibriVAD A9 test evaluation and the tested Tiny-GRU recurrent backbone.
No router, method, architecture, or hyperparameter selection follows from these outcomes.
The sealed NEW_FINAL_OOD split was not loaded, inspected, calibrated, or evaluated.
