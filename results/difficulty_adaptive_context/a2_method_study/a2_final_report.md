# A-v2 Method Study

## Frozen Protocol

- Status: **CONDITIONAL_GO**
- Primary method: **M2**
- Primary budget: **5%**
- Delta utility vs uncertainty: **0.00051674**
- Source-cluster 95% CI: **[-0.00017398, 0.00132061]**
- Seed-positive count: **5/5**
- Unseen delta utility: **0.00244032**
- Primary activation: **5.731%**
- Maximum activation: **10.862%**

## Primary Gates

| Gate | Passed |
| --- | --- |
| G1 | no |
| G2 | yes |
| G3 | yes |
| G4 | yes |
| G5 | yes |

## Budget Results

| Method | Budget | Activation | Utility |
| --- | ---: | ---: | ---: |
| M0 | 1% | 1.123% | 0.00168116 |
| M0 | 2% | 2.193% | 0.00279604 |
| M0 | 5% | 5.307% | 0.00513197 |
| M0 | 10% | 10.542% | 0.00598140 |
| M1 | 1% | 1.254% | 0.00272525 |
| M1 | 2% | 2.448% | 0.00399232 |
| M1 | 5% | 5.738% | 0.00520630 |
| M1 | 10% | 11.122% | 0.00539211 |
| M2 | 1% | 1.331% | 0.00271641 |
| M2 | 2% | 2.493% | 0.00390561 |
| M2 | 5% | 5.731% | 0.00564871 |
| M2 | 10% | 10.862% | 0.00598848 |

## Compute Diagnostic

| Method | Budget | Estimated ms/frame | AlwaysRefine advantage |
| --- | ---: | ---: | ---: |
| M0 | 1% | 2.23870519 | 0.91715726 ms (29.06%) |
| M0 | 2% | 2.24559346 | 0.91026900 ms (28.84%) |
| M0 | 5% | 2.26564861 | 0.89021384 ms (28.21%) |
| M0 | 10% | 2.29935495 | 0.85650750 ms (27.14%) |
| M1 | 1% | 1.86028121 | 1.29558124 ms (41.05%) |
| M1 | 2% | 1.86797054 | 1.28789191 ms (40.81%) |
| M1 | 5% | 1.88915266 | 1.26670979 ms (40.14%) |
| M1 | 10% | 1.92382415 | 1.23203830 ms (39.04%) |
| M2 | 1% | 1.86085156 | 1.29501089 ms (41.04%) |
| M2 | 2% | 1.86833236 | 1.28753009 ms (40.80%) |
| M2 | 5% | 1.88918175 | 1.26668070 ms (40.14%) |
| M2 | 10% | 1.92222034 | 1.23364211 ms (39.09%) |

## Independent-Backbone Sensitivity

- Primary method replicated: `M2`
- Positive independent replicates: `4/5`
- Mean delta utility: `0.00031677`

## Interpretation Boundary

- OBSERVATION: The reported utility uses actual Short-to-RF384 selection outcomes, not classifier accuracy alone.
- SUPPORTED INTERPRETATION: Differences are limited to the frozen TRAIN/DEV/INTERNAL_TEST split and the tested observables.
- UNSUPPORTED CLAIM: This experiment does not establish deployment-wide generalization or justify further router search.

`NEW_FINAL_OOD_TOUCHED=false`

`NEXT_SEARCH_AUTHORIZED=false`

## Recovery Audit

- The primary checkpoints were trained once and reused after a result-assembly domain-index error.
- No primary model was retrained; no model, threshold, feature, loss, budget, or seed search was performed.
- See `a2_recovery_audit.json` for the exact deviation record.
