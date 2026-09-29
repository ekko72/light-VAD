# E3 Boundary and Value-Robustness Audit

- E3 status: `ROBUST_VALUE_STRUCTURE`
- Baseline reproduced: `True`
- G1 decision taxonomy: `True`
- G2 proper scoring: `True`
- G3 boundary exclusion: `True`
- G4 label jitter: `True`

## Decision Layer

At threshold 0.50: P(R)=0.01334563, P(I)=0.08074757, P(H)=0.00802883, NetRefinability=0.00531680.

## Probabilistic Layer

- `v_log` mean=0.01015009, CI95=[0.00553873, 0.01522797], positive share=0.66286624.
- `v_brier` mean=0.00348136, CI95=[0.00177274, 0.00536487], positive share=0.66286624.

## Boundary Exclusion

| width ms | frames | mean v_log | P(R) | P(I) | NetRefinability |
|---:|---:|---:|---:|---:|---:|
| 0 | 295338 | 0.01076461 | 0.01317474 | 0.07527646 | 0.00541414 |
| 10 | 289420 | 0.01125088 | 0.01285329 | 0.06775275 | 0.00553866 |
| 20 | 283945 | 0.01138684 | 0.01244607 | 0.06165278 | 0.00549050 |
| 30 | 278603 | 0.01140622 | 0.01202428 | 0.05627003 | 0.00543785 |
| 50 | 268210 | 0.01117956 | 0.01119272 | 0.04813392 | 0.00530554 |
| 100 | 243752 | 0.01040518 | 0.00982146 | 0.03720585 | 0.00510765 |

## Label Jitter

| width ms | seed | mean v_log | P(R) | P(I) | NetRefinability |
|---:|---:|---:|---:|---:|---:|
| 10 | aggregate | 0.01002157 | 0.01333557 | 0.08126517 | 0.00529668 |
| 20 | aggregate | 0.00986704 | 0.01335836 | 0.08203688 | 0.00534227 |
| 30 | aggregate | 0.00960305 | 0.01332886 | 0.08292524 | 0.00528327 |
| 50 | aggregate | 0.00882185 | 0.01327389 | 0.08546765 | 0.00517332 |

## Decision versus Probabilistic Value

| group | frame share | mean v_log | total positive v_log |
|---|---:|---:|---:|
| R_decision_changing | 0.01334563 | 0.34360351 | 1367.88556774 |
| P_PLUS_NO_FLIP | 0.64952062 | 0.03929011 | 7612.53799975 |
| P_PLUS_ANY | 0.66286624 | 0.04541692 | 8980.42356749 |

## Claims

- Claim A: `SUPPORTED`
- Claim B: `TOO_STRONG`
- Claim C: `SUPPORTED`
- Claim D: `SUPPORTED`
- Claim E: `SUPPORTED`
- Claim F: `NOT_SUPPORTED`

E2 remains `INCONCLUSIVE_OR_INVALID`; E3 does not rescue or upgrade E2.

No training, checkpoint change, OOD access, or next experiment was performed.
