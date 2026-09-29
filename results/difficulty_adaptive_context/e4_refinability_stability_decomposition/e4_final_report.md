# E4 Refinability Stability Decomposition

- E4 status: `INCONCLUSIVE_OR_INVALID`
- Baseline reproduced: `True`
- Frame alignment verified: `True`
- Frames: `298300`
- Sources: `96`
- Replicates: `5`

## Primary Stability Endpoints

P(STABLE_R)=0.01334563, P(CORE_R)=0.00538719, P(STABLE_R | K_R>=1)=0.24093688, P(CORE_R | K_R>=1)=0.09725837.

Observed P(K_R>=3)=0.01334563 versus frozen null=0.00087898; excess=0.01246664, source-cluster CI95=[0.00965888, 0.01542543].
Observed P(K_R>=4)=0.00538719 versus frozen null=0.00005833; excess=0.00532886, source-cluster CI95=[0.00405698, 0.00659483].

## Gates And Claims

- G1 stable core: `True`
- G2 stable core value: `True`
- G3 structured instability: `True`
- Claim A: `NOT_SUPPORTED`
- Claim B: `CONDITIONAL`
- Claim C: `CONDITIONAL`
- Claim D: `UNRESOLVED`
- Claim E: `UNRESOLVED`

## Stability Categories

| category | frames | P(all) | P(given ever R) | mean v_log |
|---|---:|---:|---:|---:|
| NEVER_R | 281777 | 0.94460945 | 17.05362222 | 0.00290132 |
| UNSTABLE_R | 12542 | 0.04204492 | 0.75906312 | 0.25310945 |
| STABLE_R | 3981 | 0.01334563 | 0.24093688 | 0.49913892 |
| CORE_R | 1607 | 0.00538719 | 0.09725837 | 0.56449485 |

## K_R Distribution

| K_R | frames | observed P | independence expected P |
|---:|---:|---:|---:|
| 0 | 281777 | 0.94460945 | 0.90119013 |
| 1 | 8317 | 0.02788133 | 0.09479516 |
| 2 | 4225 | 0.01416359 | 0.00393364 |
| 3 | 2374 | 0.00795843 | 0.00008027 |
| 4 | 1227 | 0.00411331 | 0.00000080 |
| 5 | 380 | 0.00127389 | 0.00000000 |

## Non-R Fate Among Frames With K_R 1-4

| K_R category | frames | non-R assignments | P(SS) | P(I) | P(H) |
|---|---:|---:|---:|---:|---:|
| 1 | 8317 | 33268 | 0.55067933 | 0.42698689 | 0.02233377 |
| 2 | 4225 | 12675 | 0.51747535 | 0.47045365 | 0.01207101 |
| 3 | 2374 | 4748 | 0.49684078 | 0.49494524 | 0.00821398 |
| 4 | 1227 | 1227 | 0.49551752 | 0.49633252 | 0.00814996 |
| 1-4 | 16143 | 51918 | 0.53634578 | 0.44545244 | 0.01820178 |

## Interpretation

The frozen inputs or required statistical validity checks did not pass, so no E4 mechanism interpretation is made.

E3 remains `ROBUST_VALUE_STRUCTURE`; E4 studies only the stability of decision-changing R membership and does not reinterpret probabilistic-value frames as R.

Claim E remains hypothesis-level. E4 cannot establish that R instability causes M2's limited OOD gain.

No training, checkpoint change, OOD access, or next experiment was performed.
