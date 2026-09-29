# AX2 Correction-versus-Harm Separability

## Frozen Setup

- Calibration speakers/frames: 20/255,232
- Test speakers/frames: 20/298,300
- Primary predictor: **X3_logistic**.
- Correction target: signed value `+1` versus `{0, -1}`.
- Harm target: signed value `-1` versus `{0, +1}`.
- All model fitting uses calibration speakers only.

## Point Predictability

| Target | Model | Prevalence | AUROC | AUPRC | Normalized AUPRC lift |
| --- | --- | ---: | ---: | ---: | ---: |
| correction | X0_logistic | 1.335% | 0.973819 | 0.245677 | 0.235474 |
| correction | X0_mlp | 1.335% | 0.973819 | 0.245674 | 0.235470 |
| correction | X1_logistic | 1.335% | 0.974418 | 0.255007 | 0.244930 |
| correction | X1_mlp | 1.335% | 0.975203 | 0.267735 | 0.257830 |
| correction | X2_logistic | 1.335% | 0.978491 | 0.312001 | 0.302695 |
| correction | X2_mlp | 1.335% | 0.980548 | 0.338530 | 0.329583 |
| correction | X3_logistic | 1.335% | 0.977216 | 0.303290 | 0.293866 |
| correction | X3_mlp | 1.335% | 0.973359 | 0.281351 | 0.271630 |
| harm | X0_logistic | 0.803% | 0.967880 | 0.139549 | 0.132584 |
| harm | X0_mlp | 0.803% | 0.967944 | 0.141302 | 0.134352 |
| harm | X1_logistic | 0.803% | 0.968668 | 0.130457 | 0.123419 |
| harm | X1_mlp | 0.803% | 0.970332 | 0.144904 | 0.137983 |
| harm | X2_logistic | 0.803% | 0.974298 | 0.187472 | 0.180896 |
| harm | X2_mlp | 0.803% | 0.977419 | 0.215181 | 0.208828 |
| harm | X3_logistic | 0.803% | 0.973427 | 0.189340 | 0.182779 |
| harm | X3_mlp | 0.803% | 0.966409 | 0.194866 | 0.188349 |

## Primary Speaker-Cluster Bootstrap

| Target | AUROC 95% CI | Normalized AUPRC lift 95% CI |
| --- | ---: | ---: |
| correction | [0.973368, 0.980632] | [0.243043, 0.346466] |
| harm | [0.969560, 0.976809] | [0.146129, 0.225209] |

## Paired Differences

| Comparison | AUROC difference 95% CI | Normalized AUPRC lift difference 95% CI |
| --- | ---: | ---: |
| correction | [0.000003, 0.007304] | [0.011556, 0.108681] |
| harm | [0.000143, 0.012064] | [0.011009, 0.096130] |
| correction_vs_harm | [0.000445, 0.007211] | [0.034557, 0.182706] |

## AX2 Assessment

- Status: **BIDIRECTIONAL_SEPARABILITY**.
- Interpretation: Both correction and harm contain robust ranking signal. AX2 is diagnostic and does not authorize a new router.
