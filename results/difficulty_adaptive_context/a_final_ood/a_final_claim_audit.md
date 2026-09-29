# A Final Claim Audit

Audit ID: `A-FINAL-CLAIM-AUDIT-v1`

Execution result: `INVALID_EXECUTION`

## Claim Decision

No new OOD claim can be asserted. No new claim can be rejected by this
execution either. The final holdout was not scored, so the primary endpoint,
confidence interval, subgroup safety checks, activation checks, and gates
have no result.

This is an evidence-availability failure, not a negative scientific result.

## Prior Official States

| Study | State preserved |
| --- | --- |
| A-v1 | `CONDITIONAL_GO` |
| A-v2 | `CONDITIONAL_GO` |
| CAR | `STRONG_CROSS_ARCH_REPLICATION` |
| E1 | `REMOTE_INFORMATION_CONDITIONAL` |
| E2 | `INCONCLUSIVE_OR_INVALID` |
| E3 | `ROBUST_VALUE_STRUCTURE` |
| E4 | `INCONCLUSIVE_OR_INVALID` |

E4 post-hoc evidence remains diagnostic only:

`DIAGNOSTIC_ONLY_NOT_AN_OFFICIAL_UPGRADE`

## Claim Classification

Supported by this execution: none.

Not supported by this execution: none. The primary hypothesis was not tested
because no eligible untouched dataset was available.

Conditional claims from this execution: none.

## Integrity Boundary

The frozen protocol and freeze manifest hashes matched their frozen records.
The OOD manifest was hash-bound before reading. It exposed no candidate
dataset and explicitly prohibited fabrication of a result.

Protocol deviations: `[]`

Training performed: `false`

Next experiment authorized: `false`
