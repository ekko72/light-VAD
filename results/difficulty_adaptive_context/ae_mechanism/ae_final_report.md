# A-v2 / AE Final Mechanism Report

- Protocol: `28218C73CC8CEDFE14DC9CDB5622B50C2D927627744AB9B9679FDFEE7F6B3842`
- Status: **A-v2 AE CLAIM FREEZE**
- A-v1: `COMPLETE — CONDITIONAL GO`
- A15: `FORBIDDEN`

## Scope

AE0 uses the already opened A14 development data. AE1-AE4 use the frozen A9 test manifest and frozen predictions; AE4 adds post-hoc checkpoint inference and the pre-registered five paired Short/RF384 replication runs. The A14 and A9 frame counts are different by design and are not pooled.

## AE1-AE2

- AE1 aggregate availability net: `0.00531680`
- AE1 aggregate gate utility: `0.00444854`
- AE1 aggregate activation rate: `0.05021120`
- AE2 non-monotone rate: `0.12808180`
- AE2 mean stable-sufficient span: `69.22404718`

## AE4

- Shared-Short checkpoint sign agreement: `0.98581964`
- Independent-Short seed23 sensitivity agreement: `0.96651358`
- Replication sign agreement: `0.97082702`

five independent initializations of Short/RF384 replicates.

The five training replicates are not independent encoders relative to the A14 checkpoints.

## AE5 Mechanism Map

- `VALUE_SCARCITY`: `6`
- `OBSERVABILITY_LIMITED`: `7`
- `ACTIONABILITY_LIMITED`: `0`
- `LONG_HORIZON_DEMAND`: `0`
- `TEMPORAL_SPAN_NOT_LIMITING`: `0`
- `MODEL_RELATIVE_VALUE`: `0`
- `REPLICATED_VALUE`: `0`

| Group | Classification | Availability | Observability | Gate utility |
|---|---:|---:|---:|---:|
| all | OBSERVABILITY_LIMITED | 0.00002329 | 0.03219297 | 0.00117546 |
| seen/-5 | OBSERVABILITY_LIMITED | 0.00203377 | 0.04403000 | 0.00264925 |
| seen/0 | VALUE_SCARCITY | -0.00176171 | 0.03636124 | -0.00020070 |
| seen/5 | VALUE_SCARCITY | -0.00550813 | 0.03461150 | -0.00286780 |
| seen/10 | VALUE_SCARCITY | -0.00597197 | 0.02680986 | -0.00310418 |
| seen/15 | VALUE_SCARCITY | -0.00382670 | 0.01956688 | -0.00206499 |
| seen/20 | VALUE_SCARCITY | -0.00348774 | 0.01604252 | -0.00182415 |
| unseen/-5 | OBSERVABILITY_LIMITED | 0.01709974 | 0.04905690 | 0.01346927 |
| unseen/0 | OBSERVABILITY_LIMITED | 0.01156930 | 0.04784040 | 0.01066838 |
| unseen/5 | OBSERVABILITY_LIMITED | 0.00478115 | 0.04060523 | 0.00538771 |
| unseen/10 | OBSERVABILITY_LIMITED | 0.00393374 | 0.02692376 | 0.00429054 |
| unseen/15 | OBSERVABILITY_LIMITED | 0.00081172 | 0.02035012 | 0.00191781 |
| unseen/20 | VALUE_SCARCITY | -0.00073144 | 0.01771305 | 0.00024976 |

## Claim Freeze

- `A-v2 AE CLAIM FREEZE`
- No router, gate, architecture, or model selection follows from this report.
- Any later method study requires a separately authorized candidate and a new untouched OOD set.
