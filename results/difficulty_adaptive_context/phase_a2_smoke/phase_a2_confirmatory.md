# Phase A2: Statistical Confirmation

## Fixed Protocol

- Runs: seed17
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Primary calibration activation: 10.00%
- Speaker-cluster bootstrap repeats: 100
- Random-gate repeats: 100

The threshold is fixed on the calibration speakers. The final-test activation rate is reported as observed, not forced to the calibration budget.

## Per-Seed Fixed Gate

| Run | Threshold | Calibration activation | Test activation | Net / selected | Net / frame | Correction | Harm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed17 | 0.234942 | 10.000% | 10.317% | 9.001% | 0.929% | 26.917% | 17.917% |

## Seed-Averaged Test Result

- Test activation: 10.317% (95% CI [8.946, 12.075]%).
- Net utility per selected frame: 9.001% (95% CI [4.943, 14.033]%).
- Net utility per all frames: 0.929% (95% CI [0.497, 1.504]%).

## Unseen-Noise Check

- Activation: 13.407%.
- Net utility per selected frame: 22.163% (95% CI [12.894, 33.176]%).

## Confidence vs Random Activation

- Confidence gate: 9.001% net per selected frame.
- Speaker-matched random gate: 1.133% mean, 95th percentile 1.291%.
- One-sided Monte Carlo p-value: 0.00990.

## Activation-Budget Curve

| Calibration activation | Mean test activation | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: |
| 5.00% | 5.050% | 18.388% | 0.929% |
| 10.00% | 10.317% | 9.001% | 0.929% |
| 20.00% | 21.459% | 4.327% | 0.929% |

## Formal GO Criteria

- Overall status: **INCOMPLETE**.
- FAIL: At least the required number of seeds.
- PASS: Every seed has positive test net utility.
- PASS: Speaker-cluster 95% CI for net utility excludes zero.
- PASS: Confidence gating beats matched random activation.
- PASS: Unseen-noise speaker-cluster 95% CI excludes zero.
