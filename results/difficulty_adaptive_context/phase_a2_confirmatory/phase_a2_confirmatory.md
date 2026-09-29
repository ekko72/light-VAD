# Phase A2: Statistical Confirmation

## Fixed Protocol

- Runs: seed17, seed23, seed41
- Calibration speakers: 20
- Final-test speakers: 20
- Speaker split seed: 20260917
- Primary calibration activation: 10.00%
- Speaker-cluster bootstrap repeats: 5000
- Random-gate repeats: 1000

The threshold is fixed on the calibration speakers. The final-test activation rate is reported as observed, not forced to the calibration budget.

## Per-Seed Fixed Gate

| Run | Threshold | Calibration activation | Test activation | Net / selected | Net / frame | Correction | Harm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed17 | 0.234942 | 10.000% | 10.317% | 9.001% | 0.929% | 26.917% | 17.917% |
| seed23 | 0.240908 | 10.000% | 10.959% | 4.227% | 0.463% | 22.743% | 18.516% |
| seed41 | 0.240322 | 10.000% | 10.985% | 6.131% | 0.673% | 24.936% | 18.805% |

## Seed-Averaged Test Result

- Test activation: 10.754% (95% CI [9.225, 12.352]%).
- Net utility per selected frame: 6.453% (95% CI [3.528, 9.167]%).
- Net utility per all frames: 0.688% (95% CI [0.339, 1.087]%).

## Unseen-Noise Check

- Activation: 14.050%.
- Net utility per selected frame: 16.494% (95% CI [8.826, 23.692]%).

## Confidence vs Random Activation

- Confidence gate: 6.453% net per selected frame.
- Speaker-matched random gate: 0.908% mean, 95th percentile 1.021%.
- One-sided Monte Carlo p-value: 0.00100.

## Activation-Budget Curve

| Calibration activation | Mean test activation | Net / selected | Net / frame |
| ---: | ---: | ---: | ---: |
| 5.00% | 5.345% | 13.025% | 0.688% |
| 10.00% | 10.754% | 6.453% | 0.688% |
| 20.00% | 21.850% | 3.160% | 0.688% |

## Formal GO Criteria

- Overall status: **FORMAL_GO**.
- PASS: At least the required number of seeds.
- PASS: Every seed has positive test net utility.
- PASS: Speaker-cluster 95% CI for net utility excludes zero.
- PASS: Confidence gating beats matched random activation.
- PASS: Unseen-noise speaker-cluster 95% CI excludes zero.
