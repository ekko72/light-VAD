# A14 Final OOD Confirmation Report

- Protocol SHA256: `5F03D4CB8B12EC23B98ACBB7CC09FFFB7584FB1FC4D8D2366F7FA2945CD4F1CF`
- Unique candidate: `ShortShort Adaptive RF384Adaptive RF384`
- Router: `abs(short_score - 0.5) <= 0.13`
- Final verdict: **CONDITIONAL_GO**
- Claim freeze: `A14 -> A CLAIM FREEZE`
- A15: `FORBIDDEN`

## Checkpoint accounting

four refinement checkpoints were evaluated, of which three share the same frozen Short encoder and one uses an independently trained Short encoder.

## Gate 1: OOD accuracy utility

- Aggregate utility: `0.00117546`
- Source-cluster bootstrap 95% CI lower: `-0.00054213`
- Delta F1 (Adaptive - Short): `0.001527188577310823`
- Status: `CONDITIONAL`

## Gate 2: OOD compute behavior

- `r_OOD`: `6.705300%`
- Development reference: `5.050000%`
- Warning boundary: `15.000000%`
- Failure boundary: `20.000000%`
- Status: `PASS`

## Gate 3: OOD failure structure

- Positive-cell fraction: `58.333%` (7/12)
- Worst-cell utility: `-0.00310418` (`seen/10 dB`)
- Activation maximum: `9.763886%` (`seen/-5 dB`)
- Catastrophic domain regression: `[]`
- Taxonomy counts: `{'F1_VALUE_SCARCITY': 5, 'F2_RANKING_FAILURE': 0, 'F3_BUDGET_CALIBRATION_DRIFT': 0, 'NO_FAILURE': 7}`

## Seed sensitivity

- seed17: `0.00117546`
- seed18: `-0.00106198`
- seed19: `-0.00086624`
- seed23: `-0.00392978`
- Mean seeds 17/18/19: `-0.00025092`
- Seed23: `-0.00392978`
- Seed23 direction consistent: `True`
- Seed23 positive direction: `False`

## Interpretation boundary

This is composition-level untouched OOD confirmation, not new-speaker or new-noise OOD. The primary uncertainty unit is the source cluster; speaker-cluster intervals are sensitivity checks. The result does not authorize A15, router tuning, gate tuning, architecture changes, or stabilizer selection on Final OOD.

