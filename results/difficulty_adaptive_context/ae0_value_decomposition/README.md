# A-v2 / AE0 - OOD Value Failure Decomposition

- Protocol SHA256: `F45D202C77C0C83131E8C3BF94C8D02783E3B2A6EF4B31D3F3CADE6D02DD1870`
- Primary seed: `17`
- Analysis: post-hoc, no training, no router search, no A14 rerun
- A14 verdict carried forward: `CONDITIONAL_GO`

## Layer counts

- Availability-limited: `6/12`
- Observability-limited: `6/12`
- Actionability-limited: `0/12`
- Actionable: `0/12`
- Stop rule triggered: `False`

## Aggregate groups

- `all`: A_net=`0.00002329`, O=`0.032192965635270805`, U_gate=`0.00117546`, classification=`OBSERVABILITY_LIMITED`
- `seen`: A_net=`-0.00308708`, O=`0.02966706693617401`, U_gate=`-0.00123543`, classification=`AVAILABILITY_LIMITED`
- `unseen`: A_net=`0.00624403`, O=`0.0345685418731663`, U_gate=`0.00599725`, classification=`OBSERVABILITY_LIMITED`

## Cell decomposition

| Cell | A_net | O | U_gate | P | Classification |
|---|---:|---:|---:|---:|---|
| `seen/-5` | `0.00203377` | `0.04402999580582345` | `0.00264925` | `1.3026315789473686` | `OBSERVABILITY_LIMITED` |
| `seen/0` | `-0.00176171` | `0.03636123781847607` | `-0.00020070` | `None` | `AVAILABILITY_LIMITED` |
| `seen/5` | `-0.00550813` | `0.034611498575829194` | `-0.00286780` | `None` | `AVAILABILITY_LIMITED` |
| `seen/10` | `-0.00597197` | `0.026809855367358387` | `-0.00310418` | `None` | `AVAILABILITY_LIMITED` |
| `seen/15` | `-0.00382670` | `0.01956688159074083` | `-0.00206499` | `None` | `AVAILABILITY_LIMITED` |
| `seen/20` | `-0.00348774` | `0.01604252422907761` | `-0.00182415` | `None` | `AVAILABILITY_LIMITED` |
| `unseen/-5` | `0.01709974` | `0.04905689797974999` | `0.01346927` | `0.7876890975482526` | `OBSERVABILITY_LIMITED` |
| `unseen/0` | `0.01156930` | `0.047840400720391384` | `0.01066838` | `0.9221279876638396` | `OBSERVABILITY_LIMITED` |
| `unseen/5` | `0.00478115` | `0.04060522576666762` | `0.00538771` | `1.126865671641791` | `OBSERVABILITY_LIMITED` |
| `unseen/10` | `0.00393374` | `0.026923761468576585` | `0.00429054` | `1.090702947845805` | `OBSERVABILITY_LIMITED` |
| `unseen/15` | `0.00081172` | `0.020350124796490177` | `0.00191781` | `2.3626373626373627` | `OBSERVABILITY_LIMITED` |
| `unseen/20` | `-0.00073144` | `0.017713051302495664` | `0.00024976` | `None` | `AVAILABILITY_LIMITED` |

## Interpretation boundary

AE0 uses only seed 17 and the opened A14 OOD data. It does not authorize A15, router tuning, gate tuning, architecture changes, or a new confirmatory claim on this same OOD set.

