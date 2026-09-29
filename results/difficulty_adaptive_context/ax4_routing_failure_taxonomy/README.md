# AX4 Routing Failure Taxonomy

## Frozen Scope

- Protocol: `A-AX4-v1`.
- Raw gate threshold: `0.130000`.
- Test frames: 298,300.
- Source clusters: 197.
- Speaker clusters: 40.
- The threshold and cells are reused from frozen A12 definitions.
- The source-rank stabilizer is not evaluated or retuned.

## Cell Taxonomy

| Section | Domain | Cell | Activation | Population P(+1) | Population P(-1) | Population E[v] | Selected P(+1) | Selected P(-1) | Selected E[v] | Class |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ssr_sweep | all | SSR 10% | 12.778% | 3.514% | 1.542% | 0.019722 | 22.609% | 7.717% | 0.148913 | NO_FAILURE |
| ssr_sweep | all | SSR 30% | 9.736% | 2.792% | 1.611% | 0.011806 | 24.536% | 10.128% | 0.144080 | NO_FAILURE |
| ssr_sweep | all | SSR 50% | 8.931% | 2.597% | 1.250% | 0.013472 | 24.572% | 9.331% | 0.152411 | NO_FAILURE |
| ssr_sweep | all | SSR 70% | 6.667% | 1.833% | 1.042% | 0.007917 | 21.458% | 10.000% | 0.114583 | NO_FAILURE |
| ssr_sweep | all | SSR 90% | 4.736% | 1.542% | 0.625% | 0.009167 | 26.686% | 9.677% | 0.170088 | NO_FAILURE |
| acoustic_cell | all | SNR 20 dB | 2.922% | 0.543% | 0.405% | 0.001376 | 15.399% | 11.866% | 0.035326 | NO_FAILURE |
| acoustic_cell | all | SNR 10 dB | 4.485% | 0.936% | 0.622% | 0.003146 | 17.817% | 10.747% | 0.070701 | NO_FAILURE |
| acoustic_cell | all | SNR 5 dB | 6.378% | 2.535% | 0.887% | 0.016481 | 29.373% | 9.276% | 0.200972 | NO_FAILURE |
| acoustic_cell | all | SNR 0 dB | 6.197% | 1.811% | 0.955% | 0.008564 | 21.524% | 12.437% | 0.090871 | NO_FAILURE |
| acoustic_cell | all | SNR -5 dB | 7.282% | 2.181% | 1.320% | 0.008611 | 21.898% | 12.070% | 0.098280 | NO_FAILURE |
| ssr_sweep | seen | SSR 10% | 14.208% | 4.396% | 0.917% | 0.034792 | 25.513% | 5.425% | 0.200880 | NO_FAILURE |
| ssr_sweep | seen | SSR 30% | 12.042% | 3.708% | 0.875% | 0.028333 | 25.952% | 6.574% | 0.193772 | NO_FAILURE |
| ssr_sweep | seen | SSR 50% | 8.875% | 2.917% | 0.958% | 0.019583 | 26.995% | 8.451% | 0.185446 | NO_FAILURE |
| ssr_sweep | seen | SSR 70% | 6.646% | 1.917% | 0.979% | 0.009375 | 24.138% | 12.853% | 0.112853 | NO_FAILURE |
| ssr_sweep | seen | SSR 90% | 3.604% | 0.750% | 0.771% | -0.000208 | 16.763% | 17.919% | -0.011561 | F1_VALUE_SCARCITY |
| acoustic_cell | seen | SNR 20 dB | 2.804% | 0.502% | 0.404% | 0.000975 | 14.286% | 12.547% | 0.017391 | NO_FAILURE |
| acoustic_cell | seen | SNR 10 dB | 4.623% | 0.987% | 0.657% | 0.003300 | 18.140% | 11.067% | 0.070727 | NO_FAILURE |
| acoustic_cell | seen | SNR 5 dB | 4.900% | 1.171% | 0.847% | 0.003241 | 19.743% | 13.919% | 0.058243 | NO_FAILURE |
| acoustic_cell | seen | SNR 0 dB | 5.910% | 1.323% | 1.026% | 0.002978 | 17.915% | 14.556% | 0.033590 | NO_FAILURE |
| acoustic_cell | seen | SNR -5 dB | 5.533% | 1.007% | 1.109% | -0.001024 | 15.484% | 16.410% | -0.009254 | F1_VALUE_SCARCITY |
| ssr_sweep | unseen | SSR 10% | 8.875% | 1.667% | 2.750% | -0.010833 | 17.840% | 14.554% | 0.032864 | NO_FAILURE |
| ssr_sweep | unseen | SSR 30% | 8.583% | 1.875% | 2.167% | -0.002917 | 17.476% | 11.165% | 0.063107 | NO_FAILURE |
| ssr_sweep | unseen | SSR 50% | 7.833% | 1.875% | 1.583% | 0.002917 | 18.617% | 10.638% | 0.079787 | NO_FAILURE |
| ssr_sweep | unseen | SSR 70% | 6.208% | 2.208% | 1.000% | 0.012083 | 26.174% | 8.054% | 0.181208 | NO_FAILURE |
| ssr_sweep | unseen | SSR 90% | 5.250% | 1.833% | 0.458% | 0.013750 | 22.222% | 5.556% | 0.166667 | NO_FAILURE |
| acoustic_cell | unseen | SNR 20 dB | 3.294% | 0.672% | 0.408% | 0.002644 | 18.395% | 10.033% | 0.083612 | NO_FAILURE |
| acoustic_cell | unseen | SNR 10 dB | 3.771% | 0.673% | 0.438% | 0.002347 | 15.768% | 8.714% | 0.070539 | NO_FAILURE |
| acoustic_cell | unseen | SNR 5 dB | 8.440% | 4.439% | 0.944% | 0.034946 | 37.170% | 5.516% | 0.316547 | NO_FAILURE |
| acoustic_cell | unseen | SNR 0 dB | 6.681% | 2.633% | 0.836% | 0.017973 | 26.903% | 9.281% | 0.176225 | NO_FAILURE |
| acoustic_cell | unseen | SNR -5 dB | 10.606% | 4.411% | 1.719% | 0.026920 | 28.257% | 7.768% | 0.204893 | NO_FAILURE |

## Bootstrap Intervals

| Section | Domain | Cell | Population E[v] 95% source CI | Selected E[v] 95% source CI | Selected E[v] 95% speaker CI |
| --- | --- | --- | ---: | ---: | ---: |
| ssr_sweep | all | SSR 10% | [0.007605, 0.033451] | [0.083640, 0.208551] | [0.081716, 0.207306] |
| ssr_sweep | all | SSR 30% | [0.002034, 0.021064] | [0.072766, 0.210724] | [0.069390, 0.212362] |
| ssr_sweep | all | SSR 50% | [0.007036, 0.020478] | [0.101717, 0.204628] | [0.099689, 0.204617] |
| ssr_sweep | all | SSR 70% | [0.003296, 0.012935] | [0.064814, 0.164051] | [0.050806, 0.180404] |
| ssr_sweep | all | SSR 90% | [0.004473, 0.014792] | [0.102190, 0.241759] | [0.102166, 0.239785] |
| acoustic_cell | all | SNR 20 dB | [-0.000579, 0.003752] | [-0.016890, 0.091881] | [-0.020002, 0.092745] |
| acoustic_cell | all | SNR 10 dB | [0.001038, 0.005420] | [0.034726, 0.103650] | [0.032938, 0.105422] |
| acoustic_cell | all | SNR 5 dB | [0.002570, 0.036183] | [0.056026, 0.298077] | [0.056258, 0.300798] |
| acoustic_cell | all | SNR 0 dB | [0.001509, 0.018819] | [0.018524, 0.153226] | [-0.000533, 0.163584] |
| acoustic_cell | all | SNR -5 dB | [-0.001363, 0.023025] | [-0.000378, 0.194679] | [-0.017279, 0.194124] |
| ssr_sweep | seen | SSR 10% | [0.023366, 0.047579] | [0.143081, 0.256187] | [0.136436, 0.257580] |
| ssr_sweep | seen | SSR 30% | [0.019129, 0.037533] | [0.133440, 0.245615] | [0.134734, 0.244245] |
| ssr_sweep | seen | SSR 50% | [0.012840, 0.026654] | [0.119561, 0.250514] | [0.113635, 0.253281] |
| ssr_sweep | seen | SSR 70% | [0.003962, 0.014940] | [0.038011, 0.189660] | [0.048544, 0.180002] |
| ssr_sweep | seen | SSR 90% | [-0.003796, 0.003446] | [-0.086295, 0.068971] | [-0.092596, 0.072993] |
| acoustic_cell | seen | SNR 20 dB | [-0.000603, 0.002765] | [-0.025073, 0.060054] | [-0.024080, 0.063556] |
| acoustic_cell | seen | SNR 10 dB | [0.000802, 0.005729] | [0.029750, 0.106383] | [0.031681, 0.107335] |
| acoustic_cell | seen | SNR 5 dB | [-0.000991, 0.008125] | [-0.005323, 0.131121] | [-0.024445, 0.162381] |
| acoustic_cell | seen | SNR 0 dB | [-0.001379, 0.007533] | [-0.028645, 0.103632] | [-0.021414, 0.101223] |
| acoustic_cell | seen | SNR -5 dB | [-0.006667, 0.003139] | [-0.083208, 0.067021] | [-0.056710, 0.046364] |
| ssr_sweep | unseen | SSR 10% | [-0.025621, 0.003613] | [-0.079473, 0.141314] | [-0.077588, 0.141028] |
| ssr_sweep | unseen | SSR 30% | [-0.016842, 0.009709] | [-0.055249, 0.165958] | [-0.036274, 0.148939] |
| ssr_sweep | unseen | SSR 50% | [-0.007516, 0.014819] | [-0.015511, 0.186441] | [-0.014423, 0.194126] |
| ssr_sweep | unseen | SSR 70% | [0.004175, 0.021693] | [0.090222, 0.268293] | [0.092199, 0.255952] |
| ssr_sweep | unseen | SSR 90% | [0.005038, 0.025041] | [0.065570, 0.278351] | [0.071417, 0.278351] |
| acoustic_cell | unseen | SNR 20 dB | [-0.001642, 0.007574] | [-0.029587, 0.195022] | [-0.032111, 0.194635] |
| acoustic_cell | unseen | SNR 10 dB | [-0.001299, 0.007559] | [-0.011113, 0.128213] | [-0.022833, 0.138370] |
| acoustic_cell | unseen | SNR 5 dB | [0.002092, 0.076448] | [0.061220, 0.394114] | [0.060121, 0.403159] |
| acoustic_cell | unseen | SNR 0 dB | [-0.001804, 0.051131] | [-0.029134, 0.295430] | [-0.073564, 0.303064] |
| acoustic_cell | unseen | SNR -5 dB | [0.002463, 0.056711] | [0.108307, 0.352956] | [0.039411, 0.335424] |

## AX4 Assessment

- Status: **F1_VALUE_SCARCITY_DOMINANT**.
- Taxonomy failures: 2.
- F1 value scarcity cells: 2.
- F2 ranking failure cells: 0.
- F3 budget/calibration drift cells: 0.
- AX5 triggered: no.
- Interpretation: The frozen A12 failure cells have non-positive population and selected signed value. Long-context value is scarce in these conditions.
- Limit: AX4 is diagnostic. Its point taxonomy does not authorize gate tuning, a new router, or opening A14.
