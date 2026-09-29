OBSERVATIONS
- E1_STATUS: REMOTE_INFORMATION_CONDITIONAL
- C1 FULL reproduced: true
- C3 REMOTE_PERMUTE delta log-loss: 0.00023030291364668006
- C3 source-cluster 95% CI: [-0.00018317710908267, 0.0006266319092749803]
- C4 MATCHED_REPLACE delta log-loss: 0.032131332797710244
- C4 source-cluster 95% CI: [0.024512764554617997, 0.039888602005205116]
- C2 REMOTE_ZERO delta log-loss: 0.02067777155708292
- C5 LOCAL_PERMUTE delta log-loss: -0.0011323968320816327
- Original-R prevalence: 0.013345625209520616
- R survival under FULL/PERMUTE/MATCHED: 1.0/0.851243406179352/0.3262999246420497
- PROTOCOL_VERSION: E1-RTI-v2
- V1_INVALID_RETAINED: true
- SCIENTIFIC_PROTOCOL_CHANGED: false
- IMPLEMENTATION_CORRECTION: true
- INTERVENTION_EXECUTED_BEFORE_V2_FREEZE: false
- Authorization disclosure: E1-RTI-v1 was invalidated before intervention because the manual scoring path did not reproduce the frozen Full baseline within the preregistered tolerance. A diagnostic performed without executing interventions isolated the discrepancy to classifier chunking semantics. After explicit authorization, E1-RTI-v2 changed only the classifier scoring path to reproduce the frozen 2,000-frame chunked reference. The scientific hypotheses, interventions, endpoints, thresholds, gates, and evaluation population were unchanged.

SUPPORTED_INTERPRETATIONS
The tested data are consistent with some dependence on correctly related remote temporal information, but the pre-registered conditions for a supported conclusion are not met.
- A9/AE2 reconciliation classification: T2. The evidence points to a small, high-value correction subset, but source or realization uncertainty prevents the stronger aggregate claim.

ALTERNATIVE_EXPLANATIONS
- Intervention distribution shift may contribute to the observed loss changes.
- RF384 topology or processing effects may remain even when remote alignment is perturbed.
- Feature-statistics mismatch may contribute to destructive-control effects.
- Boundary and annotation uncertainty may affect frame-level correction survival.
- A small high-value subset may be responsible for a large share of the aggregate effect.
- Source/domain heterogeneity may limit generalization of the aggregate result.

UNSUPPORTED_CLAIMS
- 384 contiguous frames are necessary.
- Long-term semantic context is the causal mechanism.
- Architecture capacity alone has been proven to be the mechanism.
- The result generalizes to NEW_FINAL_OOD or to a new model.
- Any next experiment is authorized.
