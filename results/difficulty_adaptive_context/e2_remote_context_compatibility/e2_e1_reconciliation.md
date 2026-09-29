# E2/E1 remote-context reconciliation

E1 observed that permutation of remote windows was near FULL, ZERO and MATCHED_REPLACE were harmful, and exact temporal ordering was not established as important. E2 decomposes that MATCHED_REPLACE penalty using constrained donor matching.

## Frozen E1 reference checks

- E1 C4 replay passed: `true`.
- Replayed C4 delta log-loss: `0.03213041`.
- Frozen E1 C4 delta log-loss: `0.03213133`.

## E2 condition evidence

- C1 C1_SAME_UTT_DIFFERENT_TIME: `-0.00001556` (source CI `[-0.00048391, 0.00047865]`, `PRIMARY_INTERPRETABLE`)
- C2 C2_SAME_NOISE_INSTANCE: `nan` (source CI `[nan, nan]`, `INCOMPARABLE`)
- C3 C3_SAME_NOISE_CLASS: `0.01469826` (source CI `[0.01078857, 0.01857471]`, `PRIMARY_INTERPRETABLE`)
- C4 C4_SAME_CLASS_WRONG_SNR: `0.01157548` (source CI `[0.00788825, 0.01536931]`, `PRIMARY_INTERPRETABLE`)
- C5 C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR: `0.01895384` (source CI `[0.01449693, 0.02340475]`, `PRIMARY_INTERPRETABLE`)
- C6 C6_SPEECH_STATE_MISMATCH: `0.01148089` (source CI `[0.00574072, 0.01695291]`, `INCOMPARABLE`)
- C7 C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE: `0.01963867` (source CI `[0.01462941, 0.02470213]`, `PRIMARY_INTERPRETABLE`)

## Primary contrasts

- P1_UTTERANCE: `0.01965423` (source CI `[0.01455250, 0.02453462]`). C7 minus C1; positive means different-source replacement is more harmful
- P2_NOISE_INSTANCE: `nan` (source CI `[nan, nan]`). C3 minus C2; positive means class-only replacement is more harmful
- P3_NOISE_CLASS: `0.00425558` (source CI `[0.00096599, 0.00761030]`). C5 minus C3; positive means class mismatch adds damage
- P4_SNR: `-0.00312279` (source CI `[-0.00533231, -0.00084531]`). C4 minus C3; positive means SNR mismatch adds damage
- P5_SPEECH_STATE: `0.00338652` (source CI `[-0.00303472, 0.00964690]`). C6 minus C3; positive means speech-state mismatch adds damage
- P6_RESIDUAL_IDENTITY: `0.01963867` (source CI `[0.01450911, 0.02442867]`). C7 minus FULL; positive means residual replacement damage remains

## Reconciliation classifications

- R1_exact_temporal_order_unimportant_context_statistics_matter: `SUPPORTED`
- R2_same_utterance_identity_matters: `SUPPORTED`
- R3_background_or_noise_instance_matters: `NOT_SUPPORTED`
- R4_noise_class_matters: `PARTIALLY_SUPPORTED`
- R5_snr_or_level_matters: `NOT_SUPPORTED`
- R6_speech_state_composition_matters: `PARTIALLY_SUPPORTED`
- R7_coarse_metadata_do_not_explain: `SUPPORTED`
- R8_insufficient_evidence: `PARTIALLY_SUPPORTED`

## Boundary

- The classifications are bounded to the frozen E2 population and MarbleNet RF384 intervention.
- E1 PERMUTE behavior is not reinterpreted as proof that exact temporal ordering is irrelevant.
- No training, threshold tuning, router search, architecture search, NEW_FINAL_OOD access, or E1 modification is authorized by this reconciliation.

