# E2 Remote-Context Compatibility Decomposition

- Status: `INCONCLUSIVE_OR_INVALID`
- Protocol: `E2-RCCD-v1`
- Baseline reproduced: `true`
- E1 C4 replay reproduced: `true`
- Training performed: `false`
- NEW_FINAL_OOD touched: `false`
- Next experiment authorized: `false`

## Method boundary

E2 is an inference-only decomposition of the frozen MarbleNet RF384 checkpoint. Donors are constrained using manifest metadata, labels, and waveform energy before model outcomes are inspected. Target/local evidence and the current frame are unchanged.

## Matching feasibility

- C1_SAME_UTT_DIFFERENT_TIME: `PRIMARY_INTERPRETABLE`; eligible targets `396`, eligible sources `96`, matched coverage `5.000000`, unique donor instances `335`.
- C2_SAME_NOISE_INSTANCE: `INCOMPARABLE`; eligible targets `396`, eligible sources `96`, matched coverage `0.000000`, unique donor instances `0`.
- C3_SAME_NOISE_CLASS: `PRIMARY_INTERPRETABLE`; eligible targets `396`, eligible sources `96`, matched coverage `5.000000`, unique donor instances `324`.
- C4_SAME_CLASS_WRONG_SNR: `PRIMARY_INTERPRETABLE`; eligible targets `396`, eligible sources `96`, matched coverage `5.000000`, unique donor instances `290`.
- C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR: `PRIMARY_INTERPRETABLE`; eligible targets `396`, eligible sources `96`, matched coverage `5.000000`, unique donor instances `294`.
- C6_SPEECH_STATE_MISMATCH: `INCOMPARABLE`; eligible targets `396`, eligible sources `96`, matched coverage `1.400998`, unique donor instances `76`.
- C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE: `PRIMARY_INTERPRETABLE`; eligible targets `396`, eligible sources `96`, matched coverage `5.000000`, unique donor instances `324`.

## Condition effects

- C0 C0_FULL: `0.00000000` (source CI `[0.00000000, 0.00000000]`, `BASELINE`)
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

## Event and seen/unseen summaries

- Onset C7 effect: `0.03880445` (source CI `[0.01066299, 0.06596511]`).
- Transition C7 effect: `0.05931344` (source CI `[0.03919673, 0.07796126]`).
- Offset C7 effect: `0.00870802` (source CI `[-0.01365184, 0.03138845]`).
- Onset minus offset: `0.01256746`.
- C1_SAME_UTT_DIFFERENT_TIME: seen `0.00000554`, unseen `-0.00006108`, unseen-minus-seen `-0.00002313`.
- C2_SAME_NOISE_INSTANCE: seen `nan`, unseen `nan`, unseen-minus-seen `nan`.
- C3_SAME_NOISE_CLASS: seen `0.00636762`, unseen `0.03267110`, unseen-minus-seen `0.00599644`.
- C4_SAME_CLASS_WRONG_SNR: seen `0.00849225`, unseen `0.01822734`, unseen-minus-seen `-0.00002981`.
- C5_DIFFERENT_NOISE_CLASS_MATCHED_SNR: seen `0.00726159`, unseen `0.04417911`, unseen-minus-seen `0.00903034`.
- C6_SPEECH_STATE_MISMATCH: seen `0.01798946`, unseen `-0.00487136`, unseen-minus-seen `-0.01425468`.
- C7_BEST_METADATA_MATCHED_DIFFERENT_SOURCE: seen `0.00676394`, unseen `0.04741506`, unseen-minus-seen `0.01039524`.

## E1 reconciliation

E1 C4 replay delta log-loss: `0.03213041` (frozen reference `0.03213133`).

The frozen baseline, matching-validity, intervention-validity, or E1-preservation gate failed. No scientific status is established.

## Decision details

- B1: `true`
- B2: `false`
- B3: `true`
- B4: `true`
- B5: `true`
- SPEECH_STATE_DOMINATES: `false`
- RESIDUAL_CONTEXT_IDENTITY: `true`

## Boundary

- E2 does not train a background encoder, context classifier, router, attention module, embedding, or normalization module.
- E2 does not modify RF384 or Tiny GRU.
- E2 does not open NEW_FINAL_OOD and does not modify E1.
- E2 does not authorize E3, E4, A-v3, A14 work, or any follow-on experiment.
- The result is specific to the frozen MarbleNet RF384 checkpoint and the E2 eligible population unless stated otherwise.

