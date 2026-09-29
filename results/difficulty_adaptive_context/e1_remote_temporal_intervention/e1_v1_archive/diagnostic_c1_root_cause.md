# C1 Reproduction Diagnostic

## Frozen inputs

- Protocol SHA-256: `52D995474E455649DD81D86138B731BEFBBA725D8F5028452928FA7C2CF5F793`
- Frozen runner SHA-256: `A03A0939DF7D54C9195C05930A2BDC4AE76896BC8CF55F91A9567270003397F9`
- Frozen `score_chunk_frames`: `2000`

## Exact C1 result

The frozen C1 command completed twice with the same result:

- `passed=false`
- `max_abs_full_error=0.00019294023513793945`
- `max_abs_embedded_short_error=0.00016936659812927246`
- `tolerance=1e-06`

The two runs produced identical per-record errors. The preserved first run is
`diagnostic_c1_full_run1_preserved.json`.

## Item 68 comparison

The item with `item_index=68` is
`LibriSpeech:test:Babble_noise:0:8230-279154-0038`.

On CUDA, in one process:

- Encoded features from the runner path and the streaming path were bitwise
  identical (`max_abs=0`).
- The frozen reference wrapper reproduced the frozen Full and embedded Short
  scores bitwise (`max_abs=0`).
- The runner's full-stream embedded Short score differed from the frozen
  embedded Short score by `0.00016033649444580078`.
- The runner's full-stream Full score differed from the frozen Full score by
  `0.0001735091209411621`.
- Applying the classifier to the same encoded chunks used by the streaming
  reference reproduced the wrapper Short scores bitwise (`max_abs=0`).
- Applying the classifier to the same encoded features as one full sequence
  reproduced the runner's mismatch (`max_abs=0.00016033649444580078`).

## Conclusion

The failure is deterministic and is associated with a numerical path
difference between:

- the frozen reference, which applies the classifier and refinement per
  `chunk_frames=2000` chunk, and
- the frozen E1 runner, which applies the classifier and manual refinement to
  the full concatenated encoded sequence.

The embedded Short mismatch is directly isolated: chunked classification is
bitwise exact, while full-sequence classification is not. The Full mismatch is
consistent with the same chunking difference plus the runner's full-sequence
manual refinement path.

## Full-population isolation

The remaining Full mismatch was isolated from refinement arithmetic. For item
68, replacing only the full-sequence classifier logits with classifier logits
computed in the same 2,000-frame chunks used by the frozen reference reduced
the Full error from `0.0001735091209411621` to `5.960464477539063e-08`.

The same chunked-classifier correction was then evaluated across all 1,024
frozen records and 553,532 frames without running any intervention:

- `max_abs_full_error=2.384185791015625e-07`
- `max_abs_embedded_short_error=0.0`
- `tolerance=1e-06`
- `passed=true`

This shows that the existing manual branch/refinement path is numerically
consistent with the frozen streaming reference when classifier evaluation is
performed on the frozen 2,000-frame scoring chunks. The complete diagnostic
record is `diagnostic_c1_chunked_classifier_full.json`.

The frozen data, encoded features, checkpoint, and frozen reference are
consistent. The runner's C1 reproduction check nevertheless fails its frozen
tolerance. Under the protocol's decision precedence, this requires:

`E1_STATUS = INCONCLUSIVE_OR_INVALID`

No intervention run was started. No training, tuning, protocol change,
threshold change, A14 rerun, or `NEW_FINAL_OOD` access occurred.
