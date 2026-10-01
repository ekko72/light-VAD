# AVA-Speech Real-Recording Validation

## Protocol

- Protocol: `AVA-SPEECH-v1`
- Status: `FROZEN_BEFORE_AVA_SCORING`
- Official test videos: xO4ABy2iOQA, y7ncweROe9U
- Fit videos: 126
- Validation videos: 32
- Source cluster: `video_id`
- Speaker-disjoint evaluation was not proven and is not claimed.
- `NEW_FINAL_OOD` and `FINAL_OOD` were not opened.

## Main Results

| Model | Condition | F1 | AUC | Proper loss | TPR@FPR=0.315 |
| --- | --- | ---: | ---: | ---: | ---: |
| Short | overall | 0.7414 | 0.8009 | 0.7795 | 0.7583 |
| Short | clean_speech | 0.9129 | n/a | 0.3476 | n/a |
| Short | speech_with_noise | 0.8820 | n/a | 0.3969 | n/a |
| Short | speech_with_music | 0.9621 | n/a | 0.1632 | n/a |
| Short | no_speech | 0.0000 | n/a | 1.2961 | n/a |
| Long-RF | overall | 0.7435 | 0.7874 | 0.7562 | 0.7519 |
| Long-RF | clean_speech | 0.9063 | n/a | 0.3695 | n/a |
| Long-RF | speech_with_noise | 0.8551 | n/a | 0.4575 | n/a |
| Long-RF | speech_with_music | 0.9481 | n/a | 0.1929 | n/a |
| Long-RF | no_speech | 0.0000 | n/a | 1.1933 | n/a |
| Adaptive | overall | 0.7366 | 0.7998 | 0.7924 | 0.7566 |
| Adaptive | clean_speech | 0.9178 | n/a | 0.3433 | n/a |
| Adaptive | speech_with_noise | 0.8927 | n/a | 0.3883 | n/a |
| Adaptive | speech_with_music | 0.9577 | n/a | 0.1691 | n/a |
| Adaptive | no_speech | 0.0000 | n/a | 1.3292 | n/a |
| AlwaysRefine | overall | 0.7322 | 0.7950 | 0.8525 | 0.7559 |
| AlwaysRefine | clean_speech | 0.9193 | n/a | 0.3126 | n/a |
| AlwaysRefine | speech_with_noise | 0.8952 | n/a | 0.3348 | n/a |
| AlwaysRefine | speech_with_music | 0.9546 | n/a | 0.1716 | n/a |
| AlwaysRefine | no_speech | 0.0000 | n/a | 1.5009 | n/a |
| WebRTC VAD | overall | 0.7139 | 0.6216 | 10.0039 | 0.8410 |
| WebRTC VAD | clean_speech | 0.9139 | n/a | 4.3832 | n/a |
| WebRTC VAD | speech_with_noise | 0.8974 | n/a | 5.1441 | n/a |
| WebRTC VAD | speech_with_music | 0.9525 | n/a | 2.5045 | n/a |
| WebRTC VAD | no_speech | 0.0000 | n/a | 16.5163 | n/a |
| Silero VAD | overall | 0.8455 | 0.9347 | 0.4715 | 0.9334 |
| Silero VAD | clean_speech | 0.9193 | n/a | 0.4460 | n/a |
| Silero VAD | speech_with_noise | 0.8575 | n/a | 0.7884 | n/a |
| Silero VAD | speech_with_music | 0.8255 | n/a | 0.8171 | n/a |
| Silero VAD | no_speech | 0.0000 | n/a | 0.1824 | n/a |
| NeMo MarbleNet | overall | 0.7943 | 0.9121 | 0.6003 | 0.9220 |
| NeMo MarbleNet | clean_speech | 0.8856 | n/a | 0.6638 | n/a |
| NeMo MarbleNet | speech_with_noise | 0.8292 | n/a | 0.8463 | n/a |
| NeMo MarbleNet | speech_with_music | 0.7273 | n/a | 1.2924 | n/a |
| NeMo MarbleNet | no_speech | 0.0000 | n/a | 0.2457 | n/a |
| Shallow GBDT | overall | 0.8114 | 0.8810 | 0.4291 | 0.8546 |
| Shallow GBDT | clean_speech | 0.9332 | n/a | 0.2839 | n/a |
| Shallow GBDT | speech_with_noise | 0.8571 | n/a | 0.4562 | n/a |
| Shallow GBDT | speech_with_music | 0.8362 | n/a | 0.4917 | n/a |
| Shallow GBDT | no_speech | 0.0000 | n/a | 0.4320 | n/a |

### Cost

| Model | MACs/frame | Streaming cache (bytes) | Latency (ms/frame) | Causal |
| --- | ---: | ---: | ---: | --- |
| Short | 45312.0 | 34,304 | 0.0224 | yes |
| Long-RF | 45312.0 | 103,936 | 0.0213 | yes |
| Adaptive | 45634.6 | 230,912 | 0.0339 | yes |
| AlwaysRefine | 48768.0 | 230,912 | 0.0300 | yes |
| WebRTC VAD | n/a | n/a | n/a | yes |
| Silero VAD | n/a | n/a | 0.2479 | yes |
| NeMo MarbleNet | n/a | n/a | 0.0342 | yes |
| Shallow GBDT | n/a | n/a | 0.0507 | yes |

- Short cache: 34,304 bytes.
- Long-RF cache: 103,936 bytes.
- Adaptive cache: 230,912 bytes.
- GBDT training frames: 252,000.

## Mechanism

### Distance Sweep

| Distance (s) | Condition | Mean delta log-loss | 95% CI | Targets |
| ---: | --- | ---: | --- | ---: |
| 0.25 | order_shuffle | 0.0885 | [0.0885, 0.0885] | 512 |
| 0.25 | same_utterance_alternate | 0.3141 | [0.3141, 0.3141] | 512 |
| 0.25 | different_source | 0.8655 | [0.8655, 0.8655] | 512 |
| 0.50 | order_shuffle | 0.0095 | [0.0095, 0.0095] | 512 |
| 0.50 | same_utterance_alternate | 0.2036 | [0.2036, 0.2036] | 512 |
| 0.50 | different_source | 0.6490 | [0.6490, 0.6490] | 512 |
| 1.00 | order_shuffle | -0.0019 | [-0.0019, -0.0019] | 512 |
| 1.00 | same_utterance_alternate | 0.1099 | [0.1099, 0.1099] | 512 |
| 1.00 | different_source | 0.4646 | [0.4646, 0.4646] | 512 |
| 2.00 | order_shuffle | -0.0087 | [-0.0087, -0.0087] | 512 |
| 2.00 | same_utterance_alternate | -0.0120 | [-0.0120, -0.0120] | 512 |
| 2.00 | different_source | 0.1100 | [0.1100, 0.1100] | 512 |
| 3.00 | order_shuffle | -0.0008 | [-0.0008, -0.0008] | 512 |
| 3.00 | same_utterance_alternate | -0.0024 | [-0.0024, -0.0024] | 512 |
| 3.00 | different_source | 0.0016 | [0.0016, 0.0016] | 512 |
| 0.25 | order_shuffle | 0.1095 | [0.1095, 0.1095] | 512 |
| 0.25 | same_utterance_alternate | 0.6576 | [0.6576, 0.6576] | 512 |
| 0.25 | different_source | 1.3807 | [1.3807, 1.3807] | 512 |
| 0.50 | order_shuffle | 0.0425 | [0.0425, 0.0425] | 512 |
| 0.50 | same_utterance_alternate | 0.5057 | [0.5057, 0.5057] | 512 |
| 0.50 | different_source | 0.9987 | [0.9987, 0.9987] | 512 |
| 1.00 | order_shuffle | -0.0021 | [-0.0021, -0.0021] | 512 |
| 1.00 | same_utterance_alternate | 0.2856 | [0.2856, 0.2856] | 512 |
| 1.00 | different_source | 0.6775 | [0.6775, 0.6775] | 512 |
| 2.00 | order_shuffle | -0.0127 | [-0.0127, -0.0127] | 512 |
| 2.00 | same_utterance_alternate | 0.0606 | [0.0606, 0.0606] | 512 |
| 2.00 | different_source | 0.1678 | [0.1678, 0.1678] | 512 |
| 3.00 | order_shuffle | -0.0005 | [-0.0005, -0.0005] | 512 |
| 3.00 | same_utterance_alternate | 0.0004 | [0.0004, 0.0004] | 512 |
| 3.00 | different_source | 0.0028 | [0.0028, 0.0028] | 512 |

### Hard Frames and Gate

| Analysis | Model/Condition | Value |
| --- | --- | ---: |
| hard_frame | Short | F1=0.4040, loss=0.7210 |
| hard_frame | Long-RF | F1=0.3276, loss=0.8896 |
| hard_frame | Adaptive | F1=0.4125, loss=0.7855 |
| hard_frame | AlwaysRefine | F1=0.4040, loss=0.8387 |
| hard_frame | WebRTC VAD | F1=0.3992, loss=16.5878 |
| hard_frame | Silero VAD | F1=0.5747, loss=0.6645 |
| hard_frame | NeMo MarbleNet | F1=0.6296, loss=0.6027 |
| hard_frame | Shallow GBDT | F1=0.5283, loss=0.5513 |
| hard_frame_delta | Long-RF minus Short | delta F1=-0.0764, delta loss=0.1685 |
| gate_comparison | Adaptive uncertainty gate minus random | delta F1=-0.0039, delta loss=0.0061, activation=0.0933 |

## Lightweight Alternative

| Seed | Condition | F1 | AUC | Proper loss | Added cache (bytes) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 17 | overall | 0.8292 | 0.8969 | 0.4228 | 512 |
| 17 | clean_speech | 0.9651 | n/a | 0.1680 | 512 |
| 17 | speech_with_noise | 0.9227 | n/a | 0.3076 | 512 |
| 17 | speech_with_music | 0.9238 | n/a | 0.3395 | 512 |
| 17 | no_speech | 0.0000 | n/a | 0.5835 | 512 |
| 17 | hard_frames_top20 | 0.6231 | 0.7836 | 0.5782 | 512 |
| 23 | overall | 0.7624 | 0.8845 | 0.4721 | 512 |
| 23 | clean_speech | 0.8830 | n/a | 0.4995 | 512 |
| 23 | speech_with_noise | 0.7428 | n/a | 0.6844 | 512 |
| 23 | speech_with_music | 0.8194 | n/a | 0.5382 | 512 |
| 23 | no_speech | 0.0000 | n/a | 0.3099 | 512 |
| 23 | hard_frames_top20 | 0.4582 | 0.7805 | 0.5238 | 512 |
| 41 | overall | 0.7501 | 0.8775 | 0.4956 | 512 |
| 41 | clean_speech | 0.8972 | n/a | 0.4703 | 512 |
| 41 | speech_with_noise | 0.7273 | n/a | 0.7473 | 512 |
| 41 | speech_with_music | 0.7651 | n/a | 0.6792 | 512 |
| 41 | no_speech | 0.0000 | n/a | 0.2908 | 512 |
| 41 | hard_frames_top20 | 0.3576 | 0.7520 | 0.5474 | 512 |

- Lightweight chunks: 1,008.
- Lightweight added cache: 512 bytes.
- Long-RF reference cache: 103,936 bytes.

## Interpretation Guardrails

- This report is an external-distribution validation, not a claim that Adaptive/M2 is superior.
- The frozen gate threshold was not tuned on AVA-Speech.
- Results are reported with source-cluster bootstrap intervals; with only two official test videos, source-level uncertainty is necessarily coarse.
