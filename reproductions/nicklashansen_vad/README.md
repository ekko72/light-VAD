# nicklashansen/voice-activity-detection reproduction

This directory runs the six official checkpoints from
`nicklashansen/voice-activity-detection` with a modern PyTorch installation.
It does not retrain the models: the upstream 27 GB processed-data link has
expired, and the original corpus uses QUT-NOISE.

## Pinned revision

- Repository: <https://github.com/nicklashansen/voice-activity-detection>
- Commit: `fa6f57855a41380f44f5c95c3d1252fc5856da73` (2019-10-14)
- Model directory: the six `.net` files under upstream `data/models`

The checkpoints are full Python object pickles, not state dictionaries. The
loader therefore requires `weights_only=False`. Only use it with these pinned,
downloaded files; never load an untrusted `.net` file.

## Local data substitution

- Speech: LibriSpeech `test-clean`, selected by `data/splits/test_speech.tsv`.
- Noise: MUSAN `noise` category, peak-normalized per sampled clip.
- QUT-NOISE is not present, so the measured numbers are a reproduction of the
  upstream evaluation protocol rather than a claim of the paper's exact score.

## Protocol

The balanced evaluator follows the notebook where the local data allows:

1. Sample clean LibriSpeech slices of 1-5 seconds.
2. Add approximately the same duration of zero silence, then shuffle.
3. Put speech on a peak-normalized noise bed with overlay gain `None`, `-15`,
   or `-3` dB. In the upstream code, `None` means 0 dB overlay, not no noise.
4. Label each clean 30 ms frame with WebRTC VAD mode 0 before noise is added.
5. Extract 12 MFCC coefficients after dropping coefficient zero and 12 deltas.
6. Form 30-frame contexts, use the center frame label, step by 6, and run with
   the models' hard-coded batch size of 2048.

## Commands

```powershell
.\.venv\Scripts\python.exe -m reproductions.nicklashansen_vad.check_checkpoints
.\.venv\Scripts\python.exe -m reproductions.nicklashansen_vad.run_inference --limit 10
.\.venv\Scripts\python.exe -m reproductions.nicklashansen_vad.generate_eval --speech-minutes 10 --device auto
```

The balanced run writes `data/reproductions/nicklashansen_vad_balanced_musan.json`.
Its `--noise-categories` option defaults to MUSAN's `noise` category so music
and speech recordings are not accidentally treated as environmental noise.

## Latest local result

The 10-minute balanced run generated 601.47 s of speech and 602.16 s of
silence, with 6,144 evaluated center-frame windows per noise condition.

| Checkpoint | AUC, 0 dB | AUC, -15 dB | AUC, -3 dB | FAR @ FRR 1%, -15 dB |
| --- | ---: | ---: | ---: | ---: |
| `net_epoch014.net` | 0.7942 | 0.9142 | 0.8348 | 50.94% |
| `net_large_epoch014.net` | 0.7995 | 0.9324 | 0.8405 | 50.76% |
| `gru_epoch014.net` | 0.7773 | 0.9173 | 0.8216 | 42.68% |
| `gru_large_epoch014.net` | 0.7732 | 0.9042 | 0.8144 | 38.74% |
| `densenet_epoch012.net` | 0.7492 | 0.8968 | 0.7871 | 55.92% |
| `densenet_large_epoch014.net` | 0.7532 | 0.9063 | 0.7933 | 53.01% |

These values use MUSAN rather than the upstream QUT-NOISE corpus and are not
the paper's original benchmark numbers.

## Limitations

- The official processed dataset and original QUT-NOISE corpus are not present.
- MUSAN is a substitute, so absolute AUC and FAR values are not directly
  comparable to the paper's QUT-NOISE results.
- The official checkpoints were produced with an older PyTorch pickle format.
  `legacy_models.py` contains compatibility code for loading them in torch 2.x.
