from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import python_speech_features
import soundfile as sf
import webrtcvad


SAMPLE_RATE = 16_000
FRAME_SIZE_SAMPLES = 480
FRAME_STEP_SECONDS = 0.03
MFCC_WINDOW_FRAMES = 4
MFCC_COEFFICIENTS = 12
SEQUENCE_FRAMES = 30
SEQUENCE_FEATURES = 24


@dataclass(frozen=True)
class PreparedAudio:
    path: Path
    frames: np.ndarray
    labels: np.ndarray
    mfcc: np.ndarray
    delta: np.ndarray

    @property
    def features(self) -> np.ndarray:
        return np.hstack((self.mfcc, self.delta))


def _load_pcm16(path: str | Path) -> np.ndarray:
    waveform, sample_rate = sf.read(str(path), dtype="int16", always_2d=True)
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1).astype(np.int16)
    else:
        waveform = waveform[:, 0]

    if sample_rate != SAMPLE_RATE:
        float_waveform = waveform.astype(np.float32) / 32768.0
        float_waveform = librosa.resample(
            float_waveform,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )
        waveform = np.clip(
            np.rint(float_waveform * 32768.0),
            -32768,
            32767,
        ).astype(np.int16)
    return waveform


def _pad_and_frame(waveform: np.ndarray) -> np.ndarray:
    # Upstream always appends a complete padding frame, even for exact multiples.
    padding = FRAME_SIZE_SAMPLES - (len(waveform) % FRAME_SIZE_SAMPLES)
    padded = np.pad(waveform, (0, padding))
    return padded.reshape(-1, FRAME_SIZE_SAMPLES)


def _webrtc_labels(frames: np.ndarray) -> np.ndarray:
    vad = webrtcvad.Vad(0)
    return np.asarray(
        [
            1 if vad.is_speech(frame.astype(np.int16).tobytes(), SAMPLE_RATE) else 0
            for frame in frames
        ],
        dtype=np.int8,
    )


def _mfcc_features(
    frames: np.ndarray,
    alignment_frames: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if alignment_frames is None:
        alignment = np.zeros(
            (MFCC_WINDOW_FRAMES - 1, FRAME_SIZE_SAMPLES),
            dtype=np.int16,
        )
    else:
        alignment = np.asarray(alignment_frames, dtype=np.int16)
        expected_shape = (MFCC_WINDOW_FRAMES - 1, FRAME_SIZE_SAMPLES)
        if alignment.shape != expected_shape:
            raise ValueError(
                f"MFCC alignment shape must be {expected_shape}, got {alignment.shape}"
            )

    aligned = np.concatenate((alignment, frames), axis=0)
    mfcc = python_speech_features.mfcc(
        aligned,
        SAMPLE_RATE,
        winstep=FRAME_STEP_SECONDS,
        winlen=MFCC_WINDOW_FRAMES * FRAME_STEP_SECONDS,
        nfft=2048,
    )
    if len(mfcc) != len(frames):
        raise RuntimeError(
            f"MFCC frame mismatch: got {len(mfcc)}, expected {len(frames)}"
        )

    # Coefficient zero only represents signal energy / DC offset upstream.
    mfcc = mfcc[:, 1:].astype(np.float32)
    delta = python_speech_features.delta(mfcc, 2).astype(np.float32)
    return mfcc, delta


def prepare_audio(
    path: str | Path,
    noise_frames: np.ndarray | None = None,
    gain_db: float | None = None,
) -> PreparedAudio:
    """Prepare one waveform using the upstream 30 ms / MFCC protocol."""

    path = Path(path)
    waveform = _load_pcm16(path)
    frames = _pad_and_frame(waveform)
    labels = _webrtc_labels(frames)

    if noise_frames is not None:
        if gain_db is None:
            gain_db = 0.0
        repeated = np.resize(np.asarray(noise_frames, dtype=np.int16), len(waveform))
        scale = 10.0 ** (gain_db / 20.0)
        mixed = repeated.astype(np.float32) + waveform.astype(np.float32) * scale
        waveform = np.clip(np.rint(mixed), -32768, 32767).astype(np.int16)
        frames = _pad_and_frame(waveform)

    mfcc, delta = _mfcc_features(frames)
    return PreparedAudio(
        path=path,
        frames=frames,
        labels=labels,
        mfcc=mfcc,
        delta=delta,
    )


def center_windows(
    prepared: PreparedAudio,
    sequence_frames: int = SEQUENCE_FRAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sliding model inputs and labels for the center frame."""

    windows = np.lib.stride_tricks.sliding_window_view(
        prepared.features,
        sequence_frames,
        axis=0,
    )
    windows = np.ascontiguousarray(windows.transpose(0, 2, 1), dtype=np.float32)
    center = sequence_frames // 2
    labels = prepared.labels[center : center + len(windows)].astype(np.int64)
    return windows, labels
