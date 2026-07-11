"""Audio loading and preprocessing pipeline (Week 1).

Stages: load -> resample -> mono -> normalize -> trim silence -> fix length.
Everything downstream (features, model) consumes the output of `preprocess`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import librosa
import numpy as np

from .config import AUDIO


def load_audio(path: Union[str, Path], sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load an audio file as a float32 waveform at the target sample rate."""
    sr = sr or AUDIO.sample_rate
    y, sr = librosa.load(str(path), sr=sr, mono=AUDIO.mono)
    return y.astype(np.float32), sr


def normalize(y: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1]; safe on silent clips."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-8:
        return y
    return y / peak


def trim_silence(y: np.ndarray) -> np.ndarray:
    """Trim leading/trailing silence below the configured threshold."""
    trimmed, _ = librosa.effects.trim(y, top_db=AUDIO.trim_top_db)
    return trimmed if trimmed.size else y


def fix_length(y: np.ndarray, target: int | None = None) -> np.ndarray:
    """Pad with zeros or center-trim so every clip has the same sample count."""
    target = target or AUDIO.target_samples
    return librosa.util.fix_length(y, size=target)


def preprocess(path: Union[str, Path]) -> np.ndarray:
    """Full preprocessing chain, returning a fixed-length mono waveform."""
    y, _ = load_audio(path)
    y = normalize(y)
    y = trim_silence(y)
    y = fix_length(y)
    return y
