"""Acoustic feature extraction (Week 1).

The core AcousticSpace idea is physics-based detection, so besides the
standard Mel spectrogram / MFCC we estimate reverberation / Room Impulse
Response (RIR) descriptors. These summarize how the room 'colors' the sound:

  * RT60 estimate        -> reverberation time from the energy-decay curve
  * Direct-to-reverberant ratio (DRR)
  * Spectral decay / low-frequency reverb energy

A genuine recording has RIR statistics consistent with a physical room.
Many deepfakes are synthesized 'dry' or with mismatched reverb, so these
features become discriminative once the model is trained (Week 2+).
"""
from __future__ import annotations

import numpy as np
import librosa

from .config import AUDIO


def mel_spectrogram(y: np.ndarray) -> np.ndarray:
    """Log-scaled Mel spectrogram (dB). Shape: (n_mels, frames)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=AUDIO.sample_rate,
        n_fft=AUDIO.n_fft,
        hop_length=AUDIO.hop_length,
        win_length=AUDIO.win_length,
        n_mels=AUDIO.n_mels,
        fmin=AUDIO.fmin,
        fmax=AUDIO.fmax,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max)


def mfcc(y: np.ndarray) -> np.ndarray:
    """MFCCs capturing vocal timbre. Shape: (n_mfcc, frames)."""
    return librosa.feature.mfcc(
        y=y, sr=AUDIO.sample_rate, n_mfcc=AUDIO.n_mfcc,
        n_fft=AUDIO.n_fft, hop_length=AUDIO.hop_length,
    )


def estimate_rt60(y: np.ndarray) -> float:
    """Rough RT60 (reverberation time) via Schroeder energy-decay curve.

    We measure how long it takes the backward-integrated energy to drop
    from -5 dB to -25 dB, then extrapolate to a 60 dB decay.
    This is an approximation for feature purposes, not lab-grade acoustics.
    """
    if y.size == 0:
        return 0.0
    energy = np.cumsum((y[::-1] ** 2))[::-1]  # Schroeder backward integration
    energy = np.maximum(energy, 1e-12)
    edc_db = 10.0 * np.log10(energy / energy[0])

    try:
        i5 = int(np.argmax(edc_db <= -5.0))
        i25 = int(np.argmax(edc_db <= -25.0))
    except ValueError:
        return 0.0
    if i25 <= i5:
        return 0.0

    decay_time = (i25 - i5) / AUDIO.sample_rate      # seconds for 20 dB drop
    return float(decay_time * (60.0 / 20.0))          # extrapolate to 60 dB


def direct_to_reverberant_ratio(y: np.ndarray) -> float:
    """DRR estimate: energy in the direct-sound window vs. the reverberant tail."""
    if y.size == 0:
        return 0.0
    direct_win = int(0.0025 * AUDIO.sample_rate)  # ~2.5 ms direct-sound window
    direct_win = max(direct_win, 1)
    peak = int(np.argmax(np.abs(y)))
    start = max(peak - direct_win, 0)
    end = min(peak + direct_win, y.size)
    direct = float(np.sum(y[start:end] ** 2))
    reverb = float(np.sum(y ** 2) - direct)
    if reverb <= 1e-12:
        return 0.0
    return float(10.0 * np.log10(max(direct, 1e-12) / reverb))


def reverb_features(y: np.ndarray) -> dict[str, float]:
    """Scalar RIR / reverb descriptors used as physics-based cues."""
    spectral_flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    spectral_centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=y, sr=AUDIO.sample_rate))
    )
    zero_crossing = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    return {
        "rt60_estimate_s": round(estimate_rt60(y), 4),
        "drr_db": round(direct_to_reverberant_ratio(y), 4),
        "spectral_flatness": round(spectral_flatness, 6),
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "zero_crossing_rate": round(zero_crossing, 6),
    }


def extract_all(y: np.ndarray) -> dict[str, object]:
    """Extract the full Week-1 feature set for one preprocessed waveform."""
    mel = mel_spectrogram(y)
    m = mfcc(y)
    return {
        "mel_spectrogram": mel,          # (n_mels, frames) float32
        "mfcc": m,                       # (n_mfcc, frames) float32
        "reverb": reverb_features(y),    # dict of scalars
        "duration_s": round(len(y) / AUDIO.sample_rate, 3),
    }
