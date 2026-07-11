"""Central configuration for the AcousticSpace audio pipeline.

Keeping every tunable constant here means the pipeline, the feature
extractor and the batch scripts all agree on sample rate, FFT size, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    # --- Loading / preprocessing ---
    sample_rate: int = 16_000        # ASVspoof is 16 kHz; AST also expects 16 kHz
    mono: bool = True
    trim_top_db: float = 30.0        # silence-trim threshold (dB below peak)
    target_seconds: float = 4.0      # clips are padded/trimmed to this length

    # --- Spectrogram / Mel ---
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float | None = None        # None -> sample_rate / 2

    # --- MFCC ---
    n_mfcc: int = 20

    @property
    def target_samples(self) -> int:
        return int(self.sample_rate * self.target_seconds)


@dataclass(frozen=True)
class Paths:
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def uploads(self) -> Path:
        return self.data / "uploads"

    @property
    def features(self) -> Path:
        return self.data / "features"


AUDIO = AudioConfig()
PATHS = Paths()

# Accepted upload formats (extensions) for the analyst dashboard.
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac"}
