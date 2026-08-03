"""PyTorch dataset over the cached .npz features produced in Week 1.

Each .npz (written by scripts/extract_features.py) contains:
  mel   -> (n_mels, frames) float32 log-Mel spectrogram
  mfcc  -> (n_mfcc, frames) float32
  label -> int64 (0 = real, 1 = fake)

The dataset returns a per-sample standardized Mel image and its label.
"""
from __future__ import annotations

import sys
import pathlib
from pathlib import Path

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        data = np.load(self.files[idx])
        mel = data["mel"].astype(np.float32)
        # Per-sample standardization keeps inputs stable across clips.
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        x = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, frames)
        y = int(data["label"])
        return x, y


def list_feature_files(features_dir: str | Path) -> list[Path]:
    return sorted(Path(features_dir).glob("*.npz"))


def train_val_split(
    features_dir: str | Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[FeatureDataset, FeatureDataset]:
    """Stratified train/val split (keeps real:fake ratio in both halves)."""
    files = list_feature_files(features_dir)
    if not files:
        raise FileNotFoundError(
            f"No .npz features found in '{features_dir}'. "
            "Run scripts/extract_features.py first."
        )

    real = [f for f in files if f.name.startswith("real__")]
    fake = [f for f in files if f.name.startswith("fake__")]
    rng = np.random.default_rng(seed)

    def _split(items: list[Path]) -> tuple[list[Path], list[Path]]:
        if not items:
            return [], []
        order = rng.permutation(len(items))
        n_val = max(1, int(len(items) * val_split)) if len(items) > 1 else 0
        val = [items[i] for i in order[:n_val]]
        train = [items[i] for i in order[n_val:]]
        return train, val

    tr_r, va_r = _split(real)
    tr_f, va_f = _split(fake)
    return FeatureDataset(tr_r + tr_f), FeatureDataset(va_r + va_f)
