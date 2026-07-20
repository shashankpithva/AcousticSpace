"""Training configuration for the Week 2 baseline model."""
from __future__ import annotations

import sys
import pathlib
from dataclasses import dataclass

# Make `app` importable whether run as `python -m ml.train` or `python ml/train.py`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.config import PATHS  # noqa: E402

MODELS_DIR = PATHS.models
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.2
    seed: int = 42
    n_classes: int = 2       # 0 = real / authentic, 1 = fake / deepfake
    dropout: float = 0.3


TRAIN = TrainConfig()

# Canonical class mapping used everywhere (dataset labels, inference output).
CLASS_NAMES = {0: "authentic", 1: "deepfake"}
