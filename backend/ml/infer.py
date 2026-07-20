"""Inference helper used by the FastAPI /analyze endpoint (Week 2).

Loads the trained baseline checkpoint once and predicts real-vs-deepfake
for a preprocessed waveform. Returns None from get_predictor() when no
checkpoint exists yet, so the API can gracefully fall back to Week 1
feature-only mode.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from app.features import mel_spectrogram
from ml.config import CLASS_NAMES, MODELS_DIR
from ml.model import BaselineCNN


class BaselinePredictor:
    def __init__(self, ckpt_path: pathlib.Path) -> None:
        self.device = torch.device("cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model = BaselineCNN(n_classes=ckpt.get("n_classes", 2))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.val_acc = ckpt.get("val_acc")

    def predict(self, waveform: np.ndarray) -> dict[str, float | str]:
        mel = mel_spectrogram(waveform).astype(np.float32)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1,1,n_mels,frames)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0]
        idx = int(torch.argmax(probs))
        return {
            "prediction": CLASS_NAMES.get(idx, "undetermined"),
            "confidence": float(probs[idx]),
            "prob_fake": float(probs[1]),
        }


def get_predictor() -> BaselinePredictor | None:
    ckpt = MODELS_DIR / "baseline_cnn.pt"
    if not ckpt.exists():
        return None
    try:
        return BaselinePredictor(ckpt)
    except Exception:  # noqa: BLE001
        return None
