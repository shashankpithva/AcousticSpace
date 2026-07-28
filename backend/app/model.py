"""
Week 2 baseline CNN classifier for AcousticSpace.

Input:
    Mel spectrogram (n_mels, frames)

Output:
    2 classes:
      0 -> real
      1 -> fake
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def predict(model: AudioCNN, mel):
    """
    Convert one mel spectrogram into a prediction.

    Returns:
        label, confidence
    """

    model.eval()

    with torch.no_grad():
        tensor = torch.tensor(mel, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        confidence, index = torch.max(probs, dim=1)

    labels = ["real", "fake"]

    return labels[index.item()], float(confidence.item())