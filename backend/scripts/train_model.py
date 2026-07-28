"""
Train AcousticSpace Week 2 baseline CNN.

Input:
    data/features/*.npz

Output:
    models/baseline_cnn.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from app.model import AudioCNN


class FeatureDataset(Dataset):
    def __init__(self, folder: Path):
        self.files = sorted(folder.glob("*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        mel = data["mel"].astype(np.float32)
        label = int(data["label"])

        return (
            torch.tensor(mel).unsqueeze(0),
            torch.tensor(label),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/features",
    )
    parser.add_argument(
        "--out",
        default="models/baseline_cnn.pt",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    dataset = FeatureDataset(Path(args.features))

    if len(dataset) == 0:
        raise RuntimeError(
            "No features found. Run extract_features.py first."
        )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    model = AudioCNN()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0

        for x, y in loader:
            optimizer.zero_grad()

            output = model(x)

            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"loss={total_loss:.4f}"
        )

    out = Path(args.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        out,
    )

    print(f"Saved model: {out}")


if __name__ == "__main__":
    main()