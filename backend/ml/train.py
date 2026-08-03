"""Train the Week 2 baseline CNN on cached spectrogram features.

Usage (from the backend/ directory):
    python -m ml.train --features data/features --epochs 15

Saves the best checkpoint to models/baseline_cnn.pt and the full metrics
history to models/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.config import MODELS_DIR, TRAIN
from ml.dataset import train_val_split
from ml.model import BaselineCNN


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():   # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the AcousticSpace baseline CNN.")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN.lr)
    ap.add_argument("--out", default=str(MODELS_DIR / "baseline_cnn.pt"))
    args = ap.parse_args()

    torch.manual_seed(TRAIN.seed)
    np.random.seed(TRAIN.seed)
    device = pick_device()
    print(f"[train] device = {device}")

    train_ds, val_ds = train_val_split(args.features, TRAIN.val_split, TRAIN.seed)
    print(f"[train] train clips = {len(train_ds)}  val clips = {len(val_ds)}")
    if len(train_ds) == 0:
        raise SystemExit("No training data. Extract features first.")

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size)

    model = BaselineCNN(n_classes=TRAIN.n_classes, dropout=TRAIN.dropout).to(device)
    print(f"[train] trainable parameters = {model.count_parameters():,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=TRAIN.weight_decay
    )

    history: list[dict] = []
    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_ld, criterion, device, optimizer)
        vl, va = run_epoch(model, val_ld, criterion, device) if len(val_ds) else (0.0, 0.0)
        history.append(
            {"epoch": epoch, "train_loss": tl, "train_acc": ta,
             "val_loss": vl, "val_acc": va}
        )
        print(
            f"[epoch {epoch:02d}] "
            f"train_loss={tl:.4f} train_acc={ta:.3f} | "
            f"val_loss={vl:.4f} val_acc={va:.3f}"
        )
        if va >= best_val:
            best_val = va
            torch.save(
                {"state_dict": model.state_dict(),
                 "n_classes": TRAIN.n_classes,
                 "val_acc": va, "epoch": epoch},
                args.out,
            )
            print(f"[train]   ✓ saved best checkpoint -> {args.out} (val_acc={va:.3f})")

    (MODELS_DIR / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"[train] done. best val_acc = {best_val:.3f}")
    print(f"[train] metrics history -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
