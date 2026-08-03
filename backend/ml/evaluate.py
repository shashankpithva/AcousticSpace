"""Evaluate the trained baseline model on the validation split.

Reports accuracy, precision, recall, F1, a confusion matrix and the
Equal Error Rate (EER) — the standard metric for spoof detection.

Usage (from backend/):
    python -m ml.evaluate --features data/features
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)

from ml.config import MODELS_DIR, TRAIN
from ml.dataset import train_val_split
from ml.model import BaselineCNN


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Equal Error Rate: the point where false-accept == false-reject."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the baseline model.")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--ckpt", default=str(MODELS_DIR / "baseline_cnn.pt"))
    args = ap.parse_args()

    device = torch.device("cpu")
    _, val_ds = train_val_split(args.features, TRAIN.val_split, TRAIN.seed)
    if len(val_ds) == 0:
        raise SystemExit("Validation split is empty — add more data.")
    val_ld = DataLoader(val_ds, batch_size=32)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = BaselineCNN(n_classes=ckpt.get("n_classes", 2))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ys, preds, scores = [], [], []
    with torch.no_grad():
        for x, y in val_ld:
            probs = torch.softmax(model(x), dim=1)
            preds += probs.argmax(1).tolist()
            scores += probs[:, 1].tolist()   # P(fake)
            ys += y.tolist()

    ys = np.array(ys)
    preds = np.array(preds)
    scores = np.array(scores)

    acc = accuracy_score(ys, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        ys, preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(ys, preds, labels=[0, 1])
    eer = compute_eer(ys, scores) if len(set(ys.tolist())) > 1 else float("nan")

    print("=== AcousticSpace baseline evaluation ===")
    print(f"accuracy  = {acc:.3f}")
    print(f"precision = {p:.3f}   recall = {r:.3f}   f1 = {f1:.3f}")
    print(f"EER       = {eer:.3f}")
    print("confusion matrix  [rows = true, cols = pred]  (0=real, 1=fake):")
    print(cm)
    print(classification_report(ys, preds, target_names=["real", "fake"], zero_division=0))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["real", "fake"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["real", "fake"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("Confusion matrix")
        fig.tight_layout()
        out = MODELS_DIR / "confusion_matrix.png"
        fig.savefig(out)
        print(f"[eval] saved plot -> {out}")
    except Exception as exc:  # noqa: BLE001
        print(f"[eval] plot skipped: {exc}")


if __name__ == "__main__":
    main()
