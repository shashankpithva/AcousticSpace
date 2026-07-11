"""Batch-extract and cache features for a prepared dataset (Week 1).

Reads data/dataset/{real,fake}/*.{wav,flac,mp3}, runs the preprocessing +
feature pipeline, and writes one compressed .npz per clip plus an index CSV.
These cached features feed the Week 2 baseline model without re-decoding audio.

Usage:
    python scripts/extract_features.py --dataset data/dataset --out data/features
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow running as a plain script (python scripts/extract_features.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.audio_pipeline import preprocess  # noqa: E402
from app.features import extract_all         # noqa: E402

AUDIO_EXTS = {".wav", ".flac", ".mp3"}


def iter_clips(dataset: Path):
    for label in ("real", "fake"):
        folder = dataset / label
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in AUDIO_EXTS:
                yield label, path


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract + cache acoustic features.")
    ap.add_argument("--dataset", default=Path("data/dataset"), type=Path)
    ap.add_argument("--out", default=Path("data/features"), type=Path)
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    args = ap.parse_args()

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.csv"

    n = 0
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["npz", "label", "source", "duration_s",
             "rt60_estimate_s", "drr_db", "spectral_flatness",
             "spectral_centroid_hz", "zero_crossing_rate"]
        )
        for label, path in iter_clips(args.dataset):
            if args.limit and n >= args.limit:
                break
            try:
                y = preprocess(path)
                feats = extract_all(y)
            except Exception as exc:  # noqa: BLE001
                print(f"[skip] {path.name}: {exc}")
                continue

            npz_name = f"{label}__{path.stem}.npz"
            np.savez_compressed(
                out / npz_name,
                mel=feats["mel_spectrogram"].astype(np.float32),
                mfcc=feats["mfcc"].astype(np.float32),
                label=np.int64(1 if label == "fake" else 0),
            )
            r = feats["reverb"]
            writer.writerow([
                npz_name, label, str(path), feats["duration_s"],
                r["rt60_estimate_s"], r["drr_db"], r["spectral_flatness"],
                r["spectral_centroid_hz"], r["zero_crossing_rate"],
            ])
            n += 1
            if n % 50 == 0:
                print(f"[extract] {n} clips processed...")

    print(f"[extract] Done. {n} clips -> {out}  (index: {index_path})")


if __name__ == "__main__":
    main()
