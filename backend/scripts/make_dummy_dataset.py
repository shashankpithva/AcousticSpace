"""Generate a tiny SYNTHETIC real/fake dataset to smoke-test the Week 2 pipeline.

This lets you verify training end-to-end BEFORE downloading the real ASVspoof
corpus. 'real' clips get a synthetic room-reverb tail (physical echo); 'fake'
clips are dry with a mild artifact. They are trivially separable — this is only
to prove the training loop runs. Do NOT report accuracy on this toy data.

Usage (from backend/):
    python scripts/make_dummy_dataset.py --out data/dataset --n 40
    python scripts/extract_features.py --dataset data/dataset --out data/features
    python -m ml.train --features data/features --epochs 10
"""
from __future__ import annotations

import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from app.config import AUDIO


def make_tone(dur: float, sr: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.zeros_like(t)
    for k, amp in enumerate([1.0, 0.5, 0.25], start=1):
        sig += amp * np.sin(2 * np.pi * f0 * k * t)
    sig += 0.01 * rng.standard_normal(t.shape)
    return sig.astype(np.float32)


def add_reverb(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    ir_len = int(0.4 * sr)
    ir = (rng.standard_normal(ir_len) * np.exp(-np.linspace(0, 6, ir_len))).astype(np.float32)
    ir[0] = 1.0
    y = np.convolve(x, ir)[: len(x)]
    return (y / (np.max(np.abs(y)) + 1e-8)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a synthetic smoke-test dataset.")
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--n", type=int, default=40, help="clips per class")
    args = ap.parse_args()

    sr, dur = AUDIO.sample_rate, AUDIO.target_seconds
    out = pathlib.Path(args.out)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for i in range(args.n):
        f0 = float(rng.uniform(90, 220))
        base = make_tone(dur, sr, f0, rng)

        real = add_reverb(base, sr, rng)
        sf.write(out / "real" / f"real_{i:03d}.wav", real, sr)

        dry = base / (np.max(np.abs(base)) + 1e-8)
        dry = np.clip(dry + 0.02 * np.sign(dry), -1.0, 1.0)  # mild artifact
        sf.write(out / "fake" / f"fake_{i:03d}.wav", dry.astype(np.float32), sr)

    print(f"[dummy] wrote {args.n} real + {args.n} fake clips -> {out}")
    print("[dummy] next: python scripts/extract_features.py --dataset data/dataset --out data/features")


if __name__ == "__main__":
    main()
