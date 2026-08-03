# AcousticSpace — ML (Week 2 baseline model)

A small CNN that classifies audio as **real** vs **deepfake** from the cached
log-Mel spectrogram features produced in Week 1.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | `BaselineCNN` — 3 conv blocks → global pool → MLP head |
| `dataset.py` | `FeatureDataset` + stratified `train_val_split` over `.npz` files |
| `train.py` | Training loop, metrics logging, best-checkpoint saving |
| `evaluate.py` | Accuracy, precision/recall/F1, confusion matrix, **EER** |
| `infer.py` | Loads the checkpoint and predicts for the API |
| `config.py` | Hyperparameters + class mapping (0=real, 1=fake) |

## Full workflow (from the `backend/` directory)

```bash
# Option A — real data
python scripts/prepare_dataset.py --src /path/to/asvspoof --out data/dataset
python scripts/extract_features.py --dataset data/dataset --out data/features

# Option B — quick smoke test with synthetic data (no download needed)
python scripts/make_dummy_dataset.py --out data/dataset --n 40
python scripts/extract_features.py --dataset data/dataset --out data/features

# Train + evaluate (same for either option)
python -m ml.train --features data/features --epochs 15
python -m ml.evaluate --features data/features
```

Outputs land in `backend/models/`:
- `baseline_cnn.pt` — best checkpoint (auto-loaded by the API on next restart)
- `metrics.json` — per-epoch train/val loss & accuracy
- `confusion_matrix.png` — evaluation plot

## Wiring into the API

Once `models/baseline_cnn.pt` exists, **restart uvicorn**. On startup the server
prints `baseline model loaded (...)` and `/analyze` returns a real prediction
(`authentic` / `deepfake`) with a confidence score instead of `undetermined`.

## Device support

Training auto-selects CUDA → Apple **MPS** → CPU, so it runs on a Mac laptop.

## Notes

- **EER (Equal Error Rate)** is the headline metric for spoof detection — lower is better.
- This is a *baseline*. Week 3 replaces/augments it with a fine-tuned HuggingFace
  Audio Spectrogram Transformer (AST) plus breathing-cadence analysis.
