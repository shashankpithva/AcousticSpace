# AcousticSpace — Backend (Week 1)

Python · FastAPI · Librosa. Provides audio preprocessing, RIR/reverb + spectrogram
feature extraction, and an inference API scaffold.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Liveness + config echo |
| POST | `/analyze` | Upload audio → extract features → return JSON |

### Example

```bash
curl -F "file=@sample.wav" http://localhost:8000/analyze
```

## Dataset workflow

```bash
# 1. Organize raw ASVspoof into real/ and fake/
python scripts/prepare_dataset.py --src /path/to/asvspoof --out data/dataset

# 2. Extract + cache features (.npz per clip + index.csv)
python scripts/extract_features.py --dataset data/dataset --out data/features
```

## Module map

- `app/config.py` — all audio constants (sample rate, FFT, Mel, MFCC).
- `app/audio_pipeline.py` — load → resample → mono → normalize → trim → fix length.
- `app/features.py` — Mel spectrogram, MFCC, RT60, DRR, reverb descriptors.
- `app/schemas.py` — Pydantic response models (stable API contract).
- `app/main.py` — FastAPI app.

## What comes next

- **Week 2:** train a CNN/Transformer baseline on the cached `.npz` features.
- **Week 3:** fine-tune the HuggingFace Audio Spectrogram Transformer (AST);
  add breathing-cadence alignment.
- **Week 4:** Dockerize, optimize latency, CI/CD.
