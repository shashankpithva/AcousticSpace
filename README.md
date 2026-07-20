# AcousticSpace — Deepfake Detection via Room Impulse Response (RIR)

> **Week 1 deliverable: Core Setup & Data.**
> A working FastAPI server, a Librosa audio-processing pipeline that extracts spectrograms + RIR/reverb features, an ASVspoof dataset organizer, and a React + TypeScript analyst dashboard scaffold.

AcousticSpace detects synthetic (deepfake) audio by analyzing **physics** — the Room Impulse Response (how sound reflects off walls) and the speaker's breathing patterns — instead of relying only on vocal biometrics. When the acoustic reflection of a generated voice does not match its claimed background environment, the clip is flagged.

---

## Repository layout

```
acousticspace/
├── backend/                 # Python · FastAPI · Librosa · PyTorch (Week 1: setup + features)
│   ├── app/
│   │   ├── main.py          # FastAPI server: /health and /analyze
│   │   ├── config.py        # Central audio + path config
│   │   ├── audio_pipeline.py# Load → resample → mono → normalize → trim silence
│   │   ├── features.py      # Mel spectrogram, MFCC, RIR / reverb features
│   │   └── schemas.py       # Pydantic response models
│   ├── scripts/
│   │   ├── prepare_dataset.py  # Organize the ASVspoof dataset into real/ fake/
│   │   └── extract_features.py # Batch-extract features and cache to disk (.npz)
│   ├── requirements.txt
│   └── README.md
└── frontend/                # React · TypeScript · Vite (Week 1: upload + static dashboard)
    ├── src/
    │   ├── components/
    │   │   ├── AudioUpload.tsx
    │   │   ├── WaveformPanel.tsx
    │   │   ├── SpectrogramPanel.tsx
    │   │   └── ResultsPanel.tsx
    │   ├── api.ts
    │   ├── App.tsx
    │   ├── main.tsx
    │   └── styles.css
    ├── index.html
    ├── package.json
    ├── tsconfig.json
    └── vite.config.ts
```

---

## Quick start

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.
Health check: http://localhost:8000/health

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the printed URL (default http://localhost:5173). The dashboard proxies `/api` calls to the backend on port 8000.

### 3. Dataset (ASVspoof)

Download the ASVspoof 2019 LA dataset (or any real/fake corpus), then:

```bash
cd backend
python scripts/prepare_dataset.py --src /path/to/asvspoof --out data/dataset
python scripts/extract_features.py --dataset data/dataset --out data/features
```

---

## Week 1 checklist (this maps to the project plan)

**Backend & ML**
- [x] FastAPI server with a health-check route
- [x] Librosa pipeline: load, resample, mono, normalize, trim silence
- [x] Mel spectrogram + log scaling
- [x] RIR / reverb feature extraction
- [x] Save extracted features to disk (`.npz`) for reuse
- [x] ASVspoof dataset organizer

**Frontend**
- [x] React + TypeScript app (Vite)
- [x] Audio upload component (drag & drop / browse; .wav .mp3 .flac)
- [x] Static dashboard layout (upload, waveform, spectrogram, results)

> ⚠️ Week 1 intentionally has **no trained model yet**. `/analyze` runs the real feature pipeline and returns extracted feature summaries with a clearly-labeled placeholder prediction. The actual classifier arrives in Week 2 (baseline) and Week 3 (fine-tuned AST).

---

## 📅 Week 2 — Baseline Model (DONE)

**Backend & ML**
- [x] `BaselineCNN` — CNN classifier on Mel-spectrogram features (`backend/ml/model.py`)
- [x] PyTorch dataset + stratified train/val split over cached `.npz` (`backend/ml/dataset.py`)
- [x] Training loop with metrics logging + best-checkpoint saving (`backend/ml/train.py`)
- [x] Evaluation: accuracy, precision/recall/F1, confusion matrix, EER (`backend/ml/evaluate.py`)
- [x] First real deepfake prediction wired into `/analyze` (`backend/ml/infer.py` + `app/main.py`)
- [x] Synthetic smoke-test dataset generator (`backend/scripts/make_dummy_dataset.py`)

**Frontend**
- [x] Full Wavesurfer.js waveform + timeline (`frontend/src/components/WaveformPanel.tsx`)
- [x] Audio playback controls, live time readout, loading state, zoom

### Run Week 2 (from `backend/`)

```bash
# quick smoke test without downloading ASVspoof:
python scripts/make_dummy_dataset.py --out data/dataset --n 40
python scripts/extract_features.py --dataset data/dataset --out data/features
python -m ml.train --features data/features --epochs 10
python -m ml.evaluate --features data/features

# then restart the API so it picks up models/baseline_cnn.pt
uvicorn app.main:app --reload --port 8000
```

After training, `/analyze` returns `authentic` / `deepfake` + a confidence score
instead of `undetermined`. See `backend/ml/README.md` for full details.
