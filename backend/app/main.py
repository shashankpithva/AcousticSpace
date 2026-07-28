"""AcousticSpace FastAPI server (Week 1).

Endpoints:
  GET  /health   -> liveness + config echo
  POST /analyze  -> upload an audio file, run the real feature pipeline,
                    return extracted features + a clearly-labeled placeholder
                    prediction (no trained model until Week 2+).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .audio_pipeline import preprocess
from .config import ALLOWED_EXTENSIONS, AUDIO
from .features import extract_all
from .model import AudioCNN, predict
import torch
from .schemas import (
    AnalyzeResponse,
    HealthResponse,
    KeyIndicators,
    ReverbFeatures,
)

app = FastAPI(
    title="AcousticSpace API",
    description="Deepfake audio detection via Room Impulse Response (RIR).",
    version=__version__,
)
MODEL_PATH = Path("models/baseline_cnn.pt")

model = AudioCNN()

if MODEL_PATH.exists():
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )
    model.eval()

# Allow the Vite dev server to call the API during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="acousticspace",
        version=__version__,
        sample_rate=AUDIO.sample_rate,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Persist the upload to a temp file so librosa/soundfile can read it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        y = preprocess(tmp_path)
        feats = extract_all(y)
    except Exception as exc:  # noqa: BLE001 - surface a clean 422 to the client
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    reverb = ReverbFeatures(**feats["reverb"])
    mel = feats["mel_spectrogram"]

    # NOTE: No trained classifier yet (arrives Week 2 baseline / Week 3 AST).
    # We return an honest 'undetermined' verdict so the UI can be built and
    # tested end-to-end against a real response shape.
    prediction, confidence = predict(
        model,
        feats["mel_spectrogram"]
    )

    return AnalyzeResponse(
        filename=file.filename or "unknown",
        duration_s=float(feats["duration_s"]),
        prediction=prediction,
        confidence=confidence,
        model_stage="week2-cnn-baseline",
        reverb=reverb,
        key_indicators=KeyIndicators(),
        mel_shape=list(mel.shape),
        notes="Week 2 build: CNN baseline model prediction active.",
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AcousticSpace API. See /docs for the interactive API."}
