"""AcousticSpace FastAPI server (Week 1 + Week 2).

Week 2 adds baseline model inference: if a trained checkpoint exists under
backend/models/, /analyze returns a real real-vs-deepfake prediction. If not,
it falls back to the Week 1 feature-only 'undetermined' response so the API
keeps working before the model is trained.
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded lazily on startup; stays None if torch / a checkpoint is unavailable.
_predictor = None


@app.on_event("startup")
def _load_model() -> None:
    global _predictor
    try:
        from ml.infer import get_predictor

        _predictor = get_predictor()
        if _predictor is not None:
            print(f"[startup] baseline model loaded (val_acc={_predictor.val_acc}).")
        else:
            print("[startup] no trained checkpoint found — feature-only mode.")
    except Exception as exc:  # noqa: BLE001
        _predictor = None
        print(f"[startup] model load skipped ({exc}) — feature-only mode.")


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

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        waveform = preprocess(tmp_path)
        feats = extract_all(waveform)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not process audio: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    reverb = ReverbFeatures(**feats["reverb"])
    mel = feats["mel_spectrogram"]

    if _predictor is not None:
        out = _predictor.predict(waveform)
        prediction = str(out["prediction"])
        confidence = float(out["confidence"])
        model_stage = "week2-baseline-cnn"
        notes = (
            "Week 2 baseline CNN prediction from Mel-spectrogram features. "
            "Key indicators (RIR mismatch, breathing, cadence) arrive in Week 3."
        )
    else:
        prediction = "undetermined"
        confidence = 0.0
        model_stage = "week1-feature-extraction-only"
        notes = (
            "No trained model found. Train the baseline "
            "(see backend/ml/README.md) to enable predictions."
        )

    return AnalyzeResponse(
        filename=file.filename or "unknown",
        duration_s=float(feats["duration_s"]),
        prediction=prediction,
        confidence=confidence,
        model_stage=model_stage,
        reverb=reverb,
        key_indicators=KeyIndicators(),  # 'unknown' until Week 3
        mel_shape=list(mel.shape),
        notes=notes,
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AcousticSpace API. See /docs for the interactive API."}
