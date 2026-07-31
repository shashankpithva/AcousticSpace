"""
AcousticSpace FastAPI Server - Week 3
AST + breathing + suspicious segment detection
"""

from __future__ import annotations
from .attention import generate_attention
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .audio_pipeline import preprocess
from .ast_predict import predict_audio
from .breathing import analyze_breathing
from .config import ALLOWED_EXTENSIONS, AUDIO
from .features import extract_all
from .segments import find_suspicious_segments

from .schemas import (
    AnalyzeResponse,
    HealthResponse,
    KeyIndicators,
    ReverbFeatures,
)


app = FastAPI(
    title="AcousticSpace API",
    description="Deepfake audio detection using AST",
    version=__version__,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():

    return {
        "message": "AcousticSpace API is running"
    }



@app.get("/health", response_model=HealthResponse)
def health():

    return HealthResponse(
        status="ok",
        service="acousticspace",
        version=__version__,
        sample_rate=AUDIO.sample_rate,
    )



@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):

    ext = Path(file.filename or "").suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}",
        )


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=ext,
    ) as tmp:

        shutil.copyfileobj(
            file.file,
            tmp
        )

        tmp_path = Path(tmp.name)


    try:

        print("STEP 1 preprocess")

        y = preprocess(tmp_path)


        print("STEP 2 features")

        feats = extract_all(y)


        reverb = ReverbFeatures(
            **feats["reverb"]
        )


        mel = feats["mel_spectrogram"]


        print("STEP 3 AST")

        prediction = predict_audio(
            tmp_path
        )


        print("STEP 4 breathing")

        breathing = analyze_breathing(
            tmp_path
        )


        print("STEP 5 segments")

        segments = find_suspicious_segments(
            tmp_path
        )


        print("DONE")


        return AnalyzeResponse(

            filename=file.filename or "unknown",

            duration_s=float(
                feats["duration_s"]
            ),

            prediction=prediction["prediction"],

            confidence=prediction["confidence"],

            model_stage="week3-ast",

            reverb=reverb,


            key_indicators=KeyIndicators(

                breathing_pattern=
                breathing["pattern"]

            ),


            mel_shape=list(
                mel.shape
            ),


            suspicious_segments=segments,


            notes=
            "Week 3 AST model + breathing analysis + suspicious segment detection."

        )


    except Exception as exc:

        print("ERROR:", exc)

        raise HTTPException(

            status_code=500,

            detail=str(exc)

        )


    finally:

        tmp_path.unlink(
            missing_ok=True
        )
