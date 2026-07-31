"""
Pydantic response models for the AcousticSpace API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReverbFeatures(BaseModel):
    rt60_estimate_s: float
    drr_db: float
    spectral_flatness: float
    spectral_centroid_hz: float
    zero_crossing_rate: float


class KeyIndicators(BaseModel):
    rir_mismatch: str = Field(
        "unknown",
        description="high | medium | low | unknown",
    )

    reverb_consistency: str = Field(
        "unknown",
        description="consistent | inconsistent | unknown",
    )

    breathing_pattern: str = Field(
        "unknown",
        description="consistent | irregular | unknown",
    )

    vocal_cadence: str = Field(
        "unknown",
        description="natural | irregular | unknown",
    )


class SuspiciousSegment(BaseModel):
    start: float
    end: float
    fake_probability: float


from transformers import ASTFeatureExtractor
import librosa
import torch

from .ast_predict import model

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)


def find_suspicious_segments(audio_path, window_seconds=1.0):
    """
    Split audio into 1-second windows and compute
    fake probability for each window.
    """

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    window_samples = int(window_seconds * sr)

    segments = []

    start = 0

    while start < len(audio):

        end = min(start + window_samples, len(audio))

        clip = audio[start:end]

        # Ignore windows shorter than 1 second
        if len(clip) < window_samples:
            break

        inputs = feature_extractor(
            clip,
            sampling_rate=sr,
            return_tensors="pt",
        )

        with torch.no_grad():

            outputs = model(
                input_values=inputs["input_values"]
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1,
            )

            fake_probability = float(
                probabilities[0][1]
            )

        segments.append(
            {
                "start": round(start / sr, 2),
                "end": round(end / sr, 2),
                "fake_probability": round(fake_probability, 3),
            }
        )

        start += window_samples

    return segments