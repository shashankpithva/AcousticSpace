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
    breathing_cadence_alignment: str = Field(
        "unknown",
        description="good | moderate | poor | unknown",
    )

    breathing_cadence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
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


class AnalyzeResponse(BaseModel):
    filename: str
    duration_s: float

    prediction: str = Field(
        ...,
        description="deepfake | real | undetermined",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    model_stage: str

    reverb: ReverbFeatures

    key_indicators: KeyIndicators

    mel_shape: list[int]

    suspicious_segments: list[SuspiciousSegment]

    notes: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    sample_rate: int