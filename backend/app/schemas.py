"""Pydantic response models for the AcousticSpace API.

The /analyze response shape is designed to match the final dashboard
contract (prediction, confidence, and the four key indicators) so the
frontend does not need to change when the real model lands in Week 3.
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
    """The four analyst-facing signals shown on the dashboard."""
    rir_mismatch: str = Field("unknown", description="high | medium | low | unknown")
    reverb_consistency: str = Field("unknown", description="consistent | inconsistent | unknown")
    breathing_pattern: str = Field("unknown", description="consistent | irregular | unknown")
    vocal_cadence: str = Field("unknown", description="natural | irregular | unknown")


class AnalyzeResponse(BaseModel):
    filename: str
    duration_s: float
    prediction: str = Field(..., description="deepfake | authentic | undetermined")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_stage: str = Field(..., description="Which pipeline stage produced this result")
    reverb: ReverbFeatures
    key_indicators: KeyIndicators
    mel_shape: list[int]
    notes: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    sample_rate: int
