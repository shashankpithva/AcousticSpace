import librosa
import numpy as np


def analyze_cadence(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    # Detect energy changes
    rms = librosa.feature.rms(y=y)[0]

    variation = np.std(rms)

    if variation < 0.01:
        pattern = "irregular"
    else:
        pattern = "natural"

    return {
        "pattern": pattern,
        "variation": float(variation)
    }