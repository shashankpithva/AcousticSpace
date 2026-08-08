import librosa
import numpy as np


def analyze_breathing(audio_path):
    """
    Simple breathing analysis using RMS energy.
    """

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    rms = librosa.feature.rms(y=audio)[0]

    threshold = np.mean(rms) * 0.4

    silent_frames = rms < threshold

    breathing_events = int(np.sum(silent_frames))

    if breathing_events > 20:
        pattern = "consistent"
    else:
        pattern = "irregular"

    return {
        "pattern": pattern,
        "breaths": breathing_events,
    }