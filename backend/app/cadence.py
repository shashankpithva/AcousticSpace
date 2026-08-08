import librosa
import numpy as np


def analyze_cadence(audio_path):
    """
    Estimate vocal cadence using onset detection.
    """

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        units="frames",
    )

    speech_onsets = librosa.frames_to_time(
        onset_frames,
        sr=sr,
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=2048,
        hop_length=512,
    )[0]

    variation = float(np.std(rms))

    if len(speech_onsets) >= 2:
        intervals = np.diff(speech_onsets)
        cadence_variation = float(np.std(intervals))

        if cadence_variation < 0.15:
            pattern = "natural"
        else:
            pattern = "irregular"
    else:
        pattern = "irregular"

    return {
        "pattern": pattern,
        "variation": variation,
        "speech_onsets": speech_onsets.tolist(),
    }