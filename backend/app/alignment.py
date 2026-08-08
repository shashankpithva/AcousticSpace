import numpy as np


def calculate_alignment(breathing, cadence):
    """
    Compare estimated breathing activity with speech cadence.

    This is a heuristic alignment score, not a medical measurement.
    """

    speech_onsets = np.asarray(
        cadence.get("speech_onsets", []),
        dtype=float,
    )

    breaths = float(
        breathing.get("breaths", 0)
    )

    if len(speech_onsets) == 0:
        return {
            "alignment": "unknown",
            "alignment_score": 0.0,
        }

    # If no breathing events were detected, we cannot
    # establish strong alignment evidence.
    if breaths <= 0:
        return {
            "alignment": "poor",
            "alignment_score": 0.2,
        }

    # Compare the number of detected breathing events
    # with detected speech activity.
    speech_count = len(speech_onsets)

    ratio = breaths / max(speech_count, 1)

    # A moderate breathing-to-speech ratio is treated
    # as better aligned.
    target_ratio = 5.0

    score = 1.0 / (
        1.0
        + abs(
            np.log1p(ratio)
            - np.log1p(target_ratio)
        )
    )

    score = float(
        np.clip(score, 0.0, 1.0)
    )

    if score >= 0.7:
        alignment = "good"
    elif score >= 0.4:
        alignment = "moderate"
    else:
        alignment = "poor"

    return {
        "alignment": alignment,
        "alignment_score": round(score, 3),
    }