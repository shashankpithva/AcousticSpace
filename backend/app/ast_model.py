"""
Week 3 - Hugging Face Audio Spectrogram Transformer (AST)
"""

from transformers import ASTForAudioClassification, ASTFeatureExtractor
import torch

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Load pretrained feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

# Load pretrained AST model
model = ASTForAudioClassification.from_pretrained(MODEL_NAME)

model.eval()


def predict(audio, sample_rate=16000):
    """
    Predict using the pretrained AST model.

    Parameters
    ----------
    audio : numpy.ndarray
        Audio waveform.
    sample_rate : int
        Sampling rate.

    Returns
    -------
    logits
    """

    inputs = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.logits