from pathlib import Path

import librosa
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_PATH = Path("models/ast_model.pt")

feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

model = ASTForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

if MODEL_PATH.exists():
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
        )
    )

model.eval()


def predict_audio(audio_path):

    audio, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True,
    )

    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    )

    with torch.no_grad():

        outputs = model(
            input_values=inputs["input_values"],
        )

        probabilities = torch.softmax(outputs.logits, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1,
        )

    label = "real" if prediction.item() == 0 else "fake"

    return {
        "prediction": label,
        "confidence": float(confidence.item()),
    }