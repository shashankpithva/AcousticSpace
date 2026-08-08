"""
Week 3 - AST Fine-Tuning (Basic Training Script)
"""

from pathlib import Path

import librosa
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
)

# Dataset folder
DATASET = Path("data/dataset")

# Hugging Face AST model
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Load feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)

# Load pretrained AST model
model = ASTForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,
)

model.train()


class AudioDataset(Dataset):

    def __init__(self):
        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:

            folder = DATASET / label_name

            if not folder.exists():
                continue

            for wav in folder.glob("*.wav"):

                self.samples.append(
                    {
                        "path": wav,
                        "label": label,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        audio, sr = librosa.load(
            sample["path"],
            sr=16000,
            mono=True,
        )

        inputs = feature_extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
        )

        return {
            "input_values": inputs["input_values"].squeeze(0),
            "label": sample["label"],
        }


if __name__ == "__main__":

    dataset = AudioDataset()

    print(f"Dataset size: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
    )

    epochs = 3

    for epoch in range(epochs):

        print(f"\nEpoch {epoch + 1}/{epochs}")

        total_loss = 0.0

        for batch in loader:

            optimizer.zero_grad()

            outputs = model(
                input_values=batch["input_values"],
                labels=batch["label"],
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            print(f"Batch Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)

        print(f"Average Loss: {avg_loss:.4f}")

    print("\nSaving model...")

    Path("models").mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/ast_model.pt",
    )

    print("Training completed successfully!")
    print("Model saved to models/ast_model.pt")