"""Organize a raw ASVspoof (or similar) corpus into real/ and fake/ folders.

Usage:
    python scripts/prepare_dataset.py --src /path/to/asvspoof --out data/dataset

ASVspoof 2019 LA ships a protocol .txt where each line looks like:
    LA_0079 LA_D_1047731 - - bonafide
    LA_0079 LA_D_1105538 - A01 spoof
The 2nd column is the file id (<id>.flac) and the last column is the label
('bonafide' = real, 'spoof' = fake). This script copies each file into
data/dataset/real or data/dataset/fake accordingly.

If no protocol file is found, it falls back to scanning any nested folders
whose names contain 'real'/'bonafide' or 'fake'/'spoof'.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

AUDIO_EXTS = {".flac", ".wav", ".mp3"}


def find_protocol(src: Path) -> Path | None:
    candidates = list(src.rglob("*cm_protocols*/*.txt")) + list(src.rglob("*protocol*.txt"))
    return candidates[0] if candidates else None


def index_audio(src: Path) -> dict[str, Path]:
    """Map file-id (stem) -> path for every audio file under src."""
    index: dict[str, Path] = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            index[p.stem] = p
    return index


def copy_via_protocol(protocol: Path, index: dict[str, Path], out: Path) -> tuple[int, int]:
    real_n = fake_n = 0
    for line in protocol.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        file_id, label = parts[1], parts[-1].lower()
        path = index.get(file_id)
        if path is None:
            continue
        if label in {"bonafide", "real", "genuine"}:
            dest_dir, real_n = out / "real", real_n + 1
        elif label in {"spoof", "fake"}:
            dest_dir, fake_n = out / "fake", fake_n + 1
        else:
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / path.name)
    return real_n, fake_n


def copy_via_folders(src: Path, out: Path) -> tuple[int, int]:
    real_n = fake_n = 0
    for p in src.rglob("*"):
        if p.suffix.lower() not in AUDIO_EXTS:
            continue
        lowered = str(p).lower()
        if any(k in lowered for k in ("real", "bonafide", "genuine")):
            dest_dir, real_n = out / "real", real_n + 1
        elif any(k in lowered for k in ("fake", "spoof")):
            dest_dir, fake_n = out / "fake", fake_n + 1
        else:
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest_dir / p.name)
    return real_n, fake_n


def main() -> None:
    ap = argparse.ArgumentParser(description="Organize ASVspoof into real/ and fake/.")
    ap.add_argument("--src", required=True, type=Path, help="Path to the raw dataset root")
    ap.add_argument("--out", default=Path("data/dataset"), type=Path, help="Output folder")
    args = ap.parse_args()

    src, out = args.src.expanduser(), args.out.expanduser()
    if not src.exists():
        raise SystemExit(f"Source folder not found: {src}")
    out.mkdir(parents=True, exist_ok=True)

    protocol = find_protocol(src)
    if protocol:
        print(f"[prepare] Using protocol file: {protocol}")
        index = index_audio(src)
        real_n, fake_n = copy_via_protocol(protocol, index, out)
    else:
        print("[prepare] No protocol file found — falling back to folder-name heuristics.")
        real_n, fake_n = copy_via_folders(src, out)

    print(f"[prepare] Done. real={real_n}  fake={fake_n}  -> {out}")
    if real_n == 0 and fake_n == 0:
        print("[prepare] WARNING: nothing copied. Check the dataset layout.")


if __name__ == "__main__":
    main()
