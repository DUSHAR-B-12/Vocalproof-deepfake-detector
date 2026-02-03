"""
Run-resume pipeline:
- Convert audio in data/real to mono 16kHz WAVs (skips already-converted files)
- Extract features (skips files already present in features CSV)
- Train the model

Usage: run from the ml-service root:
    python src/run_resume_pipeline.py [data_dir]
"""
import sys
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.extract_features import extract_features, FEATURES_CSV
from src.train_model import train_model, MODEL_SAVE_PATH


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def convert_to_wav(source_dir: Path, target_dir: Path, sr: int = 16000) -> None:
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [p for p in source_dir.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    if not files:
        print(f"No audio files found in {source_dir}")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        rel = src.relative_to(source_dir)
        out_path = target_dir.joinpath(rel).with_suffix('.wav')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.is_file():
            print(f"Skipping existing: {out_path}")
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            str(sr),
            "-ac",
            "1",
            str(out_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Converted: {src} -> {out_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {src}: {e}")


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/real")
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}")
        return

    if not check_ffmpeg():
        print("ffmpeg not found on PATH. Please install ffmpeg to enable format conversion.")
        print("You can still run feature extraction directly if librosa supports your formats.")

    converted_dir = Path("data/real_wav")
    print(f"Converting audio from {data_dir} -> {converted_dir} (wav 16kHz mono)")
    convert_to_wav(data_dir, converted_dir)

    print("Extracting features (will skip already-processed files)...")
    extract_features(str(converted_dir), FEATURES_CSV)

    print("Training model on extracted features...")
    train_model(FEATURES_CSV, MODEL_SAVE_PATH)

    print("Done. Model and scaler saved to:")
    print(MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
