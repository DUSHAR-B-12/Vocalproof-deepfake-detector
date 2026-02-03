"""
Run-resume pipeline:
- Extract features from audio in data/real (skips files already present in features CSV)
- Train the model

Usage: run from the ml-service root:
    python src/run_resume_pipeline.py [data_dir]
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.extract_features import extract_features, FEATURES_CSV
from src.train_model import train_model, MODEL_SAVE_PATH


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/real")
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}")
        return

    print(f"Extracting features from audio in {data_dir}...")
    print("(librosa will load MP3, FLAC, OGG, WAV, M4A formats directly)")
    extract_features(str(data_dir), FEATURES_CSV)

    print("\nTraining model on extracted features...")
    train_model(FEATURES_CSV, MODEL_SAVE_PATH)

    print("\n=== DONE ===")
    print(f"Model and scaler saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
