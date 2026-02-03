"""
VoiceShield - Feature extraction module.
Extracts MFCCs, pitch, jitter, and energy (RMS) per audio file; combines into a single vector.
"""

import numpy as np
import pandas as pd
import librosa
from pathlib import Path

from .preprocess import load_and_preprocess

TARGET_SR = 16000
N_MFCC = 13
FEATURES_CSV = "features/features.csv"


def _compute_pitch_and_jitter(waveform: np.ndarray, sr: int = TARGET_SR) -> tuple[float, float]:
    """
    Compute mean fundamental frequency (F0) and jitter (cycle-to-cycle instability).
    Uses Parselmouth (Praat) when available. If `parselmouth` is not installed,
    returns (0.0, 0.0) and prints a warning.
    """
    try:
        import parselmouth
    except Exception:
        print("parselmouth not available — returning zeros for pitch and jitter.")
        return 0.0, 0.0

    sound = parselmouth.Sound(waveform, sampling_frequency=sr)
    pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=600)
    f0_values = pitch.selected_array["frequency"]
    f0_voiced = f0_values[f0_values > 0]
    if len(f0_voiced) < 2:
        return 0.0, 0.0
    mean_pitch = float(np.mean(f0_voiced))
    # Jitter: relative variation in period (period = 1/F0)
    periods = 1.0 / f0_voiced
    mean_period = np.mean(periods)
    if mean_period <= 0:
        return mean_pitch, 0.0
    # Average absolute difference between consecutive periods, normalized
    jitter = np.mean(np.abs(np.diff(periods))) / mean_period
    return mean_pitch, float(jitter)


def _compute_energy_rms(waveform: np.ndarray) -> float:
    """Compute RMS energy of the waveform."""
    return float(np.sqrt(np.mean(waveform ** 2)))


def extract_features_from_waveform(waveform: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract a single feature vector from a preprocessed waveform.
    Features: 13 MFCC means, mean pitch, jitter, RMS energy.

    Returns:
        1D NumPy array of shape (16,) for one sample.
    """
    # MFCCs: mean across time for 13 coefficients
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    mfcc_means = np.mean(mfcc, axis=1)
    pitch, jitter = _compute_pitch_and_jitter(waveform, sr)
    energy = _compute_energy_rms(waveform)
    feature_vector = np.concatenate([mfcc_means, [pitch, jitter, energy]])
    return feature_vector.astype(np.float64)


def extract_features_from_file(audio_path: str) -> np.ndarray:
    """
    Load, preprocess, and extract features for a single audio file.
    Returns 1D feature vector of length 16.
    """
    waveform = load_and_preprocess(audio_path)
    return extract_features_from_waveform(waveform, TARGET_SR)


def extract_features(
    data_dir: str,
    output_csv: str = FEATURES_CSV,
    max_files: int | None = None,
) -> None:
    """
    Extract features from all supported audio files in data_dir.
    Saves combined features to output_csv (default: features/features.csv).

    Args:
        data_dir: Directory containing audio files (searched recursively).
        output_csv: Path for the output CSV.
        max_files: If set, process at most this many files (useful for large datasets).
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [
        str(p) for p in data_path.rglob("*")
        if p.suffix.lower() in exts and p.is_file()
    ]
    # If output CSV exists, read already-processed filepaths and skip them (resume support)
    processed_paths = set()
    out_path = Path(output_csv)
    if out_path.is_file():
        try:
            existing = pd.read_csv(out_path)
            if "filepath" in existing.columns:
                processed_paths = set(existing["filepath"].astype(str).tolist())
                print(f"Found existing features CSV with {len(processed_paths)} files; will skip them.")
        except Exception:
            # If reading fails, ignore and re-create CSV
            processed_paths = set()
    # Filter out already-processed files
    original_count = len(audio_files)
    audio_files = [p for p in audio_files if p not in processed_paths]
    if max_files is not None and len(audio_files) > max_files:
        audio_files = audio_files[:max_files]
        print(f"Processing first {max_files} files (use max_files=None for all).")
    if processed_paths:
        print(f"Skipping {original_count - len(audio_files)} already-processed files.")
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {data_dir}")

    feature_names = [f"mfcc_{i}" for i in range(N_MFCC)] + ["pitch", "jitter", "energy"]
    rows = []
    for path in audio_files:
        try:
            vec = extract_features_from_file(path)
            rows.append([path] + list(vec))
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

    new_df = pd.DataFrame(rows, columns=["filepath"] + feature_names)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If CSV existed, append new rows; otherwise write new CSV
    if out_path.is_file():
        try:
            existing = pd.read_csv(out_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_csv(out_path, index=False)
            print(f"Appended {len(new_df)} new files; total now {len(combined)} samples in {output_csv}")
        except Exception as e:
            # If append fails, overwrite with new dataframe
            new_df.to_csv(out_path, index=False)
            print(f"Failed to append to existing CSV (error: {e}). Wrote {len(new_df)} rows to {output_csv}")
    else:
        new_df.to_csv(out_path, index=False)
        print(f"Saved features for {len(new_df)} files to {output_csv}")


if __name__ == "__main__":
    import sys
    data_directory = sys.argv[1] if len(sys.argv) > 1 else "data"
    out = sys.argv[2] if len(sys.argv) > 2 else FEATURES_CSV
    max_f = int(sys.argv[3]) if len(sys.argv) > 3 else None
    extract_features(data_directory, out, max_files=max_f)
