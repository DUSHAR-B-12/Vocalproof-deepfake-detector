"""
VoiceShield — Ensemble Prediction System
Combines two trained Keras models via probability averaging.

Default: ensembles best_model.keras + final_model.keras from the latest run.

Usage:
    python -m inference.ensemble_predict <audio_file>
    python -m inference.ensemble_predict <audio_file> --model1 PATH --model2 PATH
"""

import sys
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Reuse the v2 feature extraction pipeline (log-mel spectrogram)
from training.feature_extraction import extract_features
from training.spec_augment import SpecAugment


# ── Default model paths (auto-discovered from config) ────────────
from training.config import MODEL_DIR

def _find_latest_run() -> Path:
    """Find the most recent run directory in MODEL_DIR."""
    if MODEL_DIR.exists():
        runs = sorted(
            [d for d in MODEL_DIR.iterdir() if d.is_dir()],
            reverse=True,
        )
        if runs:
            return runs[0]
    return MODEL_DIR

MODELS_DIR = _find_latest_run()
DEFAULT_MODEL1 = MODELS_DIR / "best_model.keras"
DEFAULT_MODEL2 = MODELS_DIR / "final_model.keras"


def load_model(path: Path, label: str = "model") -> tf.keras.Model:
    """Load a single Keras model with custom objects."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    print(f"  Loading {label}: {path.name}")
    return tf.keras.models.load_model(
        str(path),
        custom_objects={"SpecAugment": SpecAugment},
    )


def prepare_input(audio_path: str) -> np.ndarray:
    """
    Extract log-mel spectrogram and prepare batch.
    Returns: np.ndarray of shape (1, n_mels, T, 1)
    """
    spec = extract_features(audio_path)   # (n_mels, T, 1)
    return np.expand_dims(spec, axis=0)   # (1, n_mels, T, 1)


def ensemble_predict(audio_path: str,
                     models: list,
                     threshold: float = 0.5) -> dict:
    """
    Run ensemble prediction on a single audio file.

    Parameters
    ----------
    audio_path : path to .wav file
    models : list of loaded tf.keras.Model instances
    threshold : decision boundary

    Returns
    -------
    dict with per-model probabilities, ensemble probability, and label.
    """
    x = prepare_input(audio_path)

    probs = []
    for m in models:
        p = float(m.predict(x, verbose=0).flatten()[0])
        probs.append(p)

    final_prob = float(np.mean(probs))
    label = "FAKE" if final_prob >= threshold else "REAL"

    return {
        "probs": probs,
        "final_prob": final_prob,
        "label": label,
        "threshold": threshold,
        "audio": audio_path,
    }


def print_result(result: dict, model_names: list) -> None:
    """Pretty-print ensemble prediction result."""
    print(f"\n{'='*50}")
    print(f"  ENSEMBLE PREDICTION")
    print(f"{'='*50}")
    print(f"  Audio: {Path(result['audio']).name}")
    print(f"  ─────────────────────────────────")
    for name, p in zip(model_names, result["probs"]):
        print(f"  {name:20s}: {p:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  {'Ensemble prob':20s}: {result['final_prob']:.4f}")
    print(f"  {'Threshold':20s}: {result['threshold']:.2f}")
    print(f"  {'Predicted label':20s}: {result['label']}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(
        description="VoiceShield Ensemble Prediction (probability averaging)"
    )
    parser.add_argument("audio_path", type=str,
                        help="Path to a .wav audio file")
    parser.add_argument("--model1", type=str, default=str(DEFAULT_MODEL1),
                        help="Path to first Keras model (default: best_model)")
    parser.add_argument("--model2", type=str, default=str(DEFAULT_MODEL2),
                        help="Path to second Keras model (default: final_model)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()

    audio = Path(args.audio_path)
    if not audio.exists():
        print(f"Error: audio file not found: {audio}")
        sys.exit(1)

    print("Loading models...")
    m1_path = Path(args.model1)
    m2_path = Path(args.model2)
    m1 = load_model(m1_path, "Model 1 (best)")
    m2 = load_model(m2_path, "Model 2 (final)")

    result = ensemble_predict(
        audio_path=str(audio),
        models=[m1, m2],
        threshold=args.threshold,
    )
    print_result(result, model_names=[m1_path.stem, m2_path.stem])


if __name__ == "__main__":
    main()
